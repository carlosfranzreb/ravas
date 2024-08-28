import base64
import json
import logging
import os
import time
from queue import Empty
from threading import Event
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch.multiprocessing import Queue, Process, Event
from websocket_server import WebsocketServer

from .chrome_runner import start_browser
from .web_server import start_server
from ...processor import Converter
from ...utils import clear_queue


class Avatar(Converter):
    def __init__(
        self,
        name: str,
        config: dict,
        input_queue: Queue,
        output_queue: Queue,
        log_queue: Queue,
        log_level: str,
        ready_signal: Event,
    ) -> None:
        """
        Initialize the Avatar Model.
        """
        super().__init__(name, config, input_queue, output_queue, log_queue, log_level, ready_signal)
        self.log_queue = log_queue
        self.log_level = log_level
        self._stopped = False

    def initializeFaceLandmarkerModel(self):
        """
        Initialize the face landmarker model.
        """

        current_file_path = os.path.dirname(os.path.abspath(__file__))
        face_landmarker_path = os.path.join(current_file_path, "face_landmarker.task")
        base_options = mp.tasks.BaseOptions(
            model_asset_path=face_landmarker_path,
        )
        mp_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(mp_options)

    def initializeRenderer(self):

        if not self.config.get('start_chrome_renderer', True):
            self.logger.info('Disabled automated start of Chrome driver for rendering avatar.')
            return

        self.render_server: Optional[Process] = None
        self.render_app: Optional[Process] = None
        self.stop_render_app: Optional[Queue] = None

        app_port = int(self.config.get('app_port', 3000))
        use_extension = self.config.get('use_chrome_extension', True)
        if not use_extension:
            server_args = {
                'port': app_port,
                'log_queue': self.log_queue,
                'log_level': self.log_level,
            }
            render_app_server = Process(target=start_server, kwargs=server_args, name='render_app_server')
            if not self._stopped:
                render_app_server.start()
                self.logger.info('Started web Server for Rendering App (pid %s)', render_app_server.pid)
            else:
                self.logger.info('Did not started web Server for Rendering App: already stopped!')
        else:
            render_app_server = None

        ws_port = int(self.config.get('ws_port', 8888))
        render_app_stop = Queue()  # NOTE: Event() is not pickable for sub-processes, so use Queue for sending stop signal
        app_args = {
            'ws_addr': 'http://127.0.0.1:{}'.format(ws_port),
            'stop_signal': render_app_stop,
            'port': app_port,
            'web_extension': use_extension,
            'run_headless': not self.config.get('show_chrome_window', False),
            'avatar_uri': self.config.get('avatar_uri', None),
            'log_queue': self.log_queue,
            'log_level': self.log_level,
        }
        render_app = Process(target=start_browser, kwargs=app_args, name='render_app')
        if not self._stopped:
            render_app.start()
            self.logger.info('Started Chrome Driver for Rendering App (pid %s)', render_app.pid)

        if not self._stopped:
            self.render_server = render_app_server
            self.render_app = render_app
            self.stop_render_app = render_app_stop
        else:
            self.stopRenderer(render_app=render_app, stop_render_app=render_app_stop, render_server=render_app_server)

    def stopRenderer(self, render_app: Optional[Process] = None, stop_render_app: Optional[Queue] = None, render_server: Optional[Process] = None):
        self._stopped = True

        # NOTE: usually, we do not really need to wait for rendering-app-process to shut down completely
        #       (i.e. only initiate its shutdown, and then leave it to its clean-up etc.)
        is_wait_for_render_app_to_finish = False  # TODO make this configurable?

        if not render_app:
            render_app = self.render_app
        if not render_server:
            render_server = self.render_server
        if not stop_render_app:
            stop_render_app = self.stop_render_app

        if render_server and render_server.is_alive():
            self.logger.info('Stopping Web Server Rendering App...')
            render_server.terminate()

        if render_app and render_app.is_alive():
            self.logger.info('Stopping Chrome Driver for Rendering App...')
            # NOTE need to signal the render_app process to stop, so that the chrome driver can be closed properly
            #      (simply calling render_app.terminate() will leave chrome instance running)
            stop_render_app.put(True)  # send any value for signaling the render_app to stop the chrome driver

            if is_wait_for_render_app_to_finish:
                start = time.time()
                for i in range(10):
                    render_app.join(1)
                    self.logger.info('Waiting for Chrome Driver (Rendering App) to stop (%.3f secs)...', time.time() - start)
                    if not render_app.is_alive():
                        break
                # if not stopped yet, force termination:
                if render_app.is_alive():
                    self.logger.info('Forcing Chrome Driver (Rendering App) to stop after waiting for %.3f secs!', time.time() - start)
                    render_app.terminate()
                else:
                    self.logger.info('Waited for Chrome Driver (Rendering App) to stop for %.3f secs!', time.time() - start)

    def initializeServer(self):
        """
        Initialize the websocket server.
        """

        self.recv_queue = Queue()
        host = self.config.get("ws_host", "0.0.0.0")
        port = int(self.config.get("ws_port", 8888))
        self.server = WebsocketServer(host, port=port)
        self.client_available = Event()
        self.server.set_fn_message_received(
            lambda client, server, message: self.recv_queue.put(message)
        )
        self.server.set_fn_new_client(
            lambda client, server: self.client_available.set()
        )
        self.server.set_fn_client_left(
            lambda client, server: self.client_available.clear()
        )
        self.server.run_forever(True)
        self.logger.info("WebSocket Server started at %s:%d", host, port)
        self.logger.info("Waiting for WebSocket Clients to connect")
        self.client_available.wait()
        self.logger.info("WebSocket Client connected")

    def convert(self) -> None:
        """
        Read the input queue, convert the data and put the converted data into the
        output queue.
        """
        self.initializeRenderer()
        self.initializeFaceLandmarkerModel()
        self.initializeServer()

        self.duration_detect = 0
        self.duration_render = 0
        self.count_detect = 0
        self.count_render = 0

        if self.config["video_file"] is None:
            clear_queue(self.input_queue)

        self.ready_signal.set()
        while True:
            try:
                ttime, data = self.input_queue.get()
                if ttime is None and data is None:
                    # end of stream
                    self.logger.info("Data is null, stopping conversion")
                    break
                else:
                    self.logger.debug(f"Converting video starting at {ttime[0]}")
                    success = False
                    # try to convert the frame until it is successful
                    while not success:
                        try:
                            out = self.convert_frame(data, ttime)
                            success = True
                        except TimeoutError:
                            success = False
                    self.output_queue.put((ttime, out))
            except Empty:
                pass
            except EOFError:
                break

        # shutdown for avatar converter: print some stats, then put stop-signal on output queue
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info('Duration for detecting (ms / frames): %s / %s (fps: %s)',
                             (self.duration_detect / 1000000) if self.duration_detect != 0 else 0,
                             self.count_detect,
                             round(self.count_detect / (self.duration_detect / 1000000000)) if self.duration_detect != 0 and self.count_detect != 0 else 0)
            self.logger.info('Duration for rendering (ms / frames): %s / %s (fps: %s)',
                             (self.duration_render / 1000000) if self.duration_render != 0 else 0,
                             self.count_render,
                             round(self.count_render / (self.duration_render / 1000000000)) if self.count_render != 0 and self.duration_render != 0 else 0)  # FIXME perf
            # FIXME note that simply adding detection and rendering time may not be accurate,
            #       if they are running (partially) in parallel, but even then, it may give a rough
            #       understanding or estimate of the total time / frame rate
            self.logger.info('Estimated duration for detecting & rendering (ms / frames): %s / %s (fps: %s)',
                             ((self.duration_detect + self.duration_render) / 1000000) if self.duration_detect + self.duration_render != 0 else 0,
                             self.count_detect + self.count_render,
                             round((self.count_detect + self.count_render) / (
                                     (self.duration_detect + self.duration_render) / 1000000000))
                             if self.count_detect + self.count_render != 0 and self.duration_detect + self.duration_render != 0 else 0)  # FIXME perf

        self.stopRenderer()
        # signal end-of-stream for writer via output-queue
        self.output_queue.put((None, None))

    def detect_face(self, data: np.ndarray, timestamp) -> dict:
        """
        Detect the face in the input image and return the face blendshapes and the
        facial transformation matrix as a dictionary. The dictionary has the correct
        format for the websocket server.
        """
        ms_timestamp = int(timestamp[0].item() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=data[0].numpy())

        start_detect = time.perf_counter_ns()  # FIXME perf

        results = self.landmarker.detect_for_video(mp_image, ms_timestamp)

        self.duration_detect += time.perf_counter_ns() - start_detect  # FIXME perf
        self.count_detect += 1

        if len(results.face_blendshapes) == 0:
            return None

        def category_to_dict(category):
            return {
                "index": category.index,
                "score": category.score,
                "categoryName": category.category_name,
                "displayName": category.display_name,
            }

        fb = results.face_blendshapes[0]
        fb_dict = [category_to_dict(category) for category in fb]

        ft = results.facial_transformation_matrixes[0].T.reshape(-1)
        out_dict = {"blendshapes": fb_dict, "transformation_matrix": ft.tolist()}

        return out_dict

    def convert_frame(self, data: np.ndarray, timestamp) -> np.ndarray:
        """
        Convert the input frame to the output frame.
        detect the face in the input frame and send the face blendshapes and the
        facial transformation matrix to the websocket client. The client will send
        back a base64 encoded image of the avatar. The avatar will be inserted into
        a black image and returned as the output frame.
        """
        face_detection_results = self.detect_face(data, timestamp)
        np_out_img = np.zeros(
            (self.config["height"], self.config["width"], 3), dtype=np.uint8
        )
        if face_detection_results:
            # check if a client is connected to the server
            self.client_available.wait()

            start_render = time.perf_counter_ns()  # FIXME perf

            self.server.send_message_to_all(json.dumps(face_detection_results))
            # wait for the client to send the avatar
            message = self.recv_queue.get(timeout=1)
            
            self.duration_render += time.perf_counter_ns() - start_render  # FIXME perf
            self.count_render += 1

            if message.startswith("/"):
                # received message is a base64 encoded image
                raw_img = base64.b64decode(message)
                raw_array = np.frombuffer(raw_img, dtype=np.uint8)
                avatar_img = cv2.imdecode(raw_array, 1)
                # insert the avatar into the image
                avatar_height, avatar_width, _ = avatar_img.shape
                # if avatar is too large, resize it
                if (
                    avatar_height > np_out_img.shape[0]
                    or avatar_width > np_out_img.shape[1]
                ):
                    factor = min(
                        np_out_img.shape[0] / avatar_height,
                        np_out_img.shape[1] / avatar_width,
                    )
                    avatar_img = cv2.resize(
                        avatar_img,
                        (int(avatar_width * factor), int(avatar_height * factor)),
                    )
                    avatar_height, avatar_width, _ = avatar_img.shape
                x = (np_out_img.shape[1] - avatar_width) // 2
                y = np_out_img.shape[0] - avatar_height
                np_out_img[y : y + avatar_height, x : x + avatar_width] = avatar_img
            else:
                self.logger.error("Received message is not a base64 encoded image")

        data = torch.from_numpy(np_out_img).unsqueeze(0)
        return data
