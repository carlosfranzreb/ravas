import base64
import json
import logging
import os
import time
from enum import Enum
from queue import Empty
from threading import Event
from typing import Optional, IO

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch.multiprocessing import Queue, Process, Event
from websocket_server import WebsocketServer

from .avatar_resources import get_avatar_models_dir
from .chrome_runner import start_browser
from .web_server import start_server
from .opengl_runner import start_renderer as start_opengl_renderer
from ...processor import Converter
from ...utils import clear_queue


class RenderAppType(Enum):
    BROWSER = "browser"
    OPENGL_APP = "opengl"

    @staticmethod
    def to_type(string: str):
        t = next((x for x in RenderAppType if x.value == string), None)
        if t:
            return t
        raise ValueError(
            'Invalid value "": must be one of ', [t.value for t in RenderAppType]
        )


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
        super().__init__(
            name, config, input_queue, output_queue, log_queue, log_level, ready_signal
        )
        self.log_queue = log_queue
        self.log_level = log_level
        self._stopped = False

        self.client_available: Event | None = None

        self.render_server: Process | None = None
        self.render_app: Process | None = None
        self.stop_render_app: Queue | None = None

        self.render_app_type: RenderAppType | None = None

        self.recv_queue: Queue | None = None
        self.render_app_input: Queue | None = None  # only for opengl-renderer
        self.server: WebsocketServer | None = None  # only for chrome/web renderer

        self.detection_log_file: IO | None = None

        # convert old configuration, if necessary:
        headless_window = config.get("show_chrome_window")
        if headless_window is not None and config.get("show_renderer_window") is None:
            config["show_renderer_window"] = headless_window

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

        detection_log_file = self.config.get("detection_log")
        if detection_log_file:
            if not os.path.isabs(detection_log_file):
                detection_log_file = os.path.join(
                    self.config.get("log_dir", ""), detection_log_file
                )
            self.detection_log_file = open(detection_log_file, "wt+", encoding="utf-8")
            self.detection_log_file.write("[\n")

    def initializeOpenGLRenderer(self):
        self.client_available = Event()
        in_queue = Queue()
        out_queue = Queue()
        avatar_file_path = os.path.join(
            get_avatar_models_dir(), self.config.get("avatar_uri", "avatar_1_f.glb")
        )
        app_args = {
            "model_path": avatar_file_path,
            "input_queue": in_queue,
            "output_queue": out_queue,
            "run_headless": not self.config.get("show_renderer_window", False),
            "log_queue": self.log_queue,
            "log_level": self.log_level,
        }
        render_app = Process(
            target=start_opengl_renderer, kwargs=app_args, name="opengl_render_app"
        )
        if not self._stopped:
            render_app.start()
            self.logger.info("Started OpenGL Rendering App (pid %s)", render_app.pid)
            self.logger.info("Waiting for OpenGL Rendering App to send ready signal")
            ready_signal = out_queue.get()
            assert (
                ready_signal is False
            ), "did not expected 'ready' signal (value False)"
            self.client_available.set()
            self.logger.info("OpenGL Rendering App is ready")

        if not self._stopped:
            self.render_app_type = RenderAppType.OPENGL_APP
            self.render_app_input = in_queue
            self.recv_queue = out_queue
            self.render_server = None
            self.render_app = render_app
            self.stop_render_app = in_queue
        else:
            self.stopRenderer(
                render_app=render_app,
                stop_render_app=in_queue,
                renderer_type=RenderAppType.OPENGL_APP,
            )

    def initializeBrowserRenderer(self):

        if not self.config.get("start_chrome_renderer", True):
            self.logger.info(
                "Disabled automated start of Chrome driver for rendering avatar."
            )
            return

        app_port = int(self.config.get("app_port", 3000))
        use_extension = self.config.get("use_chrome_extension", True)
        if not use_extension:
            server_args = {
                "port": app_port,
                "log_queue": self.log_queue,
                "log_level": self.log_level,
            }
            render_app_server = Process(
                target=start_server, kwargs=server_args, name="render_app_server"
            )
            if not self._stopped:
                render_app_server.start()
                self.logger.info(
                    "Started web Server for Rendering App (pid %s)",
                    render_app_server.pid,
                )
            else:
                self.logger.info(
                    "Did not started web Server for Rendering App: already stopped!"
                )
        else:
            render_app_server = None

        ws_port = int(self.config.get("ws_port", 8888))
        render_app_stop = (
            Queue()
        )  # NOTE: Event() is not pickable for sub-processes, so use Queue for sending stop signal
        app_args = {
            "ws_addr": "http://127.0.0.1:{}".format(ws_port),
            "stop_signal": render_app_stop,
            "port": app_port,
            "web_extension": use_extension,
            "run_headless": not self.config.get("show_renderer_window", False),
            "avatar_uri": self.config.get("avatar_uri", None),
            "hide_avatar_selection": self.config.get("hide_avatar_selection", None),
            "log_queue": self.log_queue,
            "log_level": self.log_level,
        }
        render_app = Process(target=start_browser, kwargs=app_args, name="render_app")
        if not self._stopped:
            render_app.start()
            self.logger.info(
                "Started Chrome Driver for Rendering App (pid %s)", render_app.pid
            )

        if not self._stopped:
            self.render_app_type = RenderAppType.BROWSER
            self.render_server = render_app_server
            self.render_app = render_app
            self.stop_render_app = render_app_stop
        else:
            self.stopRenderer(
                render_app=render_app,
                stop_render_app=render_app_stop,
                render_server=render_app_server,
                renderer_type=RenderAppType.BROWSER,
            )

    def stopRenderer(
        self,
        render_app: Optional[Process] = None,
        stop_render_app: Optional[Queue] = None,
        render_server: Optional[Process] = None,
        renderer_type: Optional[RenderAppType] = None,
    ):
        self._stopped = True

        if self.detection_log_file:
            # NOTE add an empty object before closing the array, to ensure a valid JSON format
            #      due to possible pending comma in previous entry / write to the file
            self.detection_log_file.write("{}\n]\n")
            self.detection_log_file.close()
            self.detection_log_file = None

        # NOTE: usually, we do not really need to wait for rendering-app-process to shut down completely
        #       (i.e. only initiate its shutdown, and then leave it to do its clean-up etc.)
        #       ... however, in some cases, the process will not shut down:
        #       since we did not wait, we cannot manually shut it down then.
        #       So in these cases, it would be best to wait, and if after some timeout, we shut it down
        #       forcefully, if it is not terminated yet.
        #       -> setting this to True, will wait and shut down the process forcefully, if it is still running
        is_wait_for_render_app_to_finish = False  # TODO make this configurable?

        if not render_app:
            render_app = self.render_app
        if not render_server:
            render_server = self.render_server
        if not stop_render_app:
            stop_render_app = self.stop_render_app
        if not renderer_type:
            renderer_type = self.render_app_type

        if render_server and render_server.is_alive():
            self.logger.info("Stopping Web Server Rendering App...")
            render_server.terminate()

        if render_app and render_app.is_alive():

            app_info_str = (
                "Chrome Driver for"
                if renderer_type == RenderAppType.BROWSER
                else "OpenGL"
            )
            self.logger.info("Stopping %s Rendering App...", app_info_str)

            if renderer_type == RenderAppType.BROWSER:
                # NOTE need to signal the render_app process to stop, so that the chrome driver can be closed properly
                #      (simply calling render_app.terminate() will leave chrome instance running)
                stop_render_app.put(
                    None
                )  # send value `None` for signaling the render_app to stop the chrome driver
            else:
                stop_render_app.put(
                    None
                )  # NOTE: OpenGL render app expects None as stop signal!

            if is_wait_for_render_app_to_finish:
                start = time.time()
                # wait for max. ~ 10 secs (10 loops a 1 sec), or break, if not alive anymore
                # (note: Chrome currently needs about ~6 secs to shut down)
                for i in range(10):
                    render_app.join(1)
                    self.logger.info(
                        "Waiting for %s Rendering App to stop (%.3f secs)...",
                        app_info_str,
                        time.time() - start,
                    )
                    if not render_app.is_alive():
                        break
                # if not stopped yet, force termination:
                if render_app.is_alive():
                    self.logger.info(
                        "Forcing %s to stop after waiting for %.3f secs!",
                        app_info_str,
                        time.time() - start,
                    )
                    render_app.terminate()
                else:
                    self.logger.info(
                        "Waited for %s to stop for %.3f secs!",
                        app_info_str,
                        time.time() - start,
                    )

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

        renderer_type_str = self.config.get(
            "avatar_renderer", RenderAppType.OPENGL_APP.value
        )
        self.render_app_type = RenderAppType.to_type(renderer_type_str)

        if self.render_app_type == RenderAppType.BROWSER:
            self.initializeBrowserRenderer()
            self.initializeFaceLandmarkerModel()
            self.initializeServer()
        else:
            self.initializeOpenGLRenderer()
            self.initializeFaceLandmarkerModel()

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
            self.logger.info(
                "Duration for detecting (ms / frames): %s / %s (fps: %s)",
                (self.duration_detect / 1000000) if self.duration_detect != 0 else 0,
                self.count_detect,
                (
                    round(self.count_detect / (self.duration_detect / 1000000000))
                    if self.duration_detect != 0 and self.count_detect != 0
                    else 0
                ),
            )
            self.logger.info(
                "Duration for rendering (ms / frames): %s / %s (fps: %s)",
                (self.duration_render / 1000000) if self.duration_render != 0 else 0,
                self.count_render,
                (
                    round(self.count_render / (self.duration_render / 1000000000))
                    if self.count_render != 0 and self.duration_render != 0
                    else 0
                ),
            )  # FIXME perf
            # FIXME note that simply adding detection and rendering time may not be accurate,
            #       if they are running (partially) in parallel, but even then, it may give a rough
            #       understanding or estimate of the total time / frame rate
            self.logger.info(
                "Estimated duration for detecting & rendering (ms / frames): %s / %s (fps: %s)",
                (
                    ((self.duration_detect + self.duration_render) / 1000000)
                    if self.duration_detect + self.duration_render != 0
                    else 0
                ),
                self.count_detect + self.count_render,
                (
                    round(
                        (self.count_detect + self.count_render)
                        / ((self.duration_detect + self.duration_render) / 1000000000)
                    )
                    if self.count_detect + self.count_render != 0
                    and self.duration_detect + self.duration_render != 0
                    else 0
                ),
            )  # FIXME perf

        self.stopRenderer()
        # signal end-of-stream for writer via output-queue
        self.output_queue.put((None, None))

    def detect_face(self, data: np.ndarray, timestamp) -> dict | None:
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

        if self.detection_log_file:
            log_data = out_dict.copy()
            log_data["ts"] = ms_timestamp
            self.detection_log_file.write(json.dumps(log_data, ensure_ascii=False))
            self.detection_log_file.write(",\n")

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

            if self.server:
                self.server.send_message_to_all(json.dumps(face_detection_results))
            else:
                self.render_app_input.put(face_detection_results)
            # wait for the client to send the avatar
            message = self.recv_queue.get(timeout=1)

            self.duration_render += time.perf_counter_ns() - start_render  # FIXME perf
            self.count_render += 1

            is_binary = message and not isinstance(message, str)
            if is_binary or (message and message.startswith("/")):
                # received message is a base64 encoded image
                raw_img = message if is_binary else base64.b64decode(message)
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
