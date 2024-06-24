import json
import os
from torch.multiprocessing import Queue
import cv2
import numpy as np
import torch
from ...processor import Converter
from ...utils import clear_queue
from websocket_server import WebsocketServer
import base64
from threading import Event
from queue import Empty
import mediapipe as mp


class Avatar(Converter):
    def __init__(
        self,
        name: str,
        config: dict,
        input_queue: Queue,
        output_queue: Queue,
        log_queue: Queue,
        log_level: str,
    ) -> None:
        """
        Initialize the Avatar Model.
        """
        super().__init__(name, config, input_queue, output_queue, log_queue, log_level)

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

    def initializeServer(self):
        """
        Initialize the websocket server.
        """

        self.recv_queue = Queue()
        self.server = WebsocketServer(
            os.environ["ws_host"], port=int(os.environ["ws_port"])
        )
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
        self.logger.info("Server started")
        self.logger.info("Waiting for clients to connect")
        self.client_available.wait()
        self.logger.info("Client connected")

    def convert(self) -> None:
        """
        Read the input queue, convert the data and put the converted data into the
        output queue.
        """
        self.initializeFaceLandmarkerModel()
        self.initializeServer()
        if self.config["video_file"] is None:
            clear_queue(self.input_queue)
        while True:
            try:
                ttime, data = self.input_queue.get()
                if ttime is None and data is None:
                    # end of stream
                    self.output_queue.put((None, None))
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

    def detect_face(self, data: np.ndarray, timestamp) -> dict:
        """
        Detect the face in the input image and return the face blendshapes and the
        facial transformation matrix as a dictionary. The dictionary has the correct
        format for the websocket server.
        """
        ms_timestamp = int(timestamp[0].item() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=data[0].numpy())
        results = self.landmarker.detect_for_video(mp_image, ms_timestamp)
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
            self.server.send_message_to_all(json.dumps(face_detection_results))
            # wait for the client to send the avatar
            message = self.recv_queue.get(timeout=1)

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
