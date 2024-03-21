from queue import Queue, Empty

import cv2
import numpy as np
import torch

from mediapipe.python.solutions import drawing_styles
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import face_mesh as mp_faces
from stream_processing.processor import Converter
from stream_processing.utils import clear_queue


class FaceMask(Converter):
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
        Initialize the FaceMesh model.
        """
        super().__init__(name, config, input_queue, output_queue, log_queue, log_level)
        self.face_mesh = mp_faces.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def convert(self) -> None:
        """
        Read the input queue, convert the data and put the converted data into the
        sync queue.
        """
        clear_queue(self.input_queue)
        while True:
            try:
                ttime, data = self.input_queue.get()
                out = self.convert_frame(data)
                self.output_queue.put((ttime, out))
            except Empty:
                pass

    def convert_frame(self, data: np.ndarray) -> np.ndarray:
        image = resize_image(
            data[0].numpy(), self.config["width"], self.config["height"]
        )
        image.flags.writeable = False
        results = self.face_mesh.process(image)

        # annotate image
        black_bg = np.zeros(
            (self.config["width"], self.config["height"], 3), dtype="uint8"
        )
        if results.multi_face_landmarks:
            black_bg = annotate(black_bg, results)

        data = torch.from_numpy(black_bg[None])
        return data


def annotate(frame, results):
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            mp_faces.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            mp_faces.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
        )
        mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            mp_faces.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
        )
    return frame


def resize_image(image: np.array, desired_width: int, desired_height: int) -> np.array:
    """
    Pad the image to the desired height and width. If the image is smaller than the
    desired height and width, the image is centered in the padded image. If the image
    is larger than the desired height and width, it is resized first to match the
    desired height and width.
    """
    height, width, _ = image.shape

    # handle the height
    vertical_pad = desired_height - height
    top_pad = vertical_pad // 2
    bottom_pad = vertical_pad - top_pad
    if vertical_pad < 0:
        width = int(width * (desired_height / height))
        image = cv2.resize(image, (width, desired_height))
        height = desired_height
        top_pad = 0
        bottom_pad = 0

    # handle the width
    horizontal_pad = desired_width - width
    left_pad = horizontal_pad // 2
    right_pad = horizontal_pad - left_pad
    if horizontal_pad < 0:
        height = int(height * (desired_width / width))
        image = cv2.resize(image, (desired_width, height))
        left_pad = 0
        right_pad = 0
        vertical_pad = desired_height - height
        top_pad = vertical_pad // 2
        bottom_pad = vertical_pad - top_pad

    if horizontal_pad > 0 or vertical_pad > 0:
        return cv2.copyMakeBorder(
            image,
            top_pad,
            bottom_pad,
            left_pad,
            right_pad,
            cv2.BORDER_CONSTANT,
            value=0,
        )
    else:
        return image
