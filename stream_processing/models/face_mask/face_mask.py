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
        # get one sample per batch
        # resize image to 720p
        image = data[0].numpy()
        image = cv2.resize(image, (1280, 720), interpolation=cv2.BORDER_DEFAULT)
        black_bg = np.zeros((720, 1280, 3), dtype="uint8")

        # to improve performance mark the image as not writeable to pass by reference
        image.flags.writeable = False
        # detect face landmarks
        results = self.face_mesh.process(image)

        # annotate image
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
