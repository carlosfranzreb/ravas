import cv2
import numpy as np
import torch

from stream_processing.Processor import ProcessingCallback


class FaceMask(ProcessingCallback):
    def init_callback(self):
        # import mediapipe in function to avoid loading in wrong process
        from mediapipe.python.solutions import drawing_styles
        from mediapipe.python.solutions import drawing_utils as mp_drawing
        from mediapipe.python.solutions import face_mesh as mp_faces

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

        face_mesh = mp_faces.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return [face_mesh, annotate]

    def callback(self, time, data, face_mesh, annotate):
        # get one sample per batch
        # resize image to 720p
        image = data[0].numpy()
        image = cv2.resize(image, (1280, 720), interpolation=cv2.BORDER_DEFAULT)
        black_bg = np.zeros((720, 1280, 3), dtype="uint8")

        # to improve performance mark the image as not writeable to pass by reference
        image.flags.writeable = False
        # detect face landmarks
        results = face_mesh.process(image)

        # annotate image
        if results.multi_face_landmarks:
            black_bg = annotate(black_bg, results)

        data = torch.from_numpy(black_bg[None])
        return time, data
