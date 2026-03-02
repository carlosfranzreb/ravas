import pytest
import cv2
import torch
import base64
import numpy as np
from multiprocessing import Queue, Event
from unittest.mock import Mock
from stream_processing.utils import resolve_file_path

from stream_processing.models.avatar.avatar import Avatar

"""
Test the Avatar function `convert_frame` using a real image.
This test verifies:
- That a real face image results in a non-black output
- The count of detections and renders is as expected
- That a non-face image results in a black output and no rendering
"""
FACE_IMAGE_PATH = resolve_file_path("./test/test_data/test_input/face.jpg")
NO_FACE_IMAGE_PATH = resolve_file_path("./test/test_data/test_input/no_face.jpg")


@pytest.fixture
def avatar_real(tmp_path):
    config = {
        "width": 640,
        "height": 480,
        "video_file": None,
        "log_dir": str(tmp_path),
    }

    avatar = Avatar(
        name="integration_convert",
        config=config,
        input_queue=Queue(),
        output_queue=Queue(),
        log_queue=Queue(),
        log_level="INFO",
        ready_signal=Event(),
    )
    avatar.initializeFaceLandmarkerModel()

    return avatar


def test_convert_frame_real_detection_and_fake_renderer(avatar_real):
    """
    Test that a real image with a face goes through the detection and rendering path, resulting in a non-black output.
    """
    img = cv2.imread(FACE_IMAGE_PATH)
    assert img is not None, "Face image not found"

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    data = torch.from_numpy(img).unsqueeze(0)
    timestamp = torch.tensor([1.0])  # must be increasing if reused

    # --- Fake renderer setup ---
    avatar_real.client_available = Mock()
    avatar_real.client_available.wait = lambda: None

    # Use server mode to simplify
    avatar_real.server = Mock()

    # Fake avatar image returned from renderer
    avatar_img = np.ones((200, 200, 3), dtype=np.uint8) * 255

    _, buffer = cv2.imencode(".jpg", avatar_img)
    base64_img = base64.b64encode(buffer.tobytes()).decode()

    avatar_real.recv_queue = Mock()
    avatar_real.recv_queue.get.return_value = base64_img

    # --- Run convert_frame ---
    result = avatar_real.convert_frame(data, timestamp)

    # --- Assertions ---
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 480, 640, 3)

    # Must not be fully black
    assert torch.any(result != 0)

    # Ensure MediaPipe was actually triggered
    assert avatar_real.count_detect == 1

    # Ensure rendering path executed
    assert avatar_real.count_render == 1

    # Ensure message was sent to renderer
    avatar_real.server.send_message_to_all.assert_called_once()


def test_convert_frame_real_no_face(avatar_real):
    """
    Test that an image with no face results in a black output and no rendering.
    """
    img = cv2.imread(NO_FACE_IMAGE_PATH)
    assert img is not None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    data = torch.from_numpy(img).unsqueeze(0)
    timestamp = torch.tensor([2.0])

    result = avatar_real.convert_frame(data, timestamp)

    # Should remain black because no detection
    assert torch.all(result == 0)
    assert avatar_real.count_render == 0
