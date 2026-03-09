import os
import time
from datetime import datetime
import cv2
import pytest
from torch.multiprocessing import Manager, Event

from stream_processing.utils import resolve_file_path
from stream_processing.processor import ProcessingSyncState, ProcessorHandler
from stream_processing.video_processor import VideoProcessor


"""
SLOW TEST!

Test the processor handler pipeline using real converters for video files.

The output video should contain the rendered Avatar if a face is detected,
and a blank video if no face is detected.

This test verifies that:
- The pipeline executes successfully without raising errors.
- The expected output file is created.
- The generated output file has frames and FPS > 0.
"""

INPUT_VIDEO_FACE = [
    resolve_file_path("./test/test_data/test_input/video_face.mp4"),
    resolve_file_path("./test/test_data/test_input/video_no_face.mp4"),
]

LOG_OUTPUT_DIR = resolve_file_path(
    "./test/test_data/test_output/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

OUTPUT_DIRS = [
    LOG_OUTPUT_DIR + "/face",
    LOG_OUTPUT_DIR + "/no_face",
]


VIDEO_TEST_CASES = [
    {
        "name": "face",
        "input_file": INPUT_VIDEO_FACE[0],
        "output_dir": OUTPUT_DIRS[0],
    },
    {
        "name": "no_face",
        "input_file": INPUT_VIDEO_FACE[1],
        "output_dir": OUTPUT_DIRS[1],
    },
]


@pytest.mark.parametrize("case", VIDEO_TEST_CASES, ids=[c["name"] for c in VIDEO_TEST_CASES])
def test_avatar_video_pipeline(case):

    input_file = case["input_file"]
    new_output_dir = case["output_dir"]

    if not os.path.exists(input_file):
        pytest.fail(f"Input video not found at {input_file}")

    os.makedirs(new_output_dir, exist_ok=True)

    manager = Manager()

    config = {
        "video_file": input_file,
        "log_dir": new_output_dir,
        "use_video": True,
        "max_unsynced_time": 0.01,
        "width": 640,
        "height": 480,
        "input_device": 0,
        "processing_size": 1,
        "max_fps": 20,
        "output_virtual_cam": False,
        "output_window": False,
        "store": True,
        "store_format": "avi",
        "converter": {
            "cls": "stream_processing.models.Avatar",
            "width": 640,
            "height": 480,
            "ws_host": "0.0.0.0",
            "ws_port": 8888,
            "app_port": 3000,
            "avatar_renderer": "opengl",
            "start_chrome_renderer": False,
            "use_chrome_extension": False,
            "show_renderer_window": False,
            "avatar_uri": "./avatar_1_f.glb",
        },
    }

    video_sync_state = ProcessingSyncState()
    audio_sync_state = ProcessingSyncState()

    # Disable audio sync since we run video only
    audio_sync_state.disabled.value = True

    pipe_ready = Event()
    log_queue = manager.Queue()

    processor = VideoProcessor(
        config=config,
        video_sync_state=video_sync_state,
        external_sync_state=audio_sync_state,
        pipeline_sync_state=pipe_ready,
        log_queue=log_queue,
        log_level="INFO",
    )

    handler = ProcessorHandler(processor)

    start_time = time.time()

    try:
        handler.start()

        success = processor.queues.finished.wait(timeout=60)

        assert success, "Pipeline timed out"

    finally:
        handler.stop()

    print(f"Processing finished in {round(time.time() - start_time, 2)}s")

    # Validate output
    output_path = os.path.join(new_output_dir, "video.avi")

    assert os.path.exists(output_path), "Output video not created"

    cap = cv2.VideoCapture(output_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    assert frame_count > 0, "Video contains no frames"
    assert fps > 0, "Invalid FPS"