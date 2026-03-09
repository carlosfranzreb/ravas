import os
import time
from datetime import datetime
import wave
import pytest
from torch.multiprocessing import Manager, Event

from stream_processing.utils import resolve_file_path
from stream_processing.processor import ProcessingSyncState, ProcessorHandler
from stream_processing.audio_processor import AudioProcessor


REAL_INPUT_VIDEO_OR_WAV = resolve_file_path("./test/test_data/test_input/audio.flac")

TARGET_FEATS = [
    resolve_file_path("./target_feats/knnvc/John.pt"),
    resolve_file_path("./target_feats/mimivc/jeffrey.pt"),
]

LOG_OUTPUT_DIR = resolve_file_path(
    "./test/test_data/test_output/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

OUTPUT_DIRS = [
    LOG_OUTPUT_DIR + "/knnvc",
    LOG_OUTPUT_DIR + "/mimivc",
]


"""
SLOW TEST!

Test the processor handler pipeline using real converters for real audio/video files.

The output should be an anonymized version using either knnvc or mimivc.

This test verifies that:
- The pipeline executes successfully without raising errors.
- The expected output file is created.
- The generated output file has a non-zero duration.
"""


TEST_CASES = [
    {
        "name": "knnvc",
        "sampling_rate": 16000,
        "record_buffersize": 1200,
        "processing_size": 9600,
        "output_buffersize": 9600,
        "target_feats": TARGET_FEATS[0],
        "output_dir": OUTPUT_DIRS[0],
        "converter": {
            "cls": "stream_processing.models.KnnVC",
            "device": "cpu",
            "n_neighbors": 4,
            "n_cluster": 0,
            "use_expressiveness": False,
            "prev_audio_queue": {"max_samples": 9600},
            "prev_ctx": {"use_previous_ctx": False, "max_samples": 0},
            "interpolator": {"n_samples": 450, "weight": 0.75},
        },
    },
    {
        "name": "mimivc",
        "sampling_rate": 24000,
        "record_buffersize": 480,
        "processing_size": 1920,
        "output_buffersize": 1920,
        "target_feats": TARGET_FEATS[1],
        "output_dir": OUTPUT_DIRS[1],
        "converter": {
            "cls": "stream_processing.models.MimiVC",
            "device": "cpu",
            "n_neighbors": 4,
            "n_cluster": 0,
            "use_expressiveness": False,
            "prev_audio_queue": {"max_samples": 9600},
            "interpolator": {"n_samples": 450, "weight": 0.75},
        },
    },
]


@pytest.mark.parametrize("input_file", [REAL_INPUT_VIDEO_OR_WAV])
@pytest.mark.parametrize("case", TEST_CASES, ids=[c["name"] for c in TEST_CASES])
def test_full_pipeline(input_file, case):
    """
    Runs the full pipeline using different converters.

    Flow:
    Read (File/Video) -> Converter -> Sync -> Write (File)
    """

    if not os.path.exists(input_file):
        pytest.fail(f"Input file not found at {input_file}")

    new_output_dir = case["output_dir"]
    os.makedirs(new_output_dir, exist_ok=True)

    manager = Manager()

    # Prepare converter configuration
    converter_cfg = case["converter"].copy()
    converter_cfg["target_feats_path"] = case["target_feats"]

    # Main pipeline configuration
    config = {
        "video_file": input_file,
        "log_dir": new_output_dir,
        "max_unsynced_time": 0.01,
        "sampling_rate": case["sampling_rate"],
        "record_buffersize": case["record_buffersize"],
        "processing_size": case["processing_size"],
        "output_buffersize": case["output_buffersize"],
        "store": True,
        "n_cluster": 0,
        "use_expressiveness": False,
        "input_device": None,
        "output_device": None,
        "converter": converter_cfg,
    }

    # Setup shared sync state
    own_state = ProcessingSyncState()
    ext_state = ProcessingSyncState()

    # In file mode we don't wait for external video sync
    ext_state.disabled.value = True

    pipe_ready = Event()
    log_queue = manager.Queue()

    # Initialize processor
    processor = AudioProcessor(
        config=config,
        audio_sync_state=own_state,
        external_sync_state=ext_state,
        pipeline_sync_state=pipe_ready,
        log_queue=log_queue,
        log_level="INFO",
    )

    handler = ProcessorHandler(processor)

    print(f"\nStarting {case['name']} pipeline for: {input_file}")

    start_time = time.time()

    try:
        handler.start()

        # Wait until pipeline signals completion
        success = processor.queues.finished.wait(timeout=60)

        assert success, (
            "Pipeline timed out. Check if models or target features are missing."
        )

        end_time = time.time()

        print(f"Processing finished in {round(end_time - start_time, 2)} seconds")

    finally:
        handler.stop()

    # Validate output
    output_path = os.path.join(new_output_dir, "audio.wav")

    assert os.path.exists(output_path), "Output file was not created."

    with wave.open(output_path, "rb") as f:
        duration = f.getnframes() / f.getframerate()

        print(f"Output duration: {round(duration, 2)} seconds")

        assert duration > 0, "Output file is empty."