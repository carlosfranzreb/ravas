import os
import time
from datetime import datetime
import wave
import torch
import pytest
from torch.multiprocessing import Manager, Event

from stream_processing.utils import resolve_file_path
from stream_processing.processor import ProcessingSyncState, ProcessorHandler
from stream_processing.audio_processor import AudioProcessor

REAL_INPUT_VIDEO_OR_WAV = resolve_file_path("./test/test_data/test_input/audio.flac")
TARGET_FEATS_PATH_KNNVC = resolve_file_path("./target_feats/knnvc/John.pt")
TARGET_FEATS_PATH_MIMIVC = resolve_file_path("./target_feats/mimivc/jeffrey.pt")
LOG_OUTPUT_DIR = resolve_file_path("./test/test_data/test_output/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
"""
Test the full end-to-end pipeline using real converters and actual input
audio/video files.

This test verifies that:
- The pipeline executes successfully without raising errors.
- The expected output file is created.
- The generated output file has a non-zero duration.
"""
@pytest.mark.parametrize("input_file", [REAL_INPUT_VIDEO_OR_WAV])
def test_knnvc_full_pipeline(input_file):
    """
    Runs the full pipeline using the real KnnVC converter.
    Flow: Read (File/Video) -> KnnVC (ONNX) -> Sync -> Write (File)
    """
    
    if not os.path.exists(input_file):
        pytest.fail(f"Input file not found at {input_file}")

    new_output_dir = LOG_OUTPUT_DIR + "/knnvc"
    if not os.path.exists(new_output_dir):
        os.makedirs(new_output_dir)

    manager = Manager()
    
    # 1. Configuration
    config = {
        "video_file": input_file,
        "log_dir": new_output_dir,
        "max_unsynced_time": 0.01,
        "sampling_rate": 16000,
        "record_buffersize": 1200,
        "processing_size": 9600,
        "output_buffersize": 9600,
        "store": True, 
        "n_cluster": 0,
        "use_expressiveness": False, 
        "input_device": None,
        "output_device": None,
        "converter": {
            "cls": "stream_processing.models.KnnVC",
            "device": "cpu",
            "target_feats_path": TARGET_FEATS_PATH_KNNVC,
            "n_neighbors": 4,
            "n_cluster": 0,
            "use_expressiveness": False,
            "prev_audio_queue": {
                "max_samples": 9600
            },
            "prev_ctx": {
                "use_previous_ctx": False,
                "max_samples": 0
            },
            "interpolator": {
                "n_samples": 450,
                "weight": 0.75
            },
        }
    }

    # 2. Setup Shared State
    own_state = ProcessingSyncState()
    ext_state = ProcessingSyncState()
    ext_state.disabled.value = True # In file mode, don't wait for video sync
    pipe_ready = Event()
    log_queue = manager.Queue()

    # 3. Initialize Processor
    # This will load ONNX models in the 'convert' process later
    processor = AudioProcessor(
        config=config,
        audio_sync_state=own_state,
        external_sync_state=ext_state,
        pipeline_sync_state=pipe_ready,
        log_queue=log_queue,
        log_level="INFO"
    )

    handler = ProcessorHandler(processor)

    print(f"\nStarting Pipeline for: {input_file}")
    start_time = time.time()
    
    try:
        handler.start()
        
        # 4. Wait for completion
        success = processor.queues.finished.wait(timeout=60)
        
        assert success, "Pipeline timed out. Check if ONNX models or target features are missing."
        
        end_time = time.time()
        print(f"Processing finished in {round(end_time - start_time, 2)}s")

    finally:
        handler.stop()

    # 5. Final Assertions
    output_path = os.path.join(new_output_dir, "audio.wav")
    assert os.path.exists(output_path), "Output file was not created."
    
    with wave.open(output_path, "rb") as f:
        duration = f.getnframes() / f.getframerate()
        print(f"Output duration: {round(duration, 2)}s")
        assert duration > 0, "Output file is empty."

@pytest.mark.parametrize("input_file", [REAL_INPUT_VIDEO_OR_WAV])
def test_mimivc_full_pipeline(input_file):
    """
    Runs the full pipeline using the real MimiVC converter.
    Flow: Read (File/Video) -> MimiVC (ONNX) -> Sync -> Write (File)
    """
    
    if not os.path.exists(input_file):
        pytest.fail(f"Input file not found at {input_file}")

    new_output_dir = LOG_OUTPUT_DIR + "/mimivc"
    if not os.path.exists(new_output_dir):
        os.makedirs(new_output_dir)

    manager = Manager()
    
    # 1. Configuration
    config = {
        "video_file": input_file,
        "log_dir": new_output_dir,
        "max_unsynced_time": 0.01,
        "sampling_rate": 24000,
        "record_buffersize": 480,
        "processing_size": 1920,
        "output_buffersize": 1920,
        "store": True, 
        "n_cluster": 0,
        "use_expressiveness": False, 
        "input_device": None,
        "output_device": None,
        "converter": {
            "cls": "stream_processing.models.MimiVC",
            "device": "cpu",
            "target_feats_path": TARGET_FEATS_PATH_MIMIVC,
            "n_neighbors": 4,
            "n_cluster": 0,
            "use_expressiveness": False,
            "prev_audio_queue": {
                "max_samples": 9600
            },
            "interpolator": {
                "n_samples": 450,
                "weight": 0.75
            },
        }
    }

    # 2. Setup Shared State
    own_state = ProcessingSyncState()
    ext_state = ProcessingSyncState()
    ext_state.disabled.value = True # In file mode, don't wait for video sync
    pipe_ready = Event()
    log_queue = manager.Queue()

    # 3. Initialize Processor
    # This will load ONNX models in the 'convert' process later
    processor = AudioProcessor(
        config=config,
        audio_sync_state=own_state,
        external_sync_state=ext_state,
        pipeline_sync_state=pipe_ready,
        log_queue=log_queue,
        log_level="INFO"
    )

    handler = ProcessorHandler(processor)

    print(f"\nStarting Pipeline for: {input_file}")
    start_time = time.time()
    
    try:
        handler.start()
        
        # 4. Wait for completion
        success = processor.queues.finished.wait(timeout=60)
        
        assert success, "Pipeline timed out. Check if ONNX models or target features are missing."
        
        end_time = time.time()
        print(f"Processing finished in {round(end_time - start_time, 2)}s")

    finally:
        handler.stop()

    # 5. Final Assertions
    output_path = os.path.join(new_output_dir, "audio.wav")
    assert os.path.exists(output_path), "Output file was not created."
    
    with wave.open(output_path, "rb") as f:
        duration = f.getnframes() / f.getframerate()
        print(f"Output duration: {round(duration, 2)}s")
        assert duration > 0, "Output file is empty."