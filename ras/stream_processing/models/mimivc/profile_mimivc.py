import yaml
from queue import Queue

import line_profiler
import torch

from stream_processing.models import MimiVC
from stream_processing.processor import ProcessingSyncState


if __name__ == "__main__":
    cfg_f = "ras/configs/onnx_models_ui.yaml"
    cfg = yaml.safe_load(open(cfg_f))
    mimivc = MimiVC(
        "mimivc",
        cfg["audio"]["converter"],
        Queue(100),
        Queue(100),
        Queue(100),
        "ERROR",
        ProcessingSyncState(),
    )

    # Create input and run profiled method
    audio_in = torch.zeros(1920, dtype=torch.int16)
    for _ in range(100):
        mimivc.convert_audio(audio_in)
