from queue import Queue

import torch
import torchaudio
import numpy as np

from LLVC.infer import load_model, infer_stream


class StreamingLLVC:
    def __init__(self):
        """
        Initialize the LLVC model.
        """
        self.model, self.model_sr = load_model(
            "checkpoints/llvc/models/checkpoints/llvc/G_500000.pth",
            "LLVC/experiments/llvc/config.json",
        )
        self.queue = Queue(50)

    def __call__(self, frame: np.ndarray, audio_sr: int) -> np.ndarray:
        """
        1. Add the frame to the queue, resampled to the model's SR and as a tensor.
        2. Concatenate the queue into a single tensor and convert it.
        """
        frame = torchaudio.transforms.Resample(
            orig_freq=audio_sr, new_freq=self.model_sr
        )(torch.from_numpy(frame))
        self.queue.put(frame.squeeze(0))
        frames = torch.concatenate(list(self.queue.queue))
        converted = self.convert(frames)
        converted = torchaudio.transforms.Resample(
            orig_freq=self.model_sr, new_freq=audio_sr
        )(converted)
        return converted.numpy()

    def convert(self, audio: torch.tensor) -> torch.tensor:
        outputs, rtf, e2e_latency = infer_stream(self.model, audio, 1.0, 24000)
        return outputs
