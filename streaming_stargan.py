from queue import Queue
from collections import defaultdict as DefaultDict

import torch
import torchaudio
from torchaudio.transforms import Resample
import numpy as np

from ap_demo.utility_components.inference_utils import (
    build_inference_pipeline,
    preprocess,
)


INPUT_SR = 16000
MODEL_SR = 24000


class StreamingSG:
    def __init__(self, target_id: int):
        (
            self.starganv2,
            self.F0_model,
            self.vocoder,
            self.style_vecs,
        ) = build_inference_pipeline(device="cpu")
        """
        Initialize the StarGAN model and define the target speaker.
        """
        self.target = self.style_vecs["p" + str(target_id)][0]
        self.queue = Queue(maxsize=10)
        self.resample_to_model = Resample(INPUT_SR, MODEL_SR)
        self.resample_to_input = Resample(MODEL_SR, INPUT_SR)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """
        1. Add the frame to the queue, resampled to the model's SR and as a tensor.
        2. Concatenate the queue into a single tensor and convert it.
        """
        self.queue.put(self.resample_to_model(torch.from_numpy(frame)))
        frames = torch.concatenate(list(self.queue.queue))
        return self.convert(frames).numpy()

    def convert(self, audio: torch.tensor) -> torch.tensor:
        source = preprocess(audio)
        with torch.no_grad():
            f0_feat = self.F0_model.get_feature_GAN(source.unsqueeze(1))
            out = self.starganv2.generator(source.unsqueeze(1), self.target, F0=f0_feat)
            c = out.transpose(-1, -2).squeeze()
            recon = self.vocoder.inference(c)

        return recon.view(-1)
