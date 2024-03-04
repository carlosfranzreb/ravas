"""
A queue for storing previous audio samples. It keeps the latest samples, up to a
maximum number, and discards the oldest ones when the maximum is reached. They are all
stored as a single tensor, with the most recent sample at the end of the tensor.
"""

import torch
from torch import Tensor


class PrevAudioQueue:
    def __init__(self, config: dict):
        self.max_samples = config["max_samples"]
        self.queue = torch.zeros(1)
        self.lerp_n = config["interpolation_samples"]
        self.lerp_samples = torch.zeros(1)

    def add(self, audio: Tensor) -> None:
        """
        Add a new audio sample to the queue. If the queue is full, remove the oldest
        samples.
        """
        self.queue = torch.cat((self.queue, audio))
        if self.queue.shape[0] > self.max_samples:
            self.queue = self.queue[-self.max_samples :]

    def get(self) -> Tensor:
        return self.queue

    def get_length(self) -> int:
        return self.queue.shape[0]

    def interpolate(self, audio_conv: Tensor) -> Tensor:
        """
        Interpolate the lerp_samples with the first samples of the given audio
        tensor. Replace the lerp_samples with the last samples of the given
        audio tensor.
        """
        min_samples = min(
            [
                self.lerp_samples.shape[0],
                audio_conv.shape[0],
                self.lerp_n,
            ]
        )
        if min_samples == 0:
            return audio_conv

        audio_samples = audio_conv[:min_samples]
        interpolated = torch.lerp(
            self.lerp_samples, audio_samples, torch.linspace(0, 1, min_samples)
        )
        audio_conv[:min_samples] = interpolated

        self.lerp_samples = torch.cat((self.lerp_samples, audio_conv))
        if self.lerp_samples.shape[0] > self.lerp_n:
            self.lerp_samples = self.lerp_samples[-self.lerp_n :]

        return audio_conv
