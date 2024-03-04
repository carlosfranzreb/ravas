"""
Class that stores the latest output audio samples and interpolates them with the first
samples of the new chunk.
"""

import torch
from torch import Tensor


class Interpolator:
    def __init__(self, config: dict):
        self.lerp_n = config["interpolation_samples"]
        self.lerp_samples = torch.zeros(1)

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
