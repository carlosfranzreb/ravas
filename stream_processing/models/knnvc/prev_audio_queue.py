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
        self.queue = torch.zeros(self.max_samples, dtype=torch.float32)

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
