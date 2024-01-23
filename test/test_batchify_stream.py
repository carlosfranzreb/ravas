from multiprocessing import Process
import time
import unittest
import numpy as np

import torch
from steam_processing.Processor import Processor, ProcessingQueues, ProcessingSyncState
from steam_processing.utils import batchify_input_stream


class TestProcess(unittest.TestCase):
    def test_batchify_video_example(self):
        input_shape = (1, 100, 100, 3)

        upper_bound_fps = 16
        out_batch_size = 7

        def read_callback():
            return torch.randint(0, 255, input_shape, dtype=torch.uint8)

        (processing_time, processing_data), (
            chunk_part_for_next_times,
            chunk_part_for_next,
        ) = batchify_input_stream(
            read_callback=read_callback,
            out_batch_size=out_batch_size,
            input_shape=input_shape,
            sampling_rate=out_batch_size,
            chunk_part_for_next_times=None,
            chunk_part_for_next=None,
            dtype=torch.uint8,
            upper_bound_fps=upper_bound_fps,
            last_frame_time=0,
        )

        t_delta = processing_time[1:] - processing_time[:-1]
        self.assertTrue(torch.all(t_delta - (1 / upper_bound_fps) <= 0.01))
        self.assertTrue(torch.all(t_delta - (1 / upper_bound_fps) >= 0))

        self.assertTrue(processing_data.shape == (out_batch_size, *input_shape[1:]))
        self.assertTrue(len(processing_time) == out_batch_size)

    def test_batchify_audio_example(self):
        input_shape = (10,)

        out_batch_size = 50
        sampling_rate = 44100

        def read_callback():
            return torch.randint(0, 255, input_shape, dtype=torch.uint8)

        (processing_time, processing_data), (
            chunk_part_for_next_times,
            chunk_part_for_next,
        ) = batchify_input_stream(
            read_callback=read_callback,
            out_batch_size=out_batch_size,
            input_shape=input_shape,
            sampling_rate=sampling_rate,
            chunk_part_for_next_times=None,
            chunk_part_for_next=None,
            dtype=torch.uint8,
        )

        t_delta = processing_time[1:] - processing_time[:-1]
        self.assertTrue(
            torch.count_nonzero(torch.abs(t_delta - 1 / sampling_rate) > 1e-06)
            <= out_batch_size // input_shape[0]
        )

        self.assertTrue(processing_data.shape == (out_batch_size, *input_shape[1:]))
        self.assertTrue(len(processing_time) == out_batch_size)

    def test_batchify_audio_example_with_chunk_part(self):
        input_shape = (10,)

        out_batch_size = 50
        sampling_rate = 44100

        def read_callback():
            return torch.ones(input_shape, dtype=torch.uint8)

        (processing_time, processing_data), (
            chunk_part_for_next_times,
            chunk_part_for_next,
        ) = batchify_input_stream(
            read_callback=read_callback,
            out_batch_size=out_batch_size,
            input_shape=input_shape,
            sampling_rate=sampling_rate,
            chunk_part_for_next_times=torch.ones(5) * 0.5,
            chunk_part_for_next=torch.ones(5) * 2,
            dtype=torch.uint8,
        )

        self.assertTrue(torch.all(processing_data[:5] == 2))
        self.assertTrue(torch.all(processing_data[5:] == 1))
        self.assertTrue(torch.all(processing_time[:5] == 0.5))
        self.assertTrue(torch.all(processing_time[5:] != 0.5))
        self.assertTrue(chunk_part_for_next.shape == (5,))
        self.assertTrue(chunk_part_for_next_times.shape == (5,))
        self.assertTrue(torch.all(chunk_part_for_next == 1))
        self.assertTrue(torch.all(chunk_part_for_next_times != 0.5))
        self.assertTrue(processing_data.shape == (out_batch_size, *input_shape[1:]))
        self.assertTrue(len(processing_time) == out_batch_size)


if __name__ == "__main__":
    unittest.main()
