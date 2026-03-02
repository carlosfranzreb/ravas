import torch
from stream_processing.utils import batchify_input_stream

"""
Test batchify_input_stream function
The batchify_input function should read from a read callback and return the data in batches.
It should also return the time of each sample in the batch.
"""


def test_batchify_video_example():
    """
    Test batchify_input_stream a video-like setup:
    - read only one frame/sample each call and use an upper bound of fps
    """
    input_shape = (1, 100, 100, 3)
    upper_bound_fps = 16
    out_batch_size = 7

    frame_time = 0

    def read_callback():
        nonlocal frame_time
        frame_time += 1 / upper_bound_fps
        return torch.randint(0, 255, input_shape, dtype=torch.uint8), frame_time

    (
        (processing_time, processing_data),
        (
            chunk_part_for_next_times,
            chunk_part_for_next,
        ),
    ) = batchify_input_stream(
        read_callback=read_callback,
        out_batch_size=out_batch_size,
        input_shape=input_shape,
        sampling_rate=out_batch_size,
        chunk_part_for_next_times=torch.empty(0),
        chunk_part_for_next=torch.empty((0, *input_shape[1:])),
        dtype=torch.uint8,
        upper_bound_fps=upper_bound_fps,
        last_frame_time=0,
    )

    t_delta = processing_time[1:] - processing_time[:-1]

    # check if the time between each sample is equal to the upper bound of fps
    assert torch.allclose(
        t_delta, (1 / upper_bound_fps) * torch.ones_like(t_delta), atol=1e-03
    )
    # check if the data and time have the correct shape
    assert processing_data.shape == (out_batch_size, *input_shape[1:])
    assert len(processing_time) == out_batch_size


def test_batchify_audio_example():
    """
    Test batchify_input_stream an audio-like setup:
    - read 10 samples each call and without an upper bound of fps
    """
    input_shape = (10,)
    out_batch_size = 50
    sampling_rate = 44100
    chunk_counter = 0

    def read_callback():
        nonlocal chunk_counter
        chunk_counter += 1
        # simulate time for the last sample in this chunk
        chunk_end_time = chunk_counter * input_shape[0] / sampling_rate
        return torch.randint(0, 255, input_shape, dtype=torch.uint8), chunk_end_time

    (
        (processing_time, processing_data),
        (
            chunk_part_for_next_times,
            chunk_part_for_next,
        ),
    ) = batchify_input_stream(
        read_callback=read_callback,
        out_batch_size=out_batch_size,
        input_shape=input_shape,
        sampling_rate=sampling_rate,
        chunk_part_for_next_times=torch.empty(0),
        chunk_part_for_next=torch.empty((0, *input_shape[1:])),
        dtype=torch.uint8,
    )

    t_delta = processing_time[1:] - processing_time[:-1]

    # check if the time between each sample is equal to 1 / sampling_rate except for the ones from different chunks
    assert (
        torch.count_nonzero(torch.abs(t_delta - 1 / sampling_rate) > 1e-06)
        <= out_batch_size // input_shape[0]
    )

    # check if the data and time have the correct shape
    assert processing_data.shape == (out_batch_size, *input_shape[1:])
    assert len(processing_time) == out_batch_size


def test_batchify_audio_example_with_chunk_part():
    """
    Test batchify_input_stream an audio-like setup:
    - read 10 samples each call and without an upper bound of fps
    - input the part of chunks which were not processed in the last call
    """
    input_shape = (10,)
    out_batch_size = 50
    sampling_rate = 44100

    def read_callback():
        # return ones for new chunks
        return torch.ones(input_shape, dtype=torch.uint8), 0.1

    (
        (processing_time, processing_data),
        (
            chunk_part_for_next_times,
            chunk_part_for_next,
        ),
    ) = batchify_input_stream(
        read_callback=read_callback,
        out_batch_size=out_batch_size,
        input_shape=input_shape,
        sampling_rate=sampling_rate,
        chunk_part_for_next_times=torch.ones(5) * 0.5,
        chunk_part_for_next=torch.ones(5) * 2,
        dtype=torch.uint8,
    )

    # the part of the chunk which was not processed in the last call should be at the beginning of the batch and the rest at the end
    assert torch.all(processing_data[:5] == 2)
    assert torch.all(processing_time[:5] == 0.5)
    assert torch.all(processing_data[5:] == 1)

    # check if the chunk part returned for the next call is correct
    assert chunk_part_for_next.shape[0] == 5
    assert chunk_part_for_next_times.shape[0] == 5

    # check if the data and time have the correct shape
    assert processing_data.shape == (out_batch_size, *input_shape[1:])
    assert len(processing_time) == out_batch_size
