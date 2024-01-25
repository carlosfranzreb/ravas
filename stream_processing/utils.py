import queue
import time
from typing import Callable, Optional
import torch


def batchify_input_stream(
    read_callback: Callable[[], torch.Tensor],
    out_batch_size: int,
    input_shape: tuple,
    sampling_rate: int,
    chunk_part_for_next_times: Optional[torch.Tensor],
    chunk_part_for_next: Optional[torch.Tensor],
    dtype: torch.dtype,
    upper_bound_fps: Optional[int] = None,
    last_frame_time: float = 0,
):
    """
    Read from the input stream and batch the data. This function can be used for audio and video streams.
    :param read_callback: Callback function that reads from the input stream and returns the data.
    :param size of the desired output batch.
    :param input_shape: Shape of the input data returned by the read_callback.
    :param sampling_rate: Sampling rate of the input stream.
    :param chunk_part_for_next_times: Part of the last chunk that was not used in the last batch.
    :param chunk_part_for_next: Part of the last chunk that was not used in the last batch.
    :param dtype: dtype of the output data.
    :param upper_bound_fps: Upper bound of the fps of the input stream. Only if the chunk size of the input stream is 1
    :param last_frame_time: Time of the last frame. Only if upper_bound_fps is not None.
    :return: Batched data and the remaining part of the last chunk. (batched_time, batched_data), (chunk_part_for_next_times, chunk_part_for_next)
    """
    assert upper_bound_fps is None or input_shape[0] == 1

    out_shape = (out_batch_size, *input_shape[1:])

    new_chunk_part_for_next = None
    new_chunk_part_for_next_times = None

    batched_data = torch.zeros(out_shape, dtype=dtype)
    batched_time = torch.zeros(out_batch_size, dtype=torch.float64)
    num_samples_in_batched_data = 0

    # if last chunk was not fully used in last batch add it to the new batch
    if chunk_part_for_next_times is not None:
        batched_data[: len(chunk_part_for_next)] = chunk_part_for_next
        batched_time[: len(chunk_part_for_next_times)] = chunk_part_for_next_times
        num_samples_in_batched_data = len(chunk_part_for_next)

    while num_samples_in_batched_data < out_batch_size:
        chunk = read_callback()
        in_chunk_size = len(chunk)
        # if a lower fps is desired, wait until the desired time has passed
        if upper_bound_fps is not None:
            if time.time() - last_frame_time < 1 / upper_bound_fps:
                continue

        # calculate the time corosponding to each sample in current chunk
        chunk_end_time = time.time()
        last_frame_time = chunk_end_time
        chunk_start_time = chunk_end_time - (in_chunk_size - 1) / sampling_rate

        chunk_times = torch.linspace(
            chunk_start_time, chunk_end_time, in_chunk_size, dtype=torch.float64
        )

        # if the chunk is larger than the desired batch size, split the chunk and save the rest for the next batch
        if in_chunk_size + num_samples_in_batched_data > out_batch_size:
            missing_chunk_size = out_batch_size - num_samples_in_batched_data
            new_chunk_part_for_next = chunk[missing_chunk_size:]
            new_chunk_part_for_next_times = chunk_times[missing_chunk_size:]
            chunk = chunk[:missing_chunk_size]
            chunk_times = chunk_times[:missing_chunk_size]

        # add the current chunk to the batch
        batched_data[
            num_samples_in_batched_data : num_samples_in_batched_data + len(chunk)
        ] = chunk
        batched_time[
            num_samples_in_batched_data : num_samples_in_batched_data + len(chunk)
        ] = chunk_times
        num_samples_in_batched_data += len(chunk)

    return (batched_time, batched_data), (
        new_chunk_part_for_next_times,
        new_chunk_part_for_next,
    )


def clear_queue(q: queue.Queue):
    """
    Clear the queue.
    :param q: Queue to clear.
    """
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass
