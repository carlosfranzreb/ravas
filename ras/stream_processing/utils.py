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
    :param out_batch_size: size of the desired output batch.
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

    if chunk_part_for_next_times is None and chunk_part_for_next is None:
        # during the last batch, the stream was finished
        return (None, None), (None, None)

    out_shape = (out_batch_size, *input_shape[1:])

    new_chunk_part_for_next = torch.empty((0, *input_shape[1:]))
    new_chunk_part_for_next_times = torch.empty((0))

    batched_data = torch.zeros(out_shape, dtype=dtype)
    batched_time = torch.zeros(out_batch_size, dtype=torch.float64)
    num_samples_in_batched_data = 0

    # if last chunk was not fully used in last batch add it to the new batch
    if len(chunk_part_for_next_times) > 0:
        batched_data[: len(chunk_part_for_next)] = chunk_part_for_next
        batched_time[: len(chunk_part_for_next_times)] = chunk_part_for_next_times
        num_samples_in_batched_data = len(chunk_part_for_next)

    while num_samples_in_batched_data < out_batch_size:
        chunk, chunk_end_time = read_callback()
        # if the stream is finished
        if chunk is None and chunk_end_time is None:
            # if there is still data in the batch, return it
            if num_samples_in_batched_data > 0:
                return (
                    batched_time[:num_samples_in_batched_data],
                    batched_data[:num_samples_in_batched_data],
                ), (
                    None,
                    None,
                )
            return (None, None), (None, None)
        in_chunk_size = len(chunk)
        # if a lower fps is desired, wait until the desired time has passed
        if upper_bound_fps is not None:
            if time.time() - last_frame_time < 1 / upper_bound_fps:
                continue

        # calculate the time corresponding to each sample in current chunk
        last_frame_time = chunk_end_time
        chunk_start_time = chunk_end_time - (in_chunk_size - 1) / sampling_rate

        chunk_times = torch.linspace(
            chunk_start_time, chunk_end_time, in_chunk_size, dtype=torch.float64
        )

        # if the chunk is larger than the desired batch size, split the chunk and save
        # the rest for the next batch
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


def kill_all_child_processes(pid: int | None = None, recursive: bool = True, verbose: bool = False):
    """
    Kill all remaining child processes (for `pid` or of the processes from which this function is called).

    Helper to ensure that no processes remaining running, after exiting the main module:
    you may use this before closing/exiting the python program.

    __NOTE__ that this will force the processes to quit, i.e. does NOT terminate the processes gracefully!

    :param pid: the PID for parent process, for which the children should be terminated,
                if omitted uses the current process
    :param recursive: if `True`, terminate child processes (of child processes) recursively
    :param verbose: if `True`, print information of terminated/killed processes to console
    """
    import psutil

    # code adapted from
    # https://psutil.readthedocs.io/en/latest/#processes

    def on_terminate(proc):
        if verbose:
            print("  process {} terminated with exit code {}".format(proc, proc.returncode), flush=True)

    main_proc = psutil.Process(pid=pid)
    procs = main_proc.children()

    if verbose:
        print('Found {} child processes (for current process with PID {})'.format(len(procs), main_proc.pid), flush=True)

    if len(procs) == 0:
        return

    count_terminate = 0
    for p in procs:
        if recursive:
            # recursively kill child processes:
            kill_all_child_processes(p.pid, recursive=recursive, verbose=verbose)
        if p.is_running():
            if verbose:
                print('  terminating child process with PID {}'.format(p.pid), flush=True)
            count_terminate += 1
            p.terminate()

    gone, alive = psutil.wait_procs(procs, timeout=3, callback=on_terminate)

    count_kill = 0
    for p in alive:
        if verbose:
            print('  KILL remaining child process with PID {} (forced termination)'.format(p.pid), flush=True)
        count_kill += 1
        p.kill()

    if verbose:
        print('Terminated {} child processes (forced termination for {} child processes)'.format(count_terminate, count_kill), flush=True)
        procs = main_proc.children()
        if len(procs) > 0:
            print('  remaining child processes: {}\n'.format(len(procs)), flush=True)
