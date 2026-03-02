import pytest
import time
import torch
import numpy as np
from torch.multiprocessing import Event, Queue

from stream_processing.processor import Processor, ProcessingQueues, ProcessingSyncState


@pytest.fixture
def setup_processor():
    """
    Fixture to create processor with mock queues and sync states.
    """
    queues = ProcessingQueues()
    own_sync_state = ProcessingSyncState()
    external_sync_state = ProcessingSyncState()
    config = {"max_unsynced_time": 0, "video_file": None, "converter": {}}

    processor = Processor(
        name="test_processor",
        config=config,
        own_sync_state=own_sync_state,
        external_sync_state=external_sync_state,
        pipeline_sync_state=Event(),
        log_queue=Queue(),
        log_level="DEBUG",
    )
    processor.queues = queues  # override default queues

    return processor, queues, own_sync_state, external_sync_state


def run_sync_once(processor):
    """
    Helper function to run one iteration of sync loop for testing.
    """
    sync_buffer = []

    try:
        dtime, data = processor.queues.sync_queue.get(timeout=0.01)
        sync_buffer.append((dtime, data))
    except Exception:
        pass

    if sync_buffer:
        next_sample_time = sync_buffer[0][0][0].item()
        external_time = processor.external_sync_state.last_sample_time.value + (
            time.time() - processor.external_sync_state.last_update.value
        )
        left_time = next_sample_time - external_time - processor.max_unsynced_time

        ignore_external_sync = (
            processor.converting_file or processor.external_sync_state.disabled.value
        )

        if left_time <= 0 or ignore_external_sync:
            d_time, data = sync_buffer.pop(0)
            if d_time is not None:
                processor.own_sync_state.last_sample_time.value = d_time[0]
                processor.own_sync_state.last_update.value = time.time()
            processor.queues.output_queue.put((d_time, data))


def test_sync_initial_batch(setup_processor):
    """
    Initial sync when external sync state is not updated yet.
    Should forward the batch immediately.
    """
    processor, queues, own_sync_state, external_sync_state = setup_processor

    external_sync_state.last_sample_time.value = np.inf
    external_sync_state.last_update.value = 0

    input_time = torch.linspace(10, 20, 100)
    input_data = torch.ones(100, 1)
    queues.sync_queue.put((input_time, input_data))

    run_sync_once(processor)

    d_time, data = queues.output_queue.get(timeout=1)

    assert torch.all(d_time == input_time)
    assert torch.all(data == input_data)
    assert own_sync_state.last_update.value > 0
    assert own_sync_state.last_sample_time.value == 10


def test_sync_unsynced_future_sample(setup_processor):
    """
    Sample timestamp is in the future compared to external sync state.
    Sample should not be forwarded.
    """
    processor, queues, own_sync_state, external_sync_state = setup_processor

    current_time = time.time()
    external_sync_state.last_sample_time.value = current_time
    external_sync_state.last_update.value = current_time

    queues.sync_queue.put(
        (torch.tensor([current_time + 10], dtype=torch.float64), torch.tensor([1]))
    )

    run_sync_once(processor)

    assert queues.output_queue.empty()


def test_sync_unsynced_past_sample(setup_processor):
    """
    Sample timestamp is in the past compared to external sync state.
    Sample should be forwarded immediately.
    """
    processor, queues, own_sync_state, external_sync_state = setup_processor

    current_time = time.time()
    external_sync_state.last_sample_time.value = current_time - 10
    external_sync_state.last_update.value = current_time - 5

    queues.sync_queue.put(
        (torch.tensor([current_time - 6], dtype=torch.float64), torch.tensor([1]))
    )

    run_sync_once(processor)

    d_time, data = queues.output_queue.get(timeout=1)

    assert d_time.item() == current_time - 6
    assert data.item() == 1
    assert own_sync_state.last_update.value > 0
