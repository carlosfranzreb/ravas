from torch.multiprocessing import Process
import time
import unittest
import numpy as np

import torch
from ..stream_processing.processor import (
    Processor,
    ProcessingQueues,
    ProcessingSyncState,
)


class TestSync(unittest.TestCase):
    """
    Test the sync function of the Processor class
    This function should:
    - get data from the sync queue
    - sync the data to the external sync state
    - put the synced data into the output queue
    """

    def setUp(self):
        """
        Setup the queues and the processor and start the process
        """
        self.queues = ProcessingQueues()
        self.own_sync_state = ProcessingSyncState()
        self.external_sync_state = ProcessingSyncState()
        self.processor = Processor(
            queues=self.queues,
            own_sync_state=self.own_sync_state,
            external_sync_state=self.external_sync_state,
        )
        self.p = Process(target=self.processor.sync)
        self.p.start()
        time.sleep(5)

    def tearDown(self):
        """
        Terminate the process
        """
        self.p.terminate()

    def test_sync_queue(self):
        """
        Test the sync function with a external sync state which was not updated yet
        This is the case when the sync function is called for the first time
        The function should return the data from the sync queue without any changes
        """
        self.external_sync_state.last_sample_time.value = np.inf
        self.external_sync_state.last_update.value = 0
        input_time = torch.linspace(10, 20, 100)
        input_data = torch.ones(100, 1)
        self.queues.sync_queue.put((input_time, input_data))
        d_time, data = self.queues.output_queue.get(timeout=5)

        # check if the data and time are the same
        self.assertTrue(
            torch.all(input_time == d_time) and torch.all(input_data == data)
        )
        # check if the last update of the own sync state is updated
        self.assertTrue(time.time() - self.own_sync_state.last_update.value < 0.1)
        # check if the last sample time of the own sync state is set to the first sample from the batch
        self.assertTrue(self.own_sync_state.last_sample_time.value == 10)

    def test_sync_unsynced_positive(self):
        """
        Test the sync function with a external sync state which was updated
        and the timestamp of the currently played sample is smaller than the timestamp of the first sample in the batch
        The function should wait until the timestamp change
        """
        current_time = time.time()
        self.external_sync_state.last_sample_time.value = current_time - 10
        self.external_sync_state.last_update.value = current_time - 5
        self.queues.sync_queue.put(
            (torch.tensor([current_time - 1], dtype=torch.float64), torch.tensor([1]))
        )
        time.sleep(0.1)
        # check if the output queue is empty
        assert self.queues.output_queue.empty()

    def test_sync_unsynced_negative(self):
        """
        Test the sync function with a external sync state which was updated
        and the timestamp of the currently played sample is larger than the timestamp of the first sample in the batch
        The function should return the data from the sync queue without any changes
        """
        current_time = time.time()
        self.external_sync_state.last_sample_time.value = current_time - 10
        self.external_sync_state.last_update.value = current_time - 5
        self.queues.sync_queue.put(
            (torch.tensor([current_time - 6], dtype=torch.float64), torch.tensor([1]))
        )
        # check if the function returns the data from the sync queue to the output queue
        try:
            _, _ = self.queues.output_queue.get(timeout=5)
        except:
            self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
