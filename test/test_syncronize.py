from multiprocessing import Process
import time
import unittest
import numpy as np

import torch
from steam_processing.Processor import Processor, ProcessingQueues, ProcessingSyncState


class TestSync(unittest.TestCase):
    def setUp(self):
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
        self.p.terminate()

    def test_sync_queue(self):
        self.external_sync_state.last_sample_time.value = np.inf
        self.external_sync_state.last_update.value = 0
        input_time = torch.linspace(10, 20, 100)
        input_data = torch.ones(100, 1)
        self.queues.sync_queue.put((input_time, input_data))
        d_time, data = self.queues.output_queue.get(timeout=5)
        self.assertTrue(
            torch.all(input_time == d_time) and torch.all(input_data == data)
        )
        self.assertTrue(time.time() - self.own_sync_state.last_update.value < 0.1)
        self.assertTrue(self.own_sync_state.last_sample_time.value == 10)

    def test_sync_unsynced_positive(self):
        current_time = time.time()
        self.external_sync_state.last_sample_time.value = current_time - 10
        self.external_sync_state.last_update.value = current_time - 5
        self.queues.sync_queue.put(
            (torch.tensor([current_time - 1], dtype=torch.float64), torch.tensor([1]))
        )
        time.sleep(0.1)
        assert self.queues.output_queue.empty()

    def test_sync_unsynced_negative(self):
        current_time = time.time()
        self.external_sync_state.last_sample_time.value = current_time - 10
        self.external_sync_state.last_update.value = current_time - 5
        self.queues.sync_queue.put(
            (torch.tensor([current_time - 6], dtype=torch.float64), torch.tensor([1]))
        )
        try:
            _, _ = self.queues.output_queue.get(timeout=5)
        except:
            self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
