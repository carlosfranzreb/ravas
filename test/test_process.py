from multiprocessing import Process
import time
import unittest
import numpy as np

import torch
from steam_processing.Processor import Processor, ProcessingQueues, ProcessingSyncState


def init_callback():
    return [2]


def callback(time, data, arg):
    return time, data * arg


class TestProcess(unittest.TestCase):
    def setUp(self):
        self.queues = ProcessingQueues()
        self.own_sync_state = ProcessingSyncState()
        self.external_sync_state = ProcessingSyncState()

        self.processor = Processor(
            queues=self.queues,
            own_sync_state=self.own_sync_state,
            external_sync_state=self.external_sync_state,
            init_callback=init_callback,
            callback=callback,
        )
        self.p = Process(target=self.processor.process)
        self.p.start()
        time.sleep(5)

    def tearDown(self):
        self.p.terminate()

    def test_process_with_callback(self):
        input_time = torch.linspace(10, 20, 100)
        input_data = torch.ones(100, 1)
        self.queues.input_queue.put((input_time, input_data))
        d_time, data = self.queues.sync_queue.get(timeout=5)
        self.assertTrue(
            torch.all(input_time == d_time) and torch.all(input_data * 2 == data)
        )


if __name__ == "__main__":
    unittest.main()
