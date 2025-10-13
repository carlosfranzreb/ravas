from torch.multiprocessing import Process
import time
import unittest

import torch
from ..stream_processing.processor import (
    ProcessingCallback,
    Processor,
    ProcessingQueues,
    ProcessingSyncState,
)


class TestCallback(ProcessingCallback):
    def init_callback(self):
        return [2]

    def callback(self, time, data, arg):
        return time, data * arg


class TestProcess(unittest.TestCase):
    """
    Test the process function of the Processor class
    This function should:
    - get data from the input queue
    - process the data by applying the callback function with the args returned by the init_callback function
    - put the processed data into the sync queue
    Note:
    - the callbacks have to be defined outside of the test class because they are pickled and unpickled when the process is started
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
            callback=TestCallback(),
        )
        self.p = Process(target=self.processor.process)
        self.p.start()
        time.sleep(5)

    def tearDown(self):
        """
        Terminate the process
        """
        self.p.terminate()

    def test_process_with_callback(self):
        """
        Test the process function with a callback function
        """
        input_time = torch.linspace(10, 20, 100)
        input_data = torch.ones(100, 1)
        self.queues.input_queue.put((input_time, input_data))
        d_time, data = self.queues.sync_queue.get(timeout=5)

        # check if the time is unchanged
        self.assertTrue(torch.all(input_time == d_time))
        # check if the data is processed with the callback function correctly
        self.assertTrue(torch.all(input_data * 2 == data))


if __name__ == "__main__":
    unittest.main()
