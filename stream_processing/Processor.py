from torch.multiprocessing import Process, Queue, Value
import queue
import time
from typing import Any, Callable, Optional, Tuple
import torch
import numpy as np

from stream_processing.utils import clear_queue


class ProcessingQueues:
    def __init__(self):
        self.input_queue = Queue()
        self.sync_queue = Queue()
        self.output_queue = Queue()


class ProcessingSyncState:
    def __init__(self):
        self.last_sample_time = Value("d", np.inf)
        self.last_update = Value("d", 0)


class ProcessingCallback:
    def init_callback(self):
        """
        Callback function that is called for initializing the callback function.
        the function should return a list of arguments that are passed to the callback function.
        """
        raise NotImplementedError

    def callback(self, ttime, data, *args):
        """
        Callback function that is called for processing the data.
        the callback function gets the batched input time and data per sample and should return the batched time and data.
        """
        raise NotImplementedError


class Processor:
    def __init__(
        self,
        queues: ProcessingQueues,
        own_sync_state: ProcessingSyncState,
        external_sync_state: ProcessingSyncState,
        callback: Optional[ProcessingCallback] = None,
        max_unsynced_time: Optional[float] = 0.01,
    ):
        """
        Initialize a Processor object.
        :param queues: ProcessingQueues object that contains all relevant queues for the processor.
        :param own_sync_state: ProcessingSyncState object that contains the sync state of this processor.
        :param external_sync_state: ProcessingSyncState object that contains the sync state of Processor object that is used for sync.
        :param callback: Callback Object that is used for initializing the callback function and the callback function.
        :param max_unsynced_time: Maximum time that the data can be unsynced.
        """
        self.queues = queues
        self.own_sync_state = own_sync_state
        self.external_sync_state = external_sync_state
        self.callback = callback
        self.callback = callback
        self.max_unsynced_time = max_unsynced_time

    def read_input_stream(self):
        """
        Read the input stream and put the data and their corresponding time into the input queue.
        """
        raise NotImplementedError

    def write_output_stream(self):
        """
        Write the data from the final output queue to the output stream.
        """
        raise NotImplementedError

    def process(self):
        """
        Read the input queue and use the callback function to process the data and put the processed data into the sync queue.
        """
        args = self.callback.init_callback() if self.callback is not None else []
        clear_queue(self.queues.input_queue)
        while True:
            try:
                ttime, data = self.queues.input_queue.get()
                if self.callback is not None:
                    out = self.callback.callback(ttime, data, *args)
                    if out is not None:
                        ttime, data = out
                    else:
                        continue
                self.queues.sync_queue.put((ttime, data))
            except queue.Empty:
                pass

    def sync(self):
        """
        Use the external sync state to sync the data of the sync queue and put the synced data into the output queue.
        """
        sync_buffer = []
        clear_queue(self.queues.sync_queue)
        while True:
            left_time = None
            if len(sync_buffer) > 0:
                external_current_play_time = (
                    self.external_sync_state.last_sample_time.value
                    + (time.time() - self.external_sync_state.last_update.value)
                )
                next_sample_time = sync_buffer[0][0][0]
                if (
                    external_current_play_time
                    > next_sample_time - self.max_unsynced_time
                ):
                    d_time, data = sync_buffer.pop(0)
                    self.own_sync_state.last_sample_time.value = dtime[0]
                    self.own_sync_state.last_update.value = time.time()
                    self.queues.output_queue.put((d_time, data))
                else:
                    left_time = (
                        next_sample_time.item()
                        - self.max_unsynced_time
                        - external_current_play_time
                    )
                    if left_time < 0:
                        left_time = None

            try:
                dtime, data = self.queues.sync_queue.get(timeout=left_time)
                sync_buffer.append((dtime, data))
            except queue.Empty:
                pass


class ProcessorProcessHandler:
    def __init__(self, processor: Processor):
        """
        Initialize a ProcessorProcessHandler object.
        :param processor: Processor object that should be handled by the ProcessorProcessHandler.
        """
        self.processor = processor
        # create the processes
        self.read_input_stream_process = Process(
            target=self.processor.read_input_stream
        )
        self.process_process = Process(target=self.processor.process)
        self.sync_process = Process(target=self.processor.sync)
        self.write_output_stream_process = Process(
            target=self.processor.write_output_stream
        )

    def start(self):
        """
        Start all processes.
        """
        self.read_input_stream_process.start()
        self.process_process.start()
        self.sync_process.start()
        self.write_output_stream_process.start()

    def stop(self):
        """
        Terminate all processes.
        """
        self.read_input_stream_process.terminate()
        self.process_process.terminate()
        self.sync_process.terminate()
        self.write_output_stream_process.terminate()
