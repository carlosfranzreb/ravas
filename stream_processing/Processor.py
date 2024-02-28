import queue
import time
from typing import Optional
import logging

import numpy as np
from torch.multiprocessing import Process, Queue, Value

from stream_processing.utils import clear_queue
from stream_processing.dist_logging import worker_configurer


class ProcessingQueues:
    def __init__(self):
        self.input_queue = Queue()
        self.sync_queue = Queue()
        self.output_queue = Queue()


class ProcessingSyncState:
    def __init__(self):
        """
        Sync state of the processor object.

        last_sample_time: Timestamp of the last sample that was synced.
        last_update: Timestamp of the last update of `last_sample_time`.
        """
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
        log_queue: Optional[Queue] = None,
    ):
        """
        Initialize a Processor object.
        :param queues: ProcessingQueues object that contains all relevant queues for
            the processor.
        :param own_sync_state: ProcessingSyncState object that contains the sync state
            of this processor.
        :param external_sync_state: ProcessingSyncState object that contains the sync
            state of Processor object that is used for sync.
        :param callback: Callback Object that is used for initializing the callback
            function and the callback function.
        :param max_unsynced_time: Maximum time that the data can be unsynced.
        :param log_queue: Queue for logging, e.g. used in the write_output_stream
            function to log delays.
        """
        self.queues = queues
        self.own_sync_state = own_sync_state
        self.external_sync_state = external_sync_state
        self.callback = callback
        self.max_unsynced_time = max_unsynced_time
        self.log_queue = log_queue

        # setup logging
        worker_configurer(self.log_queue)
        self.logger = logging.getLogger("worker")
        self.logger.info("Processor initialized")

    def read_input_stream(self):
        """
        Read the input stream and put the data and their corresponding time into the
        input queue.
        """
        raise NotImplementedError

    def write_output_stream(self):
        """
        Write the data from the final output queue to the output stream.
        """
        raise NotImplementedError

    def process(self):
        """
        Read the input queue and use the callback function to process the data and put
        the processed data into the sync queue.
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
        Use the external sync state to sync the data of the sync queue and put the
        synced data into the output queue. We check if the oldest batch is ready to be
        synced by checking if its earliest timestamp is smaller than the last update
        of the other stream, minus the max. unsync allowance. If it is, we pop the
        corresponding sample from the sync queue and put it into the output queue. If
        not, we pass the time left until it is ready to be synced to the sync queue's
        `get` method as a timeout.
        """

        def get_left_time(sync_buffer: list) -> float:
            """Return the timestamp where the output should be at the current time."""
            if len(sync_buffer) == 0:
                return None
            external_current_play_time = (
                self.external_sync_state.last_sample_time.value
                + (time.time() - self.external_sync_state.last_update.value)
            )
            next_sample_time = sync_buffer[0][0][0].item()

            # log times for debugging before returning
            left_time = (
                next_sample_time - external_current_play_time - self.max_unsynced_time
            )
            self.logger.debug(
                f"external play time: {round(external_current_play_time, 2)} s"
            )
            self.logger.debug(
                f"start time of next batch: {round(next_sample_time, 2)} s"
            )
            self.logger.debug(f"time left until sync: {round(left_time, 2)} s")
            return left_time

        sync_buffer = []
        clear_queue(self.queues.sync_queue)
        while True:
            left_time = get_left_time(sync_buffer)
            if left_time is not None and left_time <= 0:
                d_time, data = sync_buffer.pop(0)
                self.own_sync_state.last_sample_time.value = d_time[0]
                self.own_sync_state.last_update.value = time.time()
                self.queues.output_queue.put((d_time, data))
                left_time = get_left_time(sync_buffer)

            if left_time is not None and left_time > 0:
                try:
                    dtime, data = self.queues.sync_queue.get(timeout=left_time)
                    sync_buffer.append((dtime, data))
                except queue.Empty:
                    pass


class ProcessorProcessHandler:
    def __init__(self, processor: Processor):
        """
        Initialize a ProcessorProcessHandler object.
        :param processor: Processor object that should be handled by the
            ProcessorProcessHandler.
        """
        self.processor = processor
        self.read_input_stream_process = Process(
            target=self.processor.read_input_stream
        )
        self.process_process = Process(target=self.processor.process)
        self.sync_process = Process(target=self.processor.sync)
        self.write_output_stream_process = Process(
            target=self.processor.write_output_stream
        )
        self.procs = [
            self.read_input_stream_process,
            self.process_process,
            self.sync_process,
            self.write_output_stream_process,
        ]

    def start(self):
        """Start all processes."""
        for proc in self.procs:
            proc.start()

    def stop(self):
        """Terminate all processes."""
        for proc in self.procs:
            proc.terminate()
