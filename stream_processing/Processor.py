import queue
import time
import logging
import importlib

from torch import Tensor
import numpy as np
from torch.multiprocessing import Process, Queue, Value

from stream_processing.utils import clear_queue
from stream_processing.dist_logging import worker_configurer


class ProcessingQueues:
    """
    There are three queues used by the processor:
    1. input_queue: Queue for the input stream. The "read" process adds data to this
        queue, and the "convert" process reads from it.
    2. sync_queue: Queue for the sync stream. The "convert" process adds data to this
        queue, and the "sync" process reads from it.
    3. output_queue: Queue for the output stream. The "sync" process adds data to this
        queue, and the "write" process reads from it.
    """

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


class Converter:
    """
    Object that contains the conversion function and its parameters.
    """

    def __init__(
        self,
        name: str,
        config: dict,
        input_queue: Queue,
        output_queue: Queue,
        log_queue: Queue,
        log_level: str,
    ) -> None:
        """
        Initialize the Converter object.

        Args:
            name: Name of the processor that uses the converter.
            config: The config for the converter.
            input_queue: Queue for the input stream.
            output_queue: Queue for the sync stream.
            log_queue: log queue for logging messages.
            log_level: Log level for logging messages.
        """
        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue

        # setup logging
        worker_configurer(log_queue, log_level)
        self.logger = logging.getLogger(f"{name}_converter_process")
        self.logger.info(f"{name} converter initialized")

    def convert(self) -> tuple[Tensor, Tensor]:
        """
        Read the input queue, convert the data and put the converted data into the
        sync queue.
        """
        raise NotImplementedError


class Processor:
    def __init__(
        self,
        name: str,
        config: dict,
        own_sync_state: ProcessingSyncState,
        external_sync_state: ProcessingSyncState,
        log_queue: Queue,
        log_level: str,
    ):
        """
        Initialize a Processor object. It includes four processes:
        1. "read": reads the input stream and puts the data and their corresponding
            time into the input queue.
        2. "convert": reads the input queue, convertes the data and puts the converted
            data into the sync queue.
        3. "sync": uses the external sync state to sync the data of the sync queue and
            puts the synced data into the output queue.
        4. "write": writes the data from the final output queue to the output stream.

        :param name: Name of the processor.
        :param config: Config for the processor.
        :param own_sync_state: ProcessingSyncState object that contains the sync state
            of this processor.
        :param external_sync_state: ProcessingSyncState object that contains the sync
            state of Processor object that is used for sync.
        :param log_queue: Queue for logging, e.g. used in the write_output_stream
            function to log delays.
        :param log_level: Log level for logging messages.
        """
        self.name = name
        self.config = config
        self.queues = ProcessingQueues()
        self.own_sync_state = own_sync_state
        self.external_sync_state = external_sync_state
        self.max_unsynced_time = config["max_unsynced_time"]
        self.log_queue = log_queue
        self.log_level = log_level

    def read(self):
        """
        Read the input stream and put the data and their corresponding time into the
        input queue.
        """
        raise NotImplementedError

    def write(self):
        """
        Write the data from the final output queue to the output stream.
        """
        raise NotImplementedError

    def convert(self):
        """
        Initialize and start the converter.
        """
        converter_cls = get_cls(self.config["converter"]["cls"])
        self.converter = converter_cls(
            self.name,
            self.config["converter"],
            self.queues.input_queue,
            self.queues.sync_queue,
            self.log_queue,
            self.log_level,
        )
        self.converter.convert()

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

        # setup logging
        worker_configurer(self.log_queue, self.log_level)
        logger = logging.getLogger(f"{self.name}_sync")
        logger.info("Syncing initialized")

        def get_left_time() -> float:
            """Return the timestamp where the output should be at the current time."""
            if len(sync_buffer) == 0:
                logger.debug("Sync buffer is empty")
                return None
            external_current_play_time = (
                self.external_sync_state.last_sample_time.value
                + (time.time() - self.external_sync_state.last_update.value)
            )
            next_sample_time = sync_buffer[0][0][0].item()
            left_time = (
                next_sample_time - external_current_play_time - self.max_unsynced_time
            )
            logger.debug(f"time left until sync: {round(left_time, 2)} s")
            return left_time

        sync_buffer = []
        clear_queue(self.queues.sync_queue)
        while True:
            left_time = get_left_time()

            # if there is a sample to sync, sync it
            if left_time is not None and left_time <= 0:
                logger.debug("Syncing sample")
                d_time, data = sync_buffer.pop(0)
                self.own_sync_state.last_sample_time.value = d_time[0]
                self.own_sync_state.last_update.value = time.time()
                self.queues.output_queue.put((d_time, data))
                left_time = get_left_time()

            # if there is a sample to sync, skip the fetching
            if left_time is not None and left_time < 0:
                continue

            # fetch the next sample
            try:
                dtime, data = self.queues.sync_queue.get(timeout=left_time)
                sync_buffer.append((dtime, data))
            except queue.Empty:
                pass


class ProcessorHandler:
    def __init__(self, processor: Processor):
        """
        Initialize a ProcessorHandler object. It starts all four processes of
        the Processor object.
        :param processor: Processor object that should be handled by the
            ProcessorProcessHandler.
        """
        self.processor = processor
        self.read_process = Process(target=self.processor.read)
        self.convert_process = Process(target=self.processor.convert)
        self.sync_process = Process(target=self.processor.sync)
        self.write_process = Process(target=self.processor.write)
        self.procs = [
            self.read_process,
            self.convert_process,
            self.sync_process,
            self.write_process,
        ]

    def start(self):
        """Start all processes."""
        for proc in self.procs:
            proc.start()

    def stop(self):
        """Terminate all processes."""
        for proc in self.procs:
            proc.terminate()


def get_cls(cls_str: str):
    """
    Import the module and return the class. `cls_str` should be in the format
    `module.class`.
    """
    module_str, cls_str = cls_str.rsplit(".", 1)
    module = importlib.import_module(module_str)
    return getattr(module, cls_str)
