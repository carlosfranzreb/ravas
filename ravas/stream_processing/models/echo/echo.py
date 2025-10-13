import queue

from torch.multiprocessing import Queue, Event

from ...processor import Converter
from ...utils import clear_queue


class Echo(Converter):
    def __init__(
        self,
        name: str,
        config: dict,
        input_queue: Queue,
        output_queue: Queue,
        log_queue: Queue,
        log_level: str,
        ready_signal: Event,
    ) -> None:
        """
        Initialize the Echo Model.
        """
        super().__init__(
            name, config, input_queue, output_queue, log_queue, log_level, ready_signal
        )

    def convert(self) -> None:
        """
        put the data from input queue to output queue without any processing.
        """
        self.logger.info("Start converting audio")
        if self.config["video_file"] is None:
            clear_queue(self.input_queue)

        self.ready_signal.set()
        while True:
            try:
                ttime, data = self.input_queue.get(timeout=1)
                if ttime is None and data is None:
                    break
                self.output_queue.put((ttime, data))
            except queue.Empty:
                self.logger.debug("Input queue is empty")
                pass
            except EOFError:
                break
