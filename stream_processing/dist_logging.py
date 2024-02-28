import os
import logging
import logging.handlers
from time import sleep
from multiprocessing import Queue


def listener_configurer(log_dir: str, log_level: str = "INFO"):
    """Configure the root listener process to log to a file and the console."""
    log_file = os.path.join(log_dir, "progress.log")
    root = logging.getLogger()
    file_handler = logging.FileHandler(log_file, "a")
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s"
    )
    for handler in (file_handler, console_handler):
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(log_level)
    print(f"Logging to {log_file} and console")


def listener_process(log_dir: str, queue: Queue, log_level: str = "INFO"):
    """Configure the listener process."""
    listener_configurer(log_dir, log_level)
    while True:
        while not queue.empty():
            record = queue.get()
            logger = logging.getLogger(record.name)
            logger.handle(record)
        sleep(1)


def worker_configurer(queue: Queue, log_level: str = "DEBUG"):
    """Add a queue handler to the root logger."""
    queue_handler = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(queue_handler)
    root.setLevel(log_level)
