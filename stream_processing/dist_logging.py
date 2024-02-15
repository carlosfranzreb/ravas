import os
import logging
import logging.handlers
from time import sleep
from random import random, randint
from multiprocessing import Queue


def listener_configurer(log_dir: str):
    """
    Configure the listener process to log to a file and the console.
    """
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
    root.setLevel(logging.DEBUG)
    print(f"Logging to {log_file} and console")


def listener_process(log_dir: str, queue: Queue):
    """
    Configure the listener process.
    """
    listener_configurer(log_dir)
    while True:
        while not queue.empty():
            record = queue.get()
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        sleep(1)


# Same as demo code
def worker_configurer(queue):
    h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)
    # send all messages, for demo; no other level or filter logic applied.
    root.setLevel(logging.DEBUG)


# Almost the same as demo code, except the logging is simplified, and configurer
# is no longer passed as argument.
def worker_process(queue):
    worker_configurer(queue)
    for i in range(3):
        sleep(random())
        innerlogger = logging.getLogger("worker")
        innerlogger.info(f"Logging a random number {randint(0, 10)}")
