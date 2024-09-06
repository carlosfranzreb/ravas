"""
HELPER for setting up logging that allows (in addition) displaying the logs in a GUI widget
"""

import logging
import os
from datetime import datetime
from queue import Empty
from time import sleep

import yaml
from PyQt6.QtCore import Qt, QThread, QMetaObject, Q_ARG, QObject, pyqtSignal, pyqtSlot
from torch import multiprocessing

from ..dist_logging import listener_configurer, worker_configurer
from ..utils import resolve_file_path


class LogWorker(QObject):
    """
    a Qt worker that provides Qt signals to allow writing log messages into GUI widget

    in addition, will start logging to console & file in another process, so that logging may not block main thread
    """
    finished = pyqtSignal()
    emitLogLine = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.log_queue = multiprocessing.Queue(-1)

    # DISABLED use thread.requestInterruption() on wrapping QThread instead!
    # @pyqtSlot()
    # def stop(self):
    #     # FIXME signaling this does not work when processing-loop in `startProcessingLog()` is running
    #     #       ... as a WORKAROUND we can use the Event field `stop` directly
    #     print("Worker.stop()", flush=True)  # FIXME DEBUG
    #     self.stop.set()

    @pyqtSlot(dict)
    def startProcessingLog(self, config: dict):
        print("Worker.startProcessingLog()...", flush=True)  # FIXME DEBUG

        # check if already stopped, before creating log-dir & file
        if QThread.currentThread().isInterruptionRequested():
            return

        log_dir = os.path.join(resolve_file_path(config["log_dir"]), datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(log_dir, exist_ok=True)
        yaml.dump(config, open(os.path.join(log_dir, "config.yaml"), "w"))
        config["log_dir"] = log_dir
        config["audio"]["log_dir"] = log_dir
        config["video"]["log_dir"] = log_dir

        log_queue = self.log_queue
        gui_log_queue = multiprocessing.Queue(-1)

        gui_log_level = config.get("gui_log_level", config["log_level"])
        disable_console_logging = config.get("disable_console_logging", True)

        # check if already stopped, before starting logging-process
        if QThread.currentThread().isInterruptionRequested():
            return

        # start logging process for logging to console & file
        log_listener = multiprocessing.Process(target=gui_listener_process, kwargs={
            'log_dir': log_dir,
            'queue': log_queue,
            'gui_queue': gui_log_queue,
            'log_level': config["log_level"],
            'gui_log_level': gui_log_level,
            'disable_console': disable_console_logging,
        })
        log_listener.start()
        worker_configurer(log_queue, config["log_level"])
        logging.getLogger().info('Started logging process (pid %s)', log_listener.pid)

        while not QThread.currentThread().isInterruptionRequested():
            try:
                msg = gui_log_queue.get(timeout=1)
                self.emitLogLine.emit(msg)
            except Empty:
                pass

        print("Worker.startProcessingLog(): stopped!, terminating logger process now", flush=True)  # FIXME DEBUG
        log_listener.terminate()

        # send finish signal (if connected to thread.quit, will cause wrapper-thread to end)
        self.finished.emit()
        print("Worker.startProcessingLog(): emitted finish signal", flush=True)  # FIXME DEBUG


def gui_listener_process(
        log_dir: str,
        queue: multiprocessing.Queue,
        gui_queue: multiprocessing.Queue,
        log_level: str = "INFO",
        gui_log_level: str = "INFO",
        disable_console: bool = True,
):
    """
    Configure and start the listener process.

    adapted from `dist_logging.py`'s `listener_process()`:
    uses the logger-configuration from `dist_logging.py` (see `listener_configurer()`) and
    will also send the log-entries to `gui_queue` in addition to handling them
    (i.e. emitting to file; console output will be disabled)

    :param log_dir: the directory for the log-file
    :param queue: the queue for process-loggers
    :param gui_queue: the queue for sending messages to gui-logging worker
    :param log_level: the log-level for file/console logging (and used in processes)
    :param gui_log_level: the log-level for the GUI widget
                          _NOTE_ should either be same as `log_level` or higher (i.e. more restrictive); a lower
                                 value than `log_level` will have no effect
    :param disable_console: if `True` logging to console will be disabled
    """
    listener_configurer(log_dir, log_level, disable_console=disable_console)

    if isinstance(gui_log_level, str):
        gui_log_level = gui_log_level.upper()
        if hasattr(logging, gui_log_level):
            gui_log_level = getattr(logging, gui_log_level)
        if isinstance(gui_log_level, str):
            print('ERROR setting up logger: failed to detect log-level (int) for GUI logging for value "%s", using INFO instead' % gui_log_level, flush=True)
            gui_log_level = logging.INFO

    # handler for creating the log-line for gui-logging here
    # (and send the formatted log-string via `emitLogLine` signal)
    handler = logging.Handler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(processName)-10s %(process)-8d %(name)s %(levelname)-8s %(message)s"
    ))

    while True:
        while not queue.empty():
            record: logging.LogRecord = queue.get()
            logger = logging.getLogger(record.name)
            logger.handle(record)
            if record.levelno >= gui_log_level:
                msg = handler.format(record)
                gui_queue.put(msg)
        sleep(1)


def init_gui_logging(config: dict) -> tuple[LogWorker, QThread]:

    thread = QThread()
    log_worker = LogWorker()
    log_worker.moveToThread(thread)
    log_worker.finished.connect(thread.quit)

    # start thread & invoke worker's processing function
    thread.start()
    QMetaObject.invokeMethod(log_worker, 'startProcessingLog', Qt.ConnectionType.QueuedConnection, Q_ARG(dict, config))

    # NOTE must return the QThread and attach it to app (i.e. keep a reference) otherwise it will be garbage-collected,
    #      which would prematurely stop the thread!
    return log_worker, thread
