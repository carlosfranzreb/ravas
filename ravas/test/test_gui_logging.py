import logging
import io
import os
import pytest
from PyQt6.QtCore import QThread

from stream_processing.gui.gui_logging import LogWorker, init_gui_logging

'''
Test the LogWorker and init_gui_logging
This test verifies that:
- LogWorker can be initialized and emits signals correctly.
- init_gui_logging creates a LogWorker, starts a QThread, and the logging process runs
'''

@pytest.fixture
def basic_config():
    return {
        "log_dir": "/test/test_data/test_output/logs",
        "log_level": "INFO",
        "gui_log_level": "INFO",
        "disable_console_logging": False,
        "audio": {},
        "video": {},
    }


def test_logworker_init_and_emit_signal(qtbot):
    """
    Test LogWorker initialization and signal emission together.
    """
    worker = LogWorker()

    # verify queue exists
    assert worker.log_queue is not None

    received = []
    worker.emitLogLine.connect(received.append)

    worker.emitLogLine.emit("test message")

    assert received == ["test message"]


def test_init_gui_logging_thread_and_process(qtbot, basic_config):
    """
    Test init_gui_logging creates worker, starts thread and real logging process.
    """
    worker, thread = init_gui_logging(basic_config)

    assert isinstance(worker, LogWorker)
    assert isinstance(thread, QThread)
    assert thread.isRunning()

    # give the worker time to start the logging process
    qtbot.wait(300)

    # send a real log message
    received_logs = []
    worker.emitLogLine.connect(received_logs.append)

    logging.getLogger().info("thread/process startup test")

    qtbot.waitUntil(
        lambda: any("thread/process startup test" in m for m in received_logs),
        timeout=5000,
    )

    # stop thread cleanly
    thread.requestInterruption()
    qtbot.waitUntil(lambda: not thread.isRunning(), timeout=5000)

    assert any("thread/process startup test" in m for m in received_logs)