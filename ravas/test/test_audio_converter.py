import pytest
from queue import Queue
from threading import Event
import logging

from stream_processing.processor import AudioConverter

"""
Test the `AudioConverter` class.

This test focuses on the general converter workflow and queue handling,
not the actual audio conversion.

It verifies that:
- The `convert_audio` method is being mocked to an identity function to test
  the pipeline flow without performing real signal processing.
- The `ready` signal is correctly set after processing.
- The output queue receives the expected data.
- Key log messages are emitted during processing.
"""


@pytest.fixture
def audio_converter():
    input_queue = Queue()
    output_queue = Queue()
    log_queue = Queue()
    ready_signal = Event()
    config = {"video_file": "dummy.mp4"}

    conv = AudioConverter(
        name="test",
        config=config,
        input_queue=input_queue,
        output_queue=output_queue,
        log_queue=log_queue,
        log_level="INFO",
        ready_signal=ready_signal,
    )

    # Mock convert_audio to track calls
    conv.convert_audio = lambda x: x  # Identity function for testing
    return conv, input_queue, output_queue, ready_signal


def test_queue_and_logs(audio_converter, caplog):
    conv, input_queue, output_queue, ready_signal = audio_converter

    # capture logging INFO level logs
    caplog.set_level(logging.INFO)
    # Simulate putting audio stream in the input queue
    input_queue.put(([0], 10))
    input_queue.put(([1], 20))
    input_queue.put(([2], None))  # stop signal

    conv.convert()

    assert ready_signal.is_set()  # ready signal should be set after processing

    ttime1, data1 = output_queue.get()
    ttime2, data2 = output_queue.get()
    ttime3, data3 = output_queue.get()

    assert (ttime1, data1) == ([0], 10)
    assert (ttime2, data2) == ([1], 20)
    assert (ttime3, data3) == (None, None)

    log_messages = [record.getMessage() for record in caplog.records]
    assert any("Start converting audio" in m for m in log_messages)
    assert any("Data is null, stopping conversion" in m for m in log_messages)
