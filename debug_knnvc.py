"""
Initialize the kNN-VC converter, pass it an audio sample and dump the result.
"""

import yaml
from queue import Empty
import logging
from time import sleep

import torch
import torchaudio
from torch.multiprocessing import Process, Queue

from stream_processing.models import KnnVC


def init_converter(
    config: dict, input_queue: Queue, output_queue: Queue, log_queue: Queue
):
    """Initialize and start the converter."""
    knnvc = KnnVC("audio", config, input_queue, output_queue, log_queue, "DEBUG")
    knnvc.convert()


def listener_process(log_queue, log_level):
    root = logging.getLogger()
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(processName)-10s %(name)s %(levelname)-8s %(message)s"
    )
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)
    root.setLevel(log_level)

    while True:
        while not log_queue.empty():
            record = log_queue.get()
            logger = logging.getLogger(record.name)
            logger.handle(record)
        sleep(1)


def main():
    """Initialize the converter and pass it an audio sample."""

    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    input_queue = Queue()
    output_queue = Queue()
    log_queue = Queue()
    log_listener = Process(
        target=listener_process, args=(log_queue, config["log_level"])
    )
    log_listener.start()

    proc = Process(
        target=init_converter,
        args=(config["audio"]["converter"], input_queue, output_queue, log_queue),
    )
    proc.start()
    sleep(10)
    print("Done sleeping. Converter should be initialized.")

    # chunk an audio sample and place it in the input queue
    audiofile = (
        "/Users/cafr02/datasets/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac"
    )
    audio = torchaudio.load(audiofile)[0].squeeze() * 32768
    print(f"Loaded audio of shape {audio.shape}")
    chunk_size = config["audio"]["processing_size"]
    for chunk_start in range(0, len(audio), chunk_size):
        audio_chunk = audio[chunk_start : chunk_start + chunk_size]
        input_queue.put((0, audio_chunk))

    print("Audio sample placed in queue.")

    # dump the result
    out = list()
    max_wait = 3
    current_wait = 0
    while True:
        try:
            ttime, data = output_queue.get(timeout=1)
            out.append(data)
            current_wait = 0
        except Empty:
            current_wait += 1
            if current_wait == max_wait:
                break

    if len(out) == 0:
        print("No data received.")
    else:
        out = torch.cat(out, dim=0)
        print(f"Received data of shape {out.shape}")
        torchaudio.save("output.wav", out.unsqueeze(0), 16000)

    proc.terminate()
    log_listener.terminate()
    print("Done.")


if __name__ == "__main__":
    main()
