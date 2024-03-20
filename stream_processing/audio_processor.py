from queue import Queue
import logging
import time
import os

import pyaudio
import sounddevice as sd
import torch

import wave

from stream_processing.processor import Processor, ProcessingSyncState
from stream_processing.utils import batchify_input_stream, clear_queue
from stream_processing.dist_logging import worker_configurer


class AudioProcessor(Processor):
    def __init__(
        self,
        config: dict,
        audio_sync_state: ProcessingSyncState,
        external_sync_state: ProcessingSyncState,
        log_queue: Queue,
        log_level: str,
    ):
        """
        Initialize a AudioProcessor object.
        :param name: Name of the processor.
        :param config: The config for the processor.
        :param audio_sync_state: ProcessingSyncState containing the audio sync state.
        :param external_sync_state: ProcessingSyncState containing the external sync state to sync the audio.
        :param log_queue: log queue for logging messages.
        :param log_level: log level for logging messages.
        """
        super().__init__(
            "audio", config, audio_sync_state, external_sync_state, log_queue, log_level
        )
        self.audio_sync_state = audio_sync_state
        self.external_sync_state = external_sync_state

        self.sampling_rate = config["sampling_rate"]
        self.record_buffersize = config["record_buffersize"]
        self.input_device = get_device_idx(config["input_device"], True)
        self.processing_size = config["processing_size"]
        self.output_device = get_device_idx(config["output_device"], False)
        self.output_buffersize = config["output_buffersize"]
        self.store = config["store"]

        if self.store:
            self.store_path = os.path.join(config["log_dir"], "audio.wav")

    def read(self):
        # Create a PyAudio object to read the audio stream
        input_stream = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sampling_rate,
            input=True,
            frames_per_buffer=self.record_buffersize,
            input_device_index=self.input_device,
        )

        def read_audio():
            bin_data = input_stream.read(
                self.record_buffersize, exception_on_overflow=False
            )
            data = torch.frombuffer(bin_data, dtype=torch.int16)
            return data

        # read the audio stream and put the batches into the input queue
        chunk_part_for_next = None
        chunk_part_for_next_times = None
        while True:
            (processing_time, processing_data), (
                chunk_part_for_next_times,
                chunk_part_for_next,
            ) = batchify_input_stream(
                read_callback=read_audio,
                out_batch_size=self.processing_size,
                input_shape=(self.record_buffersize,),
                sampling_rate=self.sampling_rate,
                chunk_part_for_next_times=chunk_part_for_next_times,
                chunk_part_for_next=chunk_part_for_next,
                dtype=torch.int16,
            )
            self.queues.input_queue.put((processing_time, processing_data))

    def write(self):
        # Create a PyAudio object to write the audio stream
        # TODO: check if frames_per_buffer is correct
        output_stream = pyaudio.PyAudio().open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sampling_rate,
            output=True,
            frames_per_buffer=self.output_buffersize,
            output_device_index=self.output_device,
        )
        if self.store:
            wav = get_wav_obj(self.store_path, self.sampling_rate)

        # clear the queue to avoid latency caused by init. delay
        clear_queue(self.queues.output_queue)

        # setup logging
        worker_configurer(self.log_queue, self.log_level)
        logger = logging.getLogger("audio_output")

        # write the audio stream from the output queue
        while True:
            tdata, data = self.queues.output_queue.get()
            data = data.to(torch.int16)
            bin_data = data.numpy().tobytes()
            delay = round(time.time() - tdata[0].item(), 2)
            logger.info(f"delay: {delay} s")
            output_stream.write(bin_data)

            if self.store:
                wav.writeframes(bin_data)


def get_device_idx(device_name: str, is_input: bool) -> int:
    """Retrieve the device index from the device name with sounddevice."""
    devices = sd.query_devices()
    for idx, device in enumerate(devices):
        if device["name"] == device_name:
            if is_input and device["max_input_channels"] > 0:
                return idx
            elif not is_input and device["max_output_channels"] > 0:
                return idx
    raise ValueError(f"Device {device_name} not found.")


def get_wav_obj(path: str, sample_rate: int) -> wave.Wave_write:
    """Return a wave.Wave_write object for the given path."""
    mode = "wb" if not os.path.isfile(path) else "ab"
    wav = wave.open(path, mode)
    if mode == "wb":
        wav.setnchannels(1)
        wav.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wav.setframerate(sample_rate)
    return wav
