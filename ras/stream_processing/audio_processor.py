from queue import Queue, Empty
import logging
import time
import os
import subprocess

import pyaudio
import sounddevice as sd
import torch

import wave

from .processor import Processor, ProcessingSyncState
from .utils import batchify_input_stream, clear_queue
from .dist_logging import worker_configurer


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

        self.config = config
        if not self.config["video_file"]:
            self.input_device = get_device_idx(config["input_device"], True)
            self.output_device = get_device_idx(config["output_device"], False)

        if self.config["store"]:
            self.store_path = os.path.join(self.config["log_dir"], "audio.wav")

    def read(self):
        # setup logging
        worker_configurer(self.log_queue, self.log_level)
        logger = logging.getLogger("audio_input")

        logger.info('Using audio device indices AUDIO_INPUT=%s | AUDIO_OUTPUT=%s', self.input_device, self.output_device)

        # Create a PyAudio object to read the audio stream
        if self.config["video_file"]:
            # extract the audio from the video file with ffmpeg
            audio_path = os.path.join(self.config["log_dir"], "input_audio.wav")
            with open(os.path.join(self.config["log_dir"], "ffmpeg.log"), "a") as f:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i",
                        self.config["video_file"],
                        "-vn",
                        "-acodec",
                        "pcm_s16le",
                        "-ar",
                        str(self.config["sampling_rate"]),
                        "-ac",
                        "1",
                        "-f",
                        "wav",
                        audio_path,
                    ],
                    stdout=f,
                    stderr=f,
                )
            audio_reader = wave.open(audio_path, "rb")
        else:
            audio_reader = pyaudio.PyAudio().open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config["sampling_rate"],
                input=True,
                frames_per_buffer=self.config["record_buffersize"],
                input_device_index=self.input_device,
            )

        def read_audio():
            if isinstance(audio_reader, wave.Wave_read):
                bin_data = audio_reader.readframes(self.config["record_buffersize"])
                ts = audio_reader.tell() / self.config["sampling_rate"]
            else:
                bin_data = audio_reader.read(
                    self.config["record_buffersize"], exception_on_overflow=False
                )
                ts = time.time()

            try:
                data = torch.frombuffer(bin_data, dtype=torch.int16)
                return data, ts
            except ValueError as err:
                # if the audio stream is finished, return None
                if isinstance(audio_reader, wave.Wave_read):
                    return None, None
                # otherwise, this error is unexpected
                else:
                    raise ValueError(f"Error reading audio: {err}")

        # read the audio stream and put the batches into the input queue
        chunk_part_for_next = torch.empty((0, self.config["record_buffersize"]))
        chunk_part_for_next_times = torch.empty((0))
        is_finished = False

        self.queues.ready.wait()
        if self.external_sync_state.ready:
            logger.info('audio converter is ready, waiting for video converter to be ready...')
            self.external_sync_state.ready.wait()
            logger.info('audio and video converter are ready, start reading audio')
        else:
            logger.info('audio converter is ready, starting to process input...')

        while True:
            (processing_time, processing_data), (
                chunk_part_for_next_times,
                chunk_part_for_next,
            ) = batchify_input_stream(
                read_callback=read_audio,
                out_batch_size=self.config["processing_size"],
                input_shape=(self.config["record_buffersize"],),
                sampling_rate=self.config["sampling_rate"],
                chunk_part_for_next_times=chunk_part_for_next_times,
                chunk_part_for_next=chunk_part_for_next,
                dtype=torch.int16,
            )
            if processing_time is not None:
                logger.debug(f"Read from {processing_time[0]} s")
            else:
                if is_finished:
                    break
                is_finished = True
            self.queues.input_queue.put((processing_time, processing_data))

    def write(self):
        # setup logging
        worker_configurer(self.log_queue, self.log_level)
        logger = logging.getLogger("audio_output")

        # Create a PyAudio object to write the audio stream
        # TODO: check if frames_per_buffer is correct
        if self.config["store"]:
            wav = get_wav_obj(self.store_path, self.config["sampling_rate"])
        else:
            output_stream = pyaudio.PyAudio().open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.config["sampling_rate"],
                output=True,
                frames_per_buffer=self.config["output_buffersize"],
                output_device_index=self.output_device,
            )

        # clear the queue to avoid latency caused by init. delay
        clear_queue(self.queues.output_queue)
        # write the audio stream from the output queue
        data = torch.empty((0))
        while data is not None:
            try:
                tdata, data = self.queues.output_queue.get()
                if data is not None:
                    data = data.to(torch.int16)
                    bin_data = data.numpy().tobytes()
                    if self.config["store"]:
                        logger.debug(f"Writing, starting at {tdata[0]} s")
                        wav.writeframes(bin_data)
                    else:
                        delay = round(time.time() - tdata[0].item(), 2)
                        logger.info(f"delay: {delay} s")
                        output_stream.write(bin_data)

            except Empty:
                pass
            except EOFError:
                break

        logger.info("finish audio")
        if self.config["store"]:
            wav.close()
        else:
            output_stream.stop_stream()
            output_stream.close()
        self.queues.finished.set()


def get_device_idx(device_name: str | int, is_input: bool) -> int:
    """Retrieve the device index from the device name with sounddevice."""
    devices = sd.query_devices()
    try:
        device_idx = int(device_name)
        if device_idx < len(devices):
            return device_idx
    except ValueError:
        pass
    for idx, device in enumerate(devices):
        if device["name"] == device_name:
            if is_input and device["max_input_channels"] > 0:
                return idx
            elif not is_input and device["max_output_channels"] > 0:
                return idx
    raise ValueError(f"Device {device_name} not found. Device list ({len(devices)}): {devices}")


def get_wav_obj(path: str, sample_rate: int) -> wave.Wave_write:
    """Return a wave.Wave_write object for the given path."""
    mode = "wb" if not os.path.isfile(path) else "ab"
    wav = wave.open(path, mode)
    if mode == "wb":
        wav.setnchannels(1)
        wav.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wav.setframerate(sample_rate)
    return wav
