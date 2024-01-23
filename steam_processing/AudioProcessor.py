import time
from typing import Any, Callable, Optional, Tuple
import pyaudio
import torch
from steam_processing.Processor import (
    ProcessingQueues,
    ProcessingSyncState,
    Processor,
)
from steam_processing.utils import batchify_input_stream, clear_queue


class AudioProcessor(Processor):
    def __init__(
        self,
        audio_queues: ProcessingQueues,
        audio_sync_state: ProcessingSyncState,
        external_sync_state: ProcessingSyncState,
        callback: Optional[
            Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        init_callback: Optional[Callable[[], Any]] = None,
        processing_size=1024,
        pyaudio_input_device_index=0,
        sampling_rate=44100,
        record_buffersize=1024,
        pyaudio_output_device_index=1,
        output_buffersize=1024,
        max_unsynced_time: Optional[float] = 0.1,
    ):
        """
        Initialize a AudioProcessor object.
        :param audio_queues: ProcessingQueues containing the audio queues.
        :param audio_sync_state: ProcessingSyncState containing the audio sync state.
        :param external_sync_state: ProcessingSyncState containing the external sync state to sync the audio.
        :param callback: Callback function that is called for processing the data.
            the callback function gets the batched input time and data per sample and should return the batched time and data.
        :param init_callback: Callback function that is called for initializing the callback function.
            the function should return a list of arguments that are passed to the callback function.
        :param processing_size: Size of the processing batch.
        :param pyaudio_input_device_index: Index of the pyaudio input device.
        :param sampling_rate: Sampling rate for the recording.
        :param record_buffersize: Size of the system recording buffer.
        :param pyaudio_output_device_index: Index of the pyaudio output device.
        :param output_buffersize: Size of the system output buffer.
        :param max_unsynced_time: Maximum time that the data can be unsynced.

        """
        super().__init__(
            audio_queues,
            audio_sync_state,
            external_sync_state,
            callback,
            init_callback,
            max_unsynced_time,
        )
        self.audio_queues = audio_queues
        self.audio_sync_state = audio_sync_state
        self.external_sync_state = external_sync_state
        self.callback = callback

        self.sampling_rate = sampling_rate
        self.record_buffersize = record_buffersize
        self.pyaudio_input_device_index = pyaudio_input_device_index
        self.processing_size = processing_size

        self.pyaudio_output_device_index = pyaudio_output_device_index
        self.output_buffersize = output_buffersize

    def read_input_stream(self):
        # Create a PyAudio object to read the audio stream
        p = pyaudio.PyAudio()
        input_stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sampling_rate,
            input=True,
            frames_per_buffer=self.record_buffersize,
            input_device_index=self.pyaudio_input_device_index,
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
            self.audio_queues.input_queue.put((processing_time, processing_data))

    def write_output_stream(self):
        # Create a PyAudio object to write the audio stream
        p = pyaudio.PyAudio()
        # TODO: check if frames_per_buffer is correct
        output_stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sampling_rate,
            output=True,
            frames_per_buffer=self.output_buffersize,
            output_device_index=self.pyaudio_output_device_index,
        )
        # clear the queue to avoid latency caused by an delay in the initialization of this process
        clear_queue(self.audio_queues.output_queue)

        while True:
            tdata, data = self.audio_queues.output_queue.get()

            data = data.to(torch.int16)
            bin_data = data.numpy().tobytes()
            print("audio output delay: ", time.time() - tdata[0].item())
            output_stream.write(bin_data)
