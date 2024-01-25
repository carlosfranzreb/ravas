from stream_processing.AudioProcessor import AudioProcessor
from stream_processing.Processor import (
    ProcessingQueues,
    ProcessingSyncState,
    ProcessorProcessHandler,
)
from stream_processing.VideoProcessor import VideoProcessor


class AudioVideoStreamer:
    def __init__(
        self,
        audio_callback=None,
        audio_init_callback=None,
        audio_processing_size=1024,
        audio_pyaudio_input_device_index=0,
        audio_sampling_rate=44100,
        audio_record_buffersize=1024,
        audio_pyaudio_output_device_index=1,
        audio_output_buffersize=1024,
        video_callback=None,
        video_init_callback=None,
        video_processing_size=10,
        video_opencv_input_device_index=0,
        video_maximum_fps=30,
        max_unsynced_time=0.1,
        use_audio=True,
        use_video=True,
    ):
        """
        Initialize a AudioVideoStreamer object.

        :param audio_callback: Callback function that is called for processing the audio data.
            the callback function gets the batched input time and data per sample and should return the batched time and data.
        :param init_callback: Callback function that is called for initializing the callback function.
            the function should return a list of arguments that are passed to the callback function.
        :param audio_processing_size: Size of the processing batch.
        :param audio_pyaudio_input_device_index: Index of the pyaudio input device.
        :param audio_sampling_rate: Sampling rate for the recording.
        :param audio_record_buffersize: Size of the system recording buffer.
        :param audio_pyaudio_output_device_index: Index of the pyaudio output device.
        :param audio_output_buffersize: Size of the system output buffer.
        :param video_callback: Callback function that is called for processing the video data.
            the callback function gets the batched input time and data per sample and should return the batched time and data.
        :param video_processing_size: Size of the processing batch.
        :param video_opencv_input_device_index: Index of the opencv input device.
        :param video_maximum_fps: Maximum fps of the video stream.
        :param max_unsynced_time: Maximum time that the data can be unsynced.
        :param use_audio: Use the audio processor.
        :param use_video: Use the video processor.
        """
        audio_processing_sync_state = ProcessingSyncState()
        video_processing_sync_state = ProcessingSyncState()
        self.use_audio = use_audio
        self.use_video = use_video
        if use_audio:
            audio_processing_queues = ProcessingQueues()
            audio_processor = AudioProcessor(
                audio_processing_queues,
                audio_processing_sync_state,
                video_processing_sync_state,
                callback=audio_callback,
                init_callback=audio_init_callback,
                processing_size=audio_processing_size,
                pyaudio_input_device_index=audio_pyaudio_input_device_index,
                sampling_rate=audio_sampling_rate,
                record_buffersize=audio_record_buffersize,
                pyaudio_output_device_index=audio_pyaudio_output_device_index,
                output_buffersize=audio_output_buffersize,
                max_unsynced_time=max_unsynced_time,
            )
            self.audio_processor_process_handler = ProcessorProcessHandler(
                audio_processor
            )
        if use_video:
            video_processing_queues = ProcessingQueues()
            video_processor = VideoProcessor(
                video_processing_queues,
                video_processing_sync_state,
                audio_processing_sync_state,
                callback=video_callback,
                init_callback=video_init_callback,
                processing_size=video_processing_size,
                opencv_input_device_index=video_opencv_input_device_index,
                maximum_fps=video_maximum_fps,
                max_unsynced_time=max_unsynced_time,
            )
            self.video_processor_process_handler = ProcessorProcessHandler(
                video_processor
            )

    def start(self):
        """
        Start the audio and video processor.
        """
        if self.use_audio:
            self.audio_processor_process_handler.start()
        if self.use_video:
            self.video_processor_process_handler.start()

    def stop(self):
        """
        Stop the audio and video processor.
        """
        if self.use_audio:
            self.audio_processor_process_handler.stop()
        if self.use_video:
            self.video_processor_process_handler.stop()
