from torch.multiprocessing import Queue

from stream_processing.audio_processor import AudioProcessor
from stream_processing.processor import ProcessingSyncState, ProcessorHandler
from stream_processing.video_processor import VideoProcessor


class AudioVideoStreamer:
    def __init__(self, config: dict, log_queue: Queue):
        """
        Initialize a AudioVideoStreamer object.

        :param config: The config for the demonstrator.
        :param log_queue: log queue for logging messages.
        """
        audio_sync_state = ProcessingSyncState()
        video_sync_state = ProcessingSyncState()
        self.use_audio = config["audio"]["use_audio"]
        self.use_video = config["video"]["use_video"]
        log_level = config["log_level"]

        if self.use_audio:
            audio_processor = AudioProcessor(
                config["audio"],
                audio_sync_state,
                video_sync_state,
                log_queue=log_queue,
                log_level=log_level,
            )
            self.audio_handler = ProcessorHandler(audio_processor)
        if self.use_video:
            video_processor = VideoProcessor(
                config["video"],
                video_sync_state,
                audio_sync_state,
                log_queue=log_queue,
                log_level=log_level,
            )
            self.video_handler = ProcessorHandler(video_processor)

    def start(self):
        """
        Start the audio and video processor.
        """
        if self.use_audio:
            self.audio_handler.start()
        if self.use_video:
            self.video_handler.start()

    def stop(self):
        """
        Stop the audio and video processor.
        """
        if self.use_audio:
            self.audio_handler.stop()
        if self.use_video:
            self.video_handler.stop()

    def wait(self):
        """
        Wait for the audio and video processor to finish.
        """
        if self.use_audio:
            self.audio_handler.wait()
        if self.use_video:
            self.video_handler.wait()
