from torch.multiprocessing import Queue

from .audio_processor import AudioProcessor
from .processor import ProcessingSyncState, ProcessorHandler
from .video_processor import VideoProcessor


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
            if self.use_video and not (config["audio"]["store"] or config["video"]["store"]):
                # only need synced ready-signal with video, if video is enabled, and if both audio & video are NOT
                # set to be stored in a file
                # (NOTE audio is either written to file or to output-stream; not both)
                audio_sync_state.ready = audio_processor.queues.ready
            else:
                # if video is disabled (and neither video nor audio are stored):
                # indicate that sync-ing for video is disabled
                video_sync_state.disabled.value = True
            self.audio_handler = ProcessorHandler(audio_processor)
        if self.use_video:
            video_processor = VideoProcessor(
                config["video"],
                video_sync_state,
                audio_sync_state,
                log_queue=log_queue,
                log_level=log_level,
            )
            if self.use_audio and not (config["video"]["store"] or config["audio"]["store"]):
                # only need synced ready-signal with audio, if audio is enabled, and if both video & audio are NOT
                # set to be stored in a file
                video_sync_state.ready = video_processor.queues.ready
            else:
                # if audio is disabled (and neither audio nor video are stored):
                # indicate that sync-ing for audio is disabled
                audio_sync_state.disabled.value = True
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
