import queue
import time
import logging
import os
import signal

import cv2
import torch
import pyvirtualcam
import cv2 as cv
from torch.multiprocessing import Queue

from .processor import ProcessingSyncState, Processor
from .utils import batchify_input_stream
from .dist_logging import worker_configurer


class VideoProcessor(Processor):
    def __init__(
        self,
        config: dict,
        video_sync_state: ProcessingSyncState,
        external_sync_state: ProcessingSyncState,
        log_queue: Queue,
        log_level: str,
    ):
        """
        Initialize a AudioProcessor object.
        :param name: Name of the processor.
        :param config: The config for the processor.
        :param video_sync_state: ProcessingSyncState containing the video sync state.
        :param external_sync_state: ProcessingSyncState containing the external sync state to sync the video.
        :param log_queue: log queue for logging messages.
        :param log_level: log level for logging messages.
        """
        super().__init__(
            "video", config, video_sync_state, external_sync_state, log_queue, log_level
        )
        self.video_sync_state = video_sync_state
        self.external_sync_state = external_sync_state
        if self.config["store"]:
            self.store_path = os.path.join(
                config["log_dir"], "video." + config["store_format"]
            )

    def read(self):
        # setup logging
        worker_configurer(self.log_queue, self.log_level)
        logger = logging.getLogger("video_input")

        # Create a VideoCapture object to read the video stream
        if self.config["video_file"]:
            video_reader = cv2.VideoCapture(self.config["video_file"])
        else:
            video_reader = cv2.VideoCapture(self.config["input_device"])

        frame_size = (
            int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)),
            3,
        )
        sampling_rate = self.get_sampling_rate()

        def read_video():
            """
            Return the next frame and it's timestamp. If the video comes from a stream,
            the timestamp is now. If the video comes from a file, the timestamp is the
            time of the frame in the video.
            """
            ret, frame = video_reader.read()
            if ret:
                if self.config["video_file"]:
                    ts = video_reader.get(cv2.CAP_PROP_POS_MSEC) / 1000
                else:
                    ts = time.time()
                return torch.from_numpy(frame[None]), ts
            else:
                if self.config["video_file"]:
                    # the video file is finished
                    return None, None
                return torch.empty((0, *frame_size)), time.time()

        # read the video stream and put the batches into the input queue
        chunk_part_for_next = torch.empty((0, *frame_size))
        chunk_part_for_next_times = torch.empty((0))
        last_frame_time = 0
        is_finished = False
        while True:
            (processing_time, processing_data), (
                chunk_part_for_next_times,
                chunk_part_for_next,
            ) = batchify_input_stream(
                read_callback=read_video,
                out_batch_size=self.config["processing_size"],
                input_shape=(1, *frame_size),
                sampling_rate=sampling_rate,
                chunk_part_for_next_times=chunk_part_for_next_times,
                chunk_part_for_next=chunk_part_for_next,
                dtype=torch.uint8,
                upper_bound_fps=(
                    None if self.config["video_file"] else self.config["max_fps"]
                ),
                last_frame_time=last_frame_time,
            )
            if processing_time is not None:
                logger.debug(f"Read from {processing_time[0]} s")
                last_frame_time = processing_time[-1].item()
            else:
                if is_finished:
                    break
                is_finished = True
            self.queues.input_queue.put((processing_time, processing_data))

    def write(self):

        # setup logging
        worker_configurer(self.log_queue, self.log_level)
        logger = logging.getLogger("video_output")

        # ensure fps of virtual cam is higher than the fps of the input stream
        sampling_rate = self.get_sampling_rate()
        if self.config["output_virtual_cam"]:
            virtual_cam = pyvirtualcam.Camera(
                width=self.config["width"],
                height=self.config["height"],
                fps=sampling_rate,
                device="/dev/video4",
            )
        if self.config["store"]:
            if self.config["store_format"] == "avi":
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
            elif self.config["store_format"] == "mp4":
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            else:
                raise ValueError(f"Unknown video format: {self.config['store_format']}")
            file_writer = cv2.VideoWriter(
                self.store_path,
                fourcc,
                sampling_rate,
                (self.config["width"], self.config["height"]),
            )
            signal.signal(signal.SIGTERM, lambda sig, frame: file_writer.release())

        # write the video stream from the output queue
        out = torch.empty((0, 0, 0, 0))
        while out is not None:
            try:
                ttime, out = self.queues.output_queue.get()
                if out is not None:
                    start_time = time.time()
                    # send each frame separately to the virtual cam
                    self.own_sync_state.last_sample_time.value = ttime[0].item()
                    self.own_sync_state.last_update.value = time.time()
                    for i, frame in enumerate(out):
                        frame = frame.numpy()
                        if self.config["store"]:
                            logger.debug(f"Writing, starting at {ttime[0]} s")
                            file_writer.write(frame)
                            continue

                        # sleep until the time of the frame is reached
                        current_ptime = time.time() - start_time
                        delta_time = ttime[i] - ttime[0]
                        sleep_time = delta_time - current_ptime
                        if sleep_time > 0:
                            time.sleep(sleep_time.item())

                        if self.config["output_window"]:
                            cv.imshow("frame", frame)
                            cv.waitKey(1)
                        if self.config["output_virtual_cam"]:
                            virtual_cam.send(frame[:, :, ::-1])
                        if i == 0 and self.config["video_file"] is None:
                            delay = round(time.time() - ttime[i].item(), 2)
                            logger.info(f"delay: {delay} s")

            except queue.Empty:
                pass

        if self.config["store"]:
            file_writer.release()
            logger.info("finish video")
        self.queues.finished.set()

    def get_sampling_rate(self) -> float:
        """Return the sampling rate of the input video."""
        if self.config["video_file"]:
            video_reader = cv2.VideoCapture(self.config["video_file"])
        else:
            video_reader = cv2.VideoCapture(self.config["input_device"])
        return video_reader.get(cv2.CAP_PROP_FPS)
