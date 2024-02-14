import queue
import time
from typing import Optional
import logging

import cv2
import torch
from stream_processing.Processor import (
    ProcessingCallback,
    ProcessingQueues,
    ProcessingSyncState,
    Processor,
)
import pyvirtualcam
from stream_processing.utils import batchify_input_stream
import cv2 as cv


class VideoProcessor(Processor):
    def __init__(
        self,
        video_queues: ProcessingQueues,
        video_sync_state: ProcessingSyncState,
        external_sync_state: ProcessingSyncState,
        callback: Optional[ProcessingCallback] = None,
        maximum_fps=30,
        processing_size=10,
        opencv_input_device_index=0,
        max_unsynced_time: Optional[float] = 0.01,
        output_virtual_cam: bool = True,
        output_window: bool = False,
    ):
        """
        Initialize a AudioProcessor object.
        :param video_queues: ProcessingQueues containing the video queues.
        :param video_sync_state: ProcessingSyncState containing the video sync state.
        :param external_sync_state: ProcessingSyncState containing the external sync state to sync the video.
        :param callback: Callback Object that is used for initializing the callback function and the callback function.
        :param maximum_fps: Maximum fps of the video stream.
        :param processing_size: Size of the processing batch.
        :param opencv_input_device_index: Index of the opencv input device.
        :param max_unsynced_time: Maximum time that the data can be unsynced.
        :param output_virtual_cam: Activate output to virtual cam.
        :param output_window: Activate output to window.

        """
        super().__init__(
            video_queues,
            video_sync_state,
            external_sync_state,
            callback,
            max_unsynced_time,
        )
        self.video_queues = video_queues
        self.video_sync_state = video_sync_state
        self.external_sync_state = external_sync_state

        self.maximum_fps = maximum_fps
        self.processing_size = processing_size
        self.opencv_input_device_index = opencv_input_device_index

        self.output_virtual_cam = output_virtual_cam
        self.output_window = output_window

    def read_input_stream(self):
        # Create a VideoCapture object to read the video stream
        video_capture = cv2.VideoCapture(self.opencv_input_device_index)
        frame_size = (
            int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            3,
        )

        def read_video():
            ret, frame = video_capture.read()
            if ret:
                return torch.from_numpy(frame[None])
            else:
                raise Exception("Video capture failed")

        # read the video stream and put the batches into the input queue
        chunk_part_for_next = torch.empty((0, *frame_size))
        chunk_part_for_next_times = torch.empty((0))
        last_frame_time = 0
        while True:
            (processing_time, processing_data), (
                chunk_part_for_next_times,
                chunk_part_for_next,
            ) = batchify_input_stream(
                read_callback=read_video,
                out_batch_size=self.processing_size,
                input_shape=(1, *frame_size),
                sampling_rate=self.maximum_fps,
                chunk_part_for_next_times=chunk_part_for_next_times,
                chunk_part_for_next=chunk_part_for_next,
                dtype=torch.uint8,
                upper_bound_fps=self.maximum_fps,
                last_frame_time=last_frame_time,
            )
            last_frame_time = processing_time[-1].item()

            self.video_queues.input_queue.put((processing_time, processing_data))

    def write_output_stream(self):
        # ensure fps of virtual cam is higher than the fps of the input stream
        if self.output_virtual_cam:
            virtual_cam = pyvirtualcam.Camera(
                width=1280, height=720, fps=100, device="/dev/video4"
            )
        while True:
            try:
                ttime, out = self.video_queues.output_queue.get()

                start_time = time.time()
                # send each frame separately to the virtual cam
                self.own_sync_state.last_sample_time.value = ttime[0].item()
                self.own_sync_state.last_update.value = time.time()
                for i, frame in enumerate(out):
                    # sleep until the time of the frame is reached
                    current_ptime = time.time() - start_time
                    delta_time = ttime[i] - ttime[0]
                    sleep_time = delta_time - current_ptime
                    if sleep_time > 0:
                        time.sleep(sleep_time.item())
                    if self.output_window:
                        cv.imshow("frame", frame.numpy())
                        cv.waitKey(1)
                    if self.output_virtual_cam:
                        virtual_cam.send(frame.numpy()[:, :, ::-1])
                    if i == 0:
                        logging.info(
                            "video output delay: ", time.time() - ttime[i].item()
                        )
                    # sleep until the next frame should be sent (1/fps)
                    # virual_cam.sleep_until_next_frame()
            except queue.Empty:
                pass
