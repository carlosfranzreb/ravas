import logging
import os
import subprocess
import time
from argparse import ArgumentParser

import yaml
from torch import multiprocessing

from .dist_logging import listener_process
from .dist_logging import worker_configurer
from .streamer import AudioVideoStreamer


def main(config: dict, runtime: int = None) -> None:
    """
    - Create an logging directory, and store the config there.
    - Create a logging file in the logging directory.
    - Start the audio-video streamer with the given config.

    Args:
    - config: The config for the demonstrator.
    - runtime: Stop the audio-video streamer after `runtime` seconds.
    """

    # check if the config is valid
    proc_size = config["audio"]["processing_size"]
    buffer_size = config["audio"]["record_buffersize"]
    assert proc_size > buffer_size, "Proc. size should be greater than buffer size"

    # create a logging directory and store the config
    log_dir = os.path.join(config["log_dir"], str(int(time.time())))
    os.makedirs(log_dir, exist_ok=True)
    yaml.dump(config, open(os.path.join(log_dir, "config.yaml"), "w"))
    config["audio"]["log_dir"] = log_dir
    config["video"]["log_dir"] = log_dir

    # start the logging
    log_queue = multiprocessing.Queue(-1)
    log_listener = multiprocessing.Process(
        target=listener_process, args=(log_dir, log_queue, config["log_level"])
    )
    log_listener.start()

    worker_configurer(log_queue, config["log_level"])
    logger = logging.getLogger("main")
    start_time = time.perf_counter_ns()  # FIXME perf

    # start the audio-video streamer with the given config
    audio_video_streamer = AudioVideoStreamer(config, log_queue)
    audio_video_streamer.start()

    if config["video"]["video_file"] or config["audio"]["video_file"]:
        # wait for the audio-video streamer to finish with the input video file
        audio_video_streamer.wait()
    else:
        # stop the streamer after `runtime` seconds or wait indefinitely until the user
        # interrupts the program
        if runtime is not None:
            time.sleep(runtime)
            audio_video_streamer.stop()
        else:
            while True:
                time.sleep(1)

    duration = time.perf_counter_ns() - start_time  # FIXME perf
    msg = 'Total Running Time / Duration (ms): %s' % ((duration / 1000000),)
    logger.info(msg)

    # stop the audio-video streamer and the logging
    audio_video_streamer.stop()
    log_listener.terminate()
    if config["audio"]["store"] and config["video"]["store"]:
        merge_audio_video(log_dir)


def merge_audio_video(log_dir: str) -> None:
    """
    Merge the audio and video files into a single file.

    Args:
    - log_dir: The directory where the audio and video files are stored.
    """
    audio_file = os.path.join(log_dir, "audio.wav")
    video_file = os.path.join(log_dir, "video." + config["video"]["store_format"])
    output_file = os.path.join(log_dir, "merged.mp4")
    with open(os.path.join(log_dir, "ffmpeg.log"), "a") as f:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                audio_file,
                "-i",
                video_file,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-strict",
                "experimental",
                output_file,
            ],
            stdout=f,
            stderr=f,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/onnx_models.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    main(config, runtime=100)
