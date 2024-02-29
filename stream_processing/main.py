import multiprocessing
import os
import subprocess
import time
from argparse import ArgumentParser

import yaml

from stream_processing.streamer import AudioVideoStreamer
from stream_processing.dist_logging import listener_process


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
    config["commit_hash"] = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )
    yaml.dump(config, open(os.path.join(log_dir, "config.yaml"), "w"))

    # start the logging
    log_queue = multiprocessing.Queue(-1)
    log_listener = multiprocessing.Process(
        target=listener_process, args=(log_dir, log_queue, config["log_level"])
    )
    log_listener.start()

    # start the audio-video streamer with the given config
    audio_video_streamer = AudioVideoStreamer(config, log_queue)
    audio_video_streamer.start()

    # stop the streamer after `runtime` seconds or wait indefinitely until the user
    # interrupts the program
    try:
        if runtime is not None:
            time.sleep(runtime)
            audio_video_streamer.stop()
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        audio_video_streamer.stop()
    log_listener.terminate()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    main(config, runtime=180)
