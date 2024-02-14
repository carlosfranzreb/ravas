import time
import yaml
from argparse import ArgumentParser
import importlib
import logging
import os
import subprocess

from stream_processing.AudioVideoStreamer import AudioVideoStreamer


def get_cls(cls_str: str):
    """
    Import the module and return the class. `cls_str` should be in the format
    `module.class`.
    """
    module_str, cls_str = cls_str.rsplit(".", 1)
    module = importlib.import_module(module_str)
    return getattr(module, cls_str)


def main(config: dict) -> None:
    """
    - Create an logging directory, and store the config there.
    - Create a logging file in the logging directory.
    - Start the audio-video streamer with the given config.
    """

    # create a logging directory, store the config and create a logging file
    log_dir = os.path.join(config["log_dir"], str(int(time.time())))
    os.makedirs(log_dir, exist_ok=True)
    config["commit_hash"] = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )
    yaml.dump(config, open(os.path.join(log_dir, "config.yaml"), "w"))
    logging.basicConfig(
        filename=os.path.join(log_dir, "progress.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info(f"Logging to {log_dir}")

    # start the audio-video streamer
    cb_video = get_cls(config["video_callback"].pop("cls"))
    cv_audio = get_cls(config["audio_callback"].pop("cls"))
    audio_video_streamer = AudioVideoStreamer(
        video_callback=cb_video(**config["video_callback"]),
        video_processing_size=config["video"]["processing_size"],
        video_maximum_fps=config["video"]["maximum_fps"],
        audio_sampling_rate=config["audio"]["sampling_rate"],
        audio_processing_size=config["audio"]["processing_size"],
        audio_callback=cv_audio(**config["audio_callback"]),
        audio_pyaudio_input_device_index=config["audio"]["pyaudio_input_device_index"],
        use_video=config["video"]["use_video"],
        use_audio=config["audio"]["use_audio"],
        video_output_virtual_cam=config["video"]["output_virtual_cam"],
        video_output_window=config["video"]["output_window"],
        audio_pyaudio_output_device_index=config["audio"][
            "pyaudio_output_device_index"
        ],
    )

    audio_video_streamer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        audio_video_streamer.stop()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    main(config)
