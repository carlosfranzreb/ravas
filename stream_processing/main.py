import multiprocessing
import os
import subprocess
import time
from argparse import ArgumentParser

import yaml
import torch
import torchaudio

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
    config["audio"]["log_dir"] = log_dir
    config["video"]["log_dir"] = log_dir

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
            out_audio = os.path.join(log_dir, "audio.wav")
            out_video = os.path.join(log_dir, "video.mp4")
            if config["audio"]["video_file"] is None:
                while True:
                    time.sleep(1)
            elif config["audio"]["store"] and config["video"]["store"]:
                while file_written(out_audio) or file_written(out_video):
                    time.sleep(1)
            elif config["audio"]["store"]:
                while file_written(out_audio):
                    time.sleep(1)
            elif config["video"]["store"]:
                while file_written(out_video):
                    time.sleep(1)
            else:
                raise ValueError("A file is being anonymized but not stored.")
    finally:
        audio_video_streamer.stop()
        log_listener.terminate()
        if config["audio"]["store"] and config["video"]["store"]:
            merge_audio_video(log_dir)


def file_written(file_path: str, t_since_write: int = 2) -> bool:
    """
    Return true if the file has been written in the last `t_since_write` seconds.
    If the file does not exist, return True.
    """
    if not os.path.exists(file_path):
        return True
    return time.time() - os.path.getmtime(file_path) < t_since_write


def merge_audio_video(log_dir: str) -> None:
    """
    Merge the audio and video files into a single file.

    Args:
    - log_dir: The directory where the audio and video files are stored.
    """
    # assert that the input and output audio have the same length
    audio_file = os.path.join(log_dir, "audio.wav")
    audio_orig_file = os.path.join(log_dir, "input_audio.wav")
    audio, sr = torchaudio.load(audio_file)
    audio_orig, sr_orig = torchaudio.load(audio_orig_file)
    assert sr == sr_orig, "Sample rates of the audio files are different."
    if audio.shape[1] < audio_orig.shape[1]:
        audio = torch.cat([audio, torch.zeros_like(audio_orig[:, audio.shape[1] :])], 1)
    elif audio.shape[1] > audio_orig.shape[1]:
        audio = audio[:, : audio_orig.shape[1]]
    torchaudio.save(audio_file, audio, sr)

    # merge the audio and video files
    video_file = os.path.join(log_dir, "video.mp4")
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
    main(config, runtime=None)
