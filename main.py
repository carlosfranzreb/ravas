import logging
import time

import cv2
import torch
from steam_processing.AudioVideoStreamer import AudioVideoStreamer
import librosa
import numpy as np


def video_init_callback():
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return [face_cascade]


def video_callback(time, data, face_cascade):
    # detect faces
    sample = data[0].numpy()
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # draw bounding boxes
    if len(faces) > 0:
        x, y, w, h = faces[0]
        data[:, y : y + h, x : x + w, :] = data[:, y : y + h, x : x + w, :] * 0.5

    return time, data


def audio_init_callback():
    # call once before to load function
    def call(y):
        return librosa.effects.pitch_shift(y, sr=44100, n_steps=-5)

    call(np.zeros(44100))
    return [call]


def audio_callback(dtime, data, pitch_shift):
    # uint8 to float32
    data = data.float() / 255
    y = pitch_shift(data.numpy())
    data = torch.from_numpy(y) * 255
    return dtime, data


if __name__ == "__main__":
    audio_video_streamer = AudioVideoStreamer(
        video_callback=video_callback,
        video_init_callback=video_init_callback,
        video_processing_size=4,
        video_maximum_fps=20,
        audio_processing_size=4096 * 8,
        audio_callback=audio_callback,
        audio_init_callback=audio_init_callback,
        use_video=True,
    )
    audio_video_streamer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        audio_video_streamer.stop()
