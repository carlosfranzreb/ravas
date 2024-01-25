import time

import cv2
import torch
from steam_processing.AudioVideoStreamer import AudioVideoStreamer
import librosa
import numpy as np


# all callbacks have to be defined outside of the main function
# use the init_callback to load the model and return it as a list to use it as an argument for the callback function
def video_init_callback():
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    return [face_cascade]


# the callback function gets the batched input time and data and should return the modified batched time and data.
# the arguments after the first two arguments are the arguments returned by the init_callback function
def video_callback(time, data, face_cascade):
    # detect faces on the first sample
    sample = data[0].numpy()
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # draw a rectangle around the face position of the first sample on all samples
    if len(faces) > 0:
        x, y, w, h = faces[0]
        data[:, y : y + h, x : x + w, :] = data[:, y : y + h, x : x + w, :] * 0.5

    return time, data


# same as the video callback
def audio_init_callback():
    # call once before to load function and prevent delay
    def call(y):
        return librosa.effects.pitch_shift(y, sr=44100, n_steps=-5)

    call(np.zeros(44100))
    return [call]


def audio_callback(dtime, data, pitch_shift):
    # uint8 to float32
    data = data.float() / 255
    # convert to numpy and apply pitch shift
    y = pitch_shift(data.numpy())

    # from numpy float32 to torch uint8
    data = torch.from_numpy(y) * 255
    return dtime, data.to(torch.uint8)


if __name__ == "__main__":
    # start the streamer
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
