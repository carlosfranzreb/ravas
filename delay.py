from time import sleep
import av
from streamlit_webrtc import webrtc_streamer
import pydub
import numpy as np


gain = 10


async def flip_frame(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    flipped = img[::-1, :, :]
    return av.VideoFrame.from_ndarray(flipped, format="bgr24")


async def flip_frames_callback(
    frames: list[av.VideoFrame],
) -> list[av.VideoFrame]:
    flipped_frames = [await flip_frame(frame) for frame in frames]
    return flipped_frames


async def change_gain(frame: av.AudioFrame) -> av.AudioFrame:
    raw_samples = frame.to_ndarray()
    sound = pydub.AudioSegment(
        data=raw_samples.tobytes(),
        sample_width=frame.format.bytes,
        frame_rate=frame.sample_rate,
        channels=len(frame.layout.channels),
    )
    sound = sound.apply_gain(gain)
    channel_sounds = sound.split_to_mono()
    channel_samples = [s.get_array_of_samples() for s in channel_sounds]
    new_samples: np.ndarray = np.array(channel_samples).T
    new_samples = new_samples.reshape(raw_samples.shape)

    new_frame = av.AudioFrame.from_ndarray(new_samples, layout=frame.layout.name)
    new_frame.sample_rate = frame.sample_rate
    return new_frame


async def change_gain_callback(
    frames: list[av.AudioFrame],
) -> list[av.AudioFrame]:
    processed_frames = [await change_gain(frame) for frame in frames]
    return processed_frames


webrtc_streamer(
    key="idk",
    queued_video_frames_callback=flip_frames_callback,
    queued_audio_frames_callback=change_gain_callback,
    async_processing=True,
)
