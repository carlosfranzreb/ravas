import av
from streamlit_webrtc import webrtc_streamer

from streaming_vc import StreamingLLVC


if __name__ == "__main__":
    model = StreamingLLVC()

    def flip_frame(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        flipped = img[::-1, :, :]
        return av.VideoFrame.from_ndarray(flipped, format="bgr24")

    def convert(frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        converted = model(audio, frame.sample_rate)
        new_frame = av.AudioFrame.from_ndarray(converted, layout=frame.layout.name)
        new_frame.sample_rate = frame.sample_rate
        print(f"returned {new_frame.samples} samples")
        return new_frame

    webrtc_streamer(
        key="idk",
        # video_frame_callback=flip_frame,
        audio_frame_callback=convert,
        audio_receiver_size=1024,
        async_processing=True,
    )
