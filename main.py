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
        converted = model(frame.to_ndarray(), frame.sample_rate)
        converted_scaled = (converted * 32767).round().astype("int16")
        return av.AudioFrame.from_ndarray(converted_scaled, layout="mono")

    if __name__ == "__main__":
        webrtc_streamer(
            key="idk",
            video_frame_callback=flip_frame,
            audio_frame_callback=convert,
        )
