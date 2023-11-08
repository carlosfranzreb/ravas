import av
from streamlit_webrtc import webrtc_streamer

from streaming_stargan import StreamingSG


if __name__ == "__main__":
    stargan = StreamingSG(target_id=0)

    def flip_frame(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        flipped = img[::-1, :, :]
        return av.VideoFrame.from_ndarray(flipped, format="bgr24")

    def convert(frame: av.AudioFrame) -> av.AudioFrame:
        return av.AudioFrame.from_ndarray(stargan(frame.to_ndarray()), layout="mono")

    if __name__ == "__main__":
        webrtc_streamer(
            key="idk",
            video_frame_callback=flip_frame,
            audio_frame_callback=convert,
        )
