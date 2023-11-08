from streamlit_webrtc import webrtc_streamer
import av


def flip_frame(frame):
    img = frame.to_ndarray(format="bgr24")
    flipped = img[::-1, :, :]
    return av.VideoFrame.from_ndarray(flipped, format="bgr24")


def increase_pitch(frame):
    


if __name__ == "__main__":
    webrtc_streamer(
        key="sample",
        video_frame_callback=flip_frame,
        audio_frame_callback=increase_pitch,
    )
