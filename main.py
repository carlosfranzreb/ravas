import time
from stream_processing.AudioVideoStreamer import AudioVideoStreamer
from models.face_mask.FaceMask import FaceMask
from models.knn_vc.KNNVC import KNNVC

if __name__ == "__main__":
    audio_video_streamer = AudioVideoStreamer(
        video_callback=FaceMask(),
        video_processing_size=1,
        video_maximum_fps=20,
        audio_sampling_rate=16000,
        audio_processing_size=16000,
        audio_callback=KNNVC(ref_dir="models/knn_vc/LibriSpeechSamples"),
        audio_pyaudio_input_device_index=0,
        use_video=True,
        use_audio=True,
        video_output_virtual_cam=False,
        video_output_window=True,
        audio_pyaudio_output_device_index=1,
    )

    audio_video_streamer.start()
    try:
        while True:
            time.sleep(1)
    except:
        audio_video_streamer.stop()
