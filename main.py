import time
from stream_processing.AudioVideoStreamer import AudioVideoStreamer
from stream_processing.models.face_mask.FaceMask import FaceMask
from stream_processing.models.knnvc.converter import Converter

if __name__ == "__main__":
    audio_video_streamer = AudioVideoStreamer(
        video_callback=FaceMask(),
        video_processing_size=1,
        video_maximum_fps=20,
        audio_sampling_rate=16000,
        audio_processing_size=16000,
        audio_callback=Converter(
            target_feats_path="/Users/cafr02/datasets/LibriSpeech/dev-clean/84/121123",
            device="cpu",
        ),
        audio_pyaudio_input_device_index=3,
        use_video=True,
        use_audio=True,
        video_output_virtual_cam=False,
        video_output_window=True,
        audio_pyaudio_output_device_index=2,
    )

    audio_video_streamer.start()
    try:
        while True:
            time.sleep(1)
    except:
        audio_video_streamer.stop()
