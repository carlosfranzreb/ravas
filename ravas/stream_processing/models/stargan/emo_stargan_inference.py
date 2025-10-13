from emo_stargan_model import Model
import librosa
import numpy as np
import noisereduce as nr
import streamlit as st

MODEL_SR = 24000

f0_path = "emo-stargan/Utils/JDC/bst.t7"
speaker_file = "emo-stargan/Data/train_list.txt"
vocoder_config = "emo-stargan/Utils/vocoder/config.yml"


# define paths of models not in git repo
vocoder_path = "models/vocoder/checkpoint-1790000steps.pkl"
stargan_path = "models/starganv2/epoch_00064.pth"
stargan_config = "models/starganv2/config.yml"

# from FrameProcessor import FramesProcessor
pseudo_names = {
    0: "None",
    11: "Joe",
    12: "Mark",
    13: "Tony",
    14: "Tom",
    15: "Anna",
    16: "Jane",
    17: "Mary",
    18: "Julia",
    19: "Lizy",
    20: "Harry",
}
pseudo_nr = {v: k for k, v in pseudo_names.items()}


class EmoStarganProcessor:
    def __init__(self, device="cuda"):
        speaker_name = st.radio(
            "Speaker",
            options=list(pseudo_names.values()),
            index=6,
        )
        self.leading_zeros_length = st.slider("leading_zeros [s]", 0.0, 10.0, 0.0, 0.25)
        self.leading_zeros_length = int(self.leading_zeros_length * 24000)
        self.pre_audio_length = st.slider("pre_audio [s]", 0.0, 10.0, 0.25, 0.25)
        self.pre_audio_length = int(self.pre_audio_length * 24000)
        self.st_image = st.image([])
        self.st_image_current = st.image([])

        @st.cache_resource()
        def set_model():
            return Model(
                f0_path,
                vocoder_path,
                vocoder_config,
                stargan_path,
                stargan_config,
                speaker_file,
                speaker_nr=(
                    pseudo_nr[speaker_name]
                    if speaker_name != "None"
                    else pseudo_nr["Anna"]
                ),
                device=device,
            )

        self.model = set_model()

        @st.cache_resource
        def set_speaker(speaker_name):
            if speaker_name != "None":
                self.model.set_speaker(pseudo_nr[speaker_name])

        set_speaker(speaker_name)

    def set_image(self, audio, st_img, caption):
        librosa_spec = librosa.feature.melspectrogram(
            y=audio, sr=MODEL_SR, n_fft=2048, hop_length=300, n_mels=80
        )
        librosa_spec = librosa.power_to_db(librosa_spec, ref=np.max)
        librosa_spec = np.flip(librosa_spec, axis=0)
        librosa_spec = librosa_spec - np.min(librosa_spec)
        librosa_spec = librosa_spec / np.max(librosa_spec)
        st_img.image(librosa_spec, caption=caption, width=400)

    def process(self, audio, history):
        leading_zeros_num = int(self.leading_zeros_length)
        pre_audio_num = int(self.pre_audio_length)
        current_audio_length = audio.shape[0]
        current_audio_num = int(current_audio_length)

        if history.shape[0] + audio.shape[0] < pre_audio_num + current_audio_num:
            print("not enough audio")
            return np.zeros_like(audio)

        hist_audio_num = (
            history.shape[0] + audio.shape[0] - pre_audio_num - current_audio_num
        )

        leading_zeros = np.zeros(leading_zeros_num)
        all_audio = np.concatenate([history, audio], axis=0)
        hist_audio_start_idx = 0
        current_audio_start_idx = hist_audio_start_idx + hist_audio_num
        pre_audio_start_idx = current_audio_start_idx + current_audio_num

        audio_gen = all_audio[
            hist_audio_start_idx : pre_audio_start_idx + pre_audio_num
        ]

        audio_in = np.concatenate([audio_gen, leading_zeros], axis=0)

        c_in = audio_in[hist_audio_num : hist_audio_num + current_audio_num]

        self.set_image(c_in, self.st_image_current, "current")

        self.set_image(audio_in, self.st_image, "all")

        # plot librosa spec

        silence = (
            np.max(
                np.abs(audio_in[hist_audio_num : hist_audio_num + current_audio_num])
            )
            < 0.1
        )

        # audio_in = nr.reduce_noise(
        #     y=audio_in,
        #     sr=MODEL_SR,
        #     thresh_n_mult_nonstationary=1.5,
        #     stationary=False,
        #     prop_decrease=0.7,
        # )
        # audio_in = audio_in / np.max(np.abs(audio_in))

        if audio_in.shape[0] >= 1024:
            audio_out = self.model.generate_conversions(audio_in, noise_reduce=True)
            # audio_out = audio_in
        else:
            audio_out = np.zeros_like(audio_in)
        if silence:
            audio_out = np.zeros_like(audio_out)

        # audio_out = audio_in
        # audio_out = audio_in

        # save audio_out to wav file
        audio_out = librosa.core.resample(
            audio_out,
            orig_sr=audio_out.shape[0],
            target_sr=audio_in.shape[0],
            scale=False,
        )
        # audio_out = audio_in

        final_audio = audio_out[hist_audio_num : hist_audio_num + current_audio_num]

        return final_audio
