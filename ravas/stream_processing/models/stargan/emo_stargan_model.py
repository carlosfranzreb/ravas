import numpy as np
import torch
import torchaudio
import sys

sys.path.append("emo-stargan")
import noisereduce as nr
import yaml
from parallel_wavegan.utils import load_model
from munch import Munch
from Models.models import Generator, MappingNetwork, StyleEncoder
from Utils.JDC.model import JDCNet


class DefaultDict(dict):
    """Implementation of perl's autovivification feature."""

    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value


class Model:
    def __init__(
        self,
        f0_path,
        vocoder_path,
        vocoder_config,
        stargan_path,
        stargan_config,
        speaker_file,
        sample_rate=24000,
        speaker_nr=19,
        device="cuda",
    ):
        self.sr = sample_rate
        self.device = device
        self.vocoder_path = vocoder_path
        self.vocoder_config = vocoder_config
        self.stargan_path = stargan_path
        self.stargan_config = stargan_config
        self.f0_path = f0_path
        self.speaker_nr = speaker_nr
        self.speaker_file = speaker_file
        self.speaker_id = self.get_speaker_id(speaker_file, speaker_nr)
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300
        )
        self.mean, self.std = -4, 4

        (
            self.emostargan,
            self.F0_model,
            self.vocoder,
            self.reference_embedding,
        ) = self.build_inference_pipeline(device)

    def get_speaker_id(self, speaker_file, speakrer_nr):
        def get_spkr_name(fname):
            fnm_arr = file_name.split("/")
            if "ESD" in fname:
                name = fnm_arr[-1]
                name = name.split("_")
                name = name[0].replace("00", "")
            else:
                name = fnm_arr[-2]
            return name

        with open(speaker_file) as file:
            lines = file.readlines()

        for line in lines:
            file_name, sp_id = line.split("|")
            sp_id = int(sp_id)
            spkr_nm = get_spkr_name(file_name)
            if int(spkr_nm.replace("p", "")) == int(speakrer_nr):
                return sp_id

    def set_speaker(self, speaker_nr):
        self.speaker_nr = speaker_nr
        self.speaker_id = self.get_speaker_id(self.speaker_file, speaker_nr)
        self.reference_embedding = self.compute_style(self.emostargan)

    def compute_style(self, starganv2):
        """get speaker embeddings"""
        speaker = int(self.speaker_id)
        label = torch.LongTensor([speaker]).to(self.device)
        latent_dim = starganv2.mapping_network.shared[0].in_features
        ref = starganv2.mapping_network(
            torch.randn(1, latent_dim).to(self.device), label
        )

        return (ref, label)

    def build_inference_pipeline(self, device="cuda"):
        def build_model(model_params):
            args = Munch(model_params)
            generator = Generator(
                args.dim_in,
                args.style_dim,
                args.max_conv_dim,
                w_hpf=args.w_hpf,
                F0_channel=args.F0_channel,
            )

            # Can use both to generate embeddings...
            mapping_network = MappingNetwork(
                args.latent_dim,
                args.style_dim,
                args.num_domains,
                hidden_dim=args.max_conv_dim,
            )
            style_encoder = StyleEncoder(
                args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim
            )

            nets_ema = Munch(
                generator=generator,
                mapping_network=mapping_network,
                style_encoder=style_encoder,
            )

            return nets_ema

        # load F0 model
        F0_model = JDCNet(num_class=1, seq_len=192)
        params = torch.load(self.f0_path, map_location=device)["net"]
        F0_model.load_state_dict(params)
        _ = F0_model.eval()
        F0_model = F0_model.to(device)

        # load hifigan vocoder
        # load config.yml
        with open(self.vocoder_config) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        vocoder = load_model(self.vocoder_path, config).to(device).eval()
        vocoder.remove_weight_norm()
        _ = vocoder.eval()

        # load emo-stargan model
        with open(self.stargan_config) as f:
            emostargan_config = yaml.safe_load(f)

        emostargan = build_model(model_params=emostargan_config["model_params"])
        params = torch.load(self.stargan_path, map_location=device)
        params = params["model_ema"]
        _ = [
            emostargan[key].load_state_dict(params[key], strict=False)
            for key in emostargan
        ]
        _ = [emostargan[key].eval() for key in emostargan]
        emostargan.style_encoder = emostargan.style_encoder.to(device)
        emostargan.mapping_network = emostargan.mapping_network.to(device)
        emostargan.generator = emostargan.generator.to(device)

        # initialise speaker details from config file
        reference_embedding = self.compute_style(emostargan)

        return emostargan, F0_model, vocoder, reference_embedding

    def generate_conversions(self, wav, noise_reduce=False):
        """
        generate conversion for the selected speakers, whose speaker embeddings were generated
        Args:
            wav:
            starganv2:
            reference_embeddings:
            F0_model:
            vocoder:
            noise_reduce:

        Returns:

        """

        def preprocess(wave):
            wave_tensor = torch.from_numpy(wave).float()
            mel_tensor = self.to_mel(wave_tensor)
            mel_tensor = (
                torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean
            ) / self.std
            return mel_tensor

        source = preprocess(wav).to(self.device)

        # print('Target Speakers:')
        ref, _ = self.reference_embedding
        # start = time.time()
        with torch.no_grad():
            f0_feat = self.F0_model.get_feature_GAN(source.unsqueeze(1))
            out = self.emostargan.generator(source.unsqueeze(1), ref, F0=f0_feat)
            c = out.transpose(-1, -2).squeeze()
            recon = self.vocoder.inference(c)
            recon = recon.view(-1).cpu().numpy()
            recon = recon / np.max(np.abs(recon))
        # end = time.time()
        # t.append(end - start)
        if noise_reduce:
            recon = nr.reduce_noise(y=recon, sr=self.sr, prop_decrease=0.5)

        # print('Average processing time per speaker: %.3f sec' % (sum(t) / len(t)))
        return recon
