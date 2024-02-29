"""
Selects targets based on speaker embeddings.
"""

import os
import json
from glob import glob
import logging

import torch
from torch import Tensor
import torchaudio

from stream_processing.Processor import ProcessingCallback
from stream_processing.models.knnvc.wavlm.model import WavLM, WavLMConfig
from stream_processing.models.knnvc.hifigan import Generator, AttrDict


class Converter(ProcessingCallback):
    def __init__(
        self,
        target_feats_path: str,
        device: str,
        hifigan_cfg: str,
        hifigan_ckpt: str,
        wavlm_ckpt: str,
        wavlm_layer: int,
        n_neighbors: int,
    ) -> None:
        """
        Args:
            target_feats: either a torch file with a speaker's target features or a
                LibriSpeech directory of a speaker.
            device: the device where the model should run.
            hifigan_cfg: the path to the HiFiGAN configuration file.
            hifigan_ckpt: the path to the HiFiGAN checkpoint.
            wavlm_ckpt: the path to the WavLM checkpoint.
            wavlm_layer: the layer of the WavLM model to use for feature extraction.
            n_neighbors: the number of neighbors to average when converting a feature.
        """
        super().__init__()
        self.device = device
        self.target_feats_path = target_feats_path
        self.hifigan_cfg = hifigan_cfg
        self.hifigan_ckpt = hifigan_ckpt
        self.wavlm_ckpt = wavlm_ckpt
        self.wavlm_layer = wavlm_layer
        self.n_neighbors = n_neighbors

    def init_callback(self):
        """
        Initialize and return the model and target features.

        If `target_feats_path` is a file, load the target features from it.
        Otherwise, compute the target features from the given LibriSpeech directory, and
        dump them to a file in the `target_feats` directory.
        """
        # initialize the WavLM model
        ckpt = torch.load(self.wavlm_ckpt, map_location=self.device)
        wavlm = WavLM(WavLMConfig(ckpt["cfg"]))
        wavlm.load_state_dict(ckpt["model"])
        wavlm.eval()

        # initialize the HiFiGAN model
        hifigan = Generator(AttrDict(json.load(open(self.hifigan_cfg))))
        hifigan.load_state_dict(
            torch.load(self.hifigan_ckpt, map_location=self.device)["generator"]
        )
        hifigan.eval()
        hifigan.remove_weight_norm()

        # if target_feats is a file, load the target features
        if os.path.isfile(self.target_feats_path):
            target_feats = torch.load(self.target_feats_path)
            logging.info(f"Loaded {target_feats.shape[0]} target features")

        # otherwise, compute the target features from the given LibriSpeech directory
        else:
            target_feats = list()
            for audiofile in glob(self.target_feats_path + "/*.flac"):
                audio = torchaudio.load(audiofile)[0].to(self.device)
                feats = wavlm.extract_features(audio, output_layer=self.wavlm_layer)[0]
                target_feats.append(feats.squeeze(0))
            target_feats = torch.cat(target_feats, dim=0)

            # dump the features
            dump_file = os.path.join(
                "target_feats", os.path.basename(self.target_feats_path) + ".pt"
            )
            os.makedirs("target_feats", exist_ok=True)
            torch.save(target_feats, dump_file)
            logging.info(
                f"Dumped {target_feats.shape[0]} target features to {dump_file}"
            )

        history = []
        time_history = []
        return [
            wavlm,
            hifigan,
            target_feats,
            self.wavlm_layer,
            self.n_neighbors,
            history,
            time_history,
        ]

    def callback(
        self,
        dtime: Tensor,
        audio: Tensor,
        wavlm: WavLM,
        hifigan: Generator,
        target_feats: Tensor,
        wavlm_layer: int,
        n_neighbors: int,
        history: list,
        time_history: list,
    ) -> tuple[list, Tensor]:
        """Convert the given audio"""
        audio = audio.to(torch.float32) / 32768

        # if energy is too low, return silence
        frame_length = 2048  # TODO: make this a config parameter
        hop_length = 512  # TODO: make this a config parameter
        energy = rms(audio, frame_length, hop_length)
        if torch.max(energy) < 0.023:  # TODO: make this a config parameter
            history = [audio]
            return dtime, torch.zeros_like(audio, dtype=torch.int16)

        # append the old audio
        audio_concat = torch.cat(history + [audio], dim=0)

        # convert the audio
        source_feats = wavlm.extract_features(
            audio_concat.unsqueeze(0), output_layer=wavlm_layer
        )[0]
        if source_feats.ndim == 3:
            source_feats = source_feats.squeeze(0)
        conv_feats = convert_vecs(source_feats, target_feats, n_neighbors)
        out = hifigan(conv_feats.unsqueeze(0)).squeeze()

        # extract the converted audio and update the history
        out = out[: audio.shape[0]]
        history = [audio]

        # float32 to int16
        out = torch.clamp(out, -1.0, 1.0)
        out = (audio * 32768).to(torch.int16)
        return dtime, out


def convert_vecs(source_vecs: Tensor, target_vecs: Tensor, n_neighbors: int) -> Tensor:
    """
    Given the WavLM vecs of the source and target audios, convert them with the
    KnnVC matching algorithm.

    Args:
        source_vec: tensor of shape (n_vecs_s, vec_dim)
        target_vecs: tensor of shape (n_vecs_t, vec_dim)
        n_neighbors: the number of neighbors to average when converting a feature.

    Returns:
        converted wavLM vectors: tensor of shape (n_vecs_s, vec_dim)
    """
    cos_sim = cosine_similarity(source_vecs, target_vecs)
    best = cos_sim.topk(k=n_neighbors, dim=1)
    return target_vecs[best.indices].mean(dim=1)


def cosine_similarity(tensor_a: Tensor, tensor_b: Tensor) -> Tensor:
    """
    Compute the cosine similarity among all vectors in `tensor_a` and `tensor_b`.

    Args:
        tensor_a: tensor of shape (n_vecs_a, vec_dim)
        tensor_b: tensor of shape (n_vecs_b, vec_dim)

    Returns:
        cosine similarity tensor: tensor of shape (n_vecs_a, n_vecs_b)
    """
    dot_product = torch.matmul(tensor_a, tensor_b.transpose(-1, -2))
    source_norm = torch.norm(tensor_a, dim=-1)
    target_norm = torch.norm(tensor_b, dim=-1)
    cos_sim = dot_product / torch.outer(source_norm, target_norm)
    return cos_sim


def rms(audio: Tensor, frame_length: int, hop_length: int) -> Tensor:
    """
    Compute root-mean-square (RMS) value for each frame from the audio samples.

    Args:
        audio (shape (n)): audio time series.
        frame_length : no. of samples for energy calculation.
        hop_length : hop length for STFT.

    Returns:
        rms (shape (t)): RMS value for each frame
    """

    # if the audio comprises only one frame, return its RMS
    if audio.shape[0] <= frame_length:
        return audio.pow(2).mean().sqrt()

    return torch.sqrt(
        torch.mean(audio.unfold(0, frame_length, hop_length).pow(2), dim=-1)
    )
