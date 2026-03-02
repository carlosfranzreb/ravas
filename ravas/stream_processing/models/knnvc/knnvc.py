"""
Selects targets based on speaker embeddings.

TODO: Optimize interpolation. If glitchiness persists, look into VAD through WavLM
feature classification, as done in URythmic.
"""

import logging

import torch
from torch import Tensor
from torch.multiprocessing import Queue, Event
import onnxruntime as ort

from ...processor import AudioConverter
from ...utils import resolve_file_path, compute_target_features

from .prev_audio_queue import PrevAudioQueue
from .interpolator import Interpolator


class KnnVC(AudioConverter):
    def __init__(
        self,
        name: str,
        config: dict,
        input_queue: Queue,
        output_queue: Queue,
        log_queue: Queue,
        log_level: str,
        ready_signal: Event,
    ) -> None:
        """
        Initialize WavLM and the HiFiGAN models, and load the target features.

        If `target_feats_path` is a file, load the target features from it.
        Otherwise, compute the target features from the given LibriSpeech directory, and
        dump them to a file in the `target_feats` directory.

        NOTE: if `target_feats_path` is a relative path, it will be resolved against the application directory.
        """
        super().__init__(
            name, config, input_queue, output_queue, log_queue, log_level, ready_signal
        )

        # model config
        self.device = config["device"]
        self.target_feats_path = resolve_file_path(config["target_feats_path"])
        self.n_neighbors = config["n_neighbors"]

        # initialize the audio queue and the interpolator
        self.audio_queue = PrevAudioQueue(config["prev_audio_queue"])
        self.interpolator = Interpolator(config["interpolator"])

        # initialize previous context
        self.prev = (
            config["prev_ctx"]["use_previous_ctx"]
            and config["prev_ctx"]["max_samples"] > 0
        )
        self.previous_samples = config["prev_ctx"]["max_samples"]
        self.prev_context = PrevAudioQueue(config["prev_ctx"])

        self.input_size = config["prev_audio_queue"]["max_samples"]
        self.model_size = (
            self.input_size + self.previous_samples if self.prev else self.input_size
        )

        # initialize the WavLM and HiFiGAN models, compiling them if needed
        self.wavlm = ort.InferenceSession(
            resolve_file_path(f"onnx/wavlm_{self.model_size}.onnx")
        )
        self.hifigan = ort.InferenceSession(
            resolve_file_path(f"onnx/hifigan_{self.model_size}.onnx")
        )

        # load the target features
        self.target_feats = compute_target_features(
            self.target_feats_path, config["n_cluster"], config["use_expressiveness"]
        )
        logging.info(f"Loaded {self.target_feats.shape[0]} target features")

    @torch.inference_mode()
    def convert_audio(self, audio_in: Tensor) -> Tensor:
        """
        Convert the audio to the target speaker.
        """
        audio_in = (audio_in / 32768).to(torch.float32)
        self.audio_queue.add(audio_in)

        # convert the audio
        audio_concat = self.audio_queue.get()
        audio_chunk = audio_concat
        # add context if previous context is enabled
        if self.prev:
            prev_ctx = self.prev_context.get()
            audio_chunk = torch.cat([prev_ctx, audio_chunk], dim=-1)
            self.prev_context.add(audio_concat)

        audio_chunk = audio_chunk.unsqueeze(0)

        source_feats = self.wavlm.run(["output"], {"input": audio_chunk.numpy()})[0]
        source_feats = torch.tensor(source_feats, dtype=torch.float32)
        if source_feats.ndim == 3:
            source_feats = source_feats.squeeze(0)
        conv_feats = convert_vecs(source_feats, self.target_feats, self.n_neighbors)
        out = self.hifigan.run(["output"], {"input": conv_feats.unsqueeze(0).numpy()})[
            0
        ]
        out = torch.tensor(out, dtype=torch.float32).squeeze()

        # interpolate the converted audio with the previous samples
        audio_out = out[-audio_in.shape[0] :]
        audio_out = self.interpolator.interpolate(audio_out)

        # transform and return the converted audio
        audio_out = torch.clamp(audio_out, -1.0, 1.0)
        audio_out = (audio_out * 32768).to(torch.int16)
        return audio_out


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
