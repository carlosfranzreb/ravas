"""
Selects targets based on speaker embeddings.

TODO: Optimize interpolation. If glitchiness persists, look into VAD through WavLM
feature classification, as done in URythmic.
"""

import logging
import queue

import torch
from torch import Tensor
from torch.multiprocessing import Queue, Event
import onnxruntime as ort

from ...processor import Converter
from ...utils import clear_queue

from .prev_audio_queue import PrevAudioQueue
from .interpolator import Interpolator


class KnnVC(Converter):
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
        """
        super().__init__(name, config, input_queue, output_queue, log_queue, log_level, ready_signal)

        # model config
        self.device = config["device"]
        self.target_feats_path = config["target_feats_path"]
        self.n_neighbors = config["n_neighbors"]

        # VAD config
        self.vad_frame_length = config["vad"]["frame_length"]
        self.vad_hop_length = config["vad"]["hop_length"]
        self.vad_threshold = config["vad"]["threshold"]

        # initialize the audio queue and the interpolator
        self.audio_queue = PrevAudioQueue(config["prev_audio_queue"])
        self.interpolator = Interpolator(config["interpolator"])

        # initialize the WavLM and HiFiGAN models, compiling them if needed
        input_size = config["prev_audio_queue"]["max_samples"]
        self.wavlm = ort.InferenceSession(f"onnx/wavlm_{input_size}.onnx")
        self.hifigan = ort.InferenceSession(f"onnx/hifigan_{input_size}.onnx")

        # load the target features
        self.target_feats = torch.load(self.target_feats_path)
        logging.info(f"Loaded {self.target_feats.shape[0]} target features")

    def convert(self) -> None:
        """
        Read the input queue, convert the data and put the converted data into the
        sync queue.
        """
        self.logger.info("Start converting audio")
        if self.config["video_file"] is None:
            clear_queue(self.input_queue)

        self.ready_signal.set()
        while True:
            try:
                ttime, data = self.input_queue.get(timeout=1)
                if data is not None:
                    self.logger.debug(f"Converting audio starting at {ttime[0]}")
                    data = self.convert_audio(data)
                else:
                    self.logger.info("Data is null, stopping conversion")
                    self.output_queue.put((None, None))
                    break
                self.output_queue.put((ttime, data))

            except queue.Empty:
                pass
            except EOFError:
                break

    @torch.inference_mode()
    def convert_audio(self, audio_in: Tensor) -> Tensor:
        """
        Convert the audio to the target speaker.
        """
        audio_in = (audio_in / 32768).to(torch.float32)
        self.audio_queue.add(audio_in)

        # if energy is too low, return silence
        energy = rms(audio_in, self.vad_frame_length, self.vad_hop_length)
        if torch.max(energy) < self.vad_threshold:
            self.logger.debug(f"Energy too low ({torch.max(energy)}).")
            return torch.zeros_like(audio_in, dtype=torch.int16)

        # convert the audio
        audio_concat = self.audio_queue.get().unsqueeze(0)
        source_feats = self.wavlm.run(["output"], {"input": audio_concat.numpy()})[0]
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
