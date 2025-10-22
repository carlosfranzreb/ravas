"""
Adapt Mimi for voice conversion following the kNN-VC approach.
The feature embeddings output by the encoder's transformer are
compared with cosine similarity to convert source features to
target features. They encode cosine similarity due to the WavLM
distillation.
"""

import logging

import torch
from torch import Tensor
from torch.multiprocessing import Queue, Event

from ...processor import AudioConverter
from ...utils import resolve_file_path
from ..knnvc.knnvc import convert_vecs
from .mimi import init_mimi


class TorchMimiVC(AudioConverter):
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
        Initialize Mimi and load the target features.

        If `target_feats_path` is a file, load the target features from it.
        Otherwise, compute the target features from the given LibriSpeech directory, and
        dump them to a file in the `target_feats` directory.

        NOTE: if `target_feats_path` is a relative path, it will be resolved against the
        application directory.
        """
        super().__init__(
            name, config, input_queue, output_queue, log_queue, log_level, ready_signal
        )

        # model config
        self.device = config["device"]
        self.target_feats_path = resolve_file_path(config["target_feats_path"])
        self.n_neighbors = config["n_neighbors"]

        # load the target features
        self.target_feats = torch.load(self.target_feats_path)
        logging.info(f"Loaded {self.target_feats.shape[0]} target features")

        # load mimi and declare its states
        (
            self.mimi,
            self.enc_state,
            self.tr_enc_state,
            self.downsample_state,
            self.upsample_state,
            self.tr_dec_state,
            self.dec_state,
        ) = init_mimi()

    @torch.inference_mode()
    def convert_audio(self, audio_in: Tensor) -> Tensor:
        """
        Convert the audio to the target speaker.
        """

        audio_in = (audio_in / 32768).to(torch.float32)
        if audio_in.shape[0] < 1920:
            audio_in = torch.nn.functional.pad(
                audio_in, (0, 1920 - audio_in.shape[0]), "constant", 0
            )

        # gather source features
        audio_in.unsqueeze_(0).unsqueeze_(0)
        source_feats, *self.enc_state = self.mimi.encoder(audio_in, *self.enc_state)
        source_feats, *self.tr_enc_state = self.mimi.encoder_transformer(
            source_feats, *self.tr_enc_state
        )
        source_feats, *self.downsample_state = self.mimi.downsample(
            source_feats, *self.downsample_state
        )

        # convert the audio
        source_feats = source_feats.squeeze(0).T
        conv_feats = convert_vecs(source_feats, self.target_feats, self.n_neighbors)
        conv_feats = conv_feats.T.unsqueeze(0)

        # decode the audio
        semantic_feats = self.mimi.quantizer.rvq_first.output_proj(
            self.mimi.quantizer.rvq_first.input_proj(conv_feats)
        )
        acoustic_feats = self.mimi.quantizer.rvq_rest.output_proj(
            self.mimi.quantizer.rvq_rest.input_proj(conv_feats)
        )
        conv_feats = semantic_feats + acoustic_feats

        conv_feats, *self.upsample_state = self.mimi.upsample(
            conv_feats, *self.upsample_state
        )

        conv_feats, *self.tr_dec_state = self.mimi.decoder_transformer(
            conv_feats, *self.tr_dec_state
        )
        audio_out, *self.dec_state = self.mimi.decoder(conv_feats, *self.dec_state)

        audio_out.squeeze_()

        # transform and return the converted audio
        audio_out = torch.clamp(audio_out, -1.0, 1.0)
        audio_out = (audio_out * 32768).to(torch.int16)
        return audio_out
