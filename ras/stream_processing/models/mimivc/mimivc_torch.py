"""
Adapt Mimi for voice conversion following the kNN-VC approach.
The feature embeddings output by the encoder's transformer are
compared with cosine similarity to convert source features to
target features. They encode cosine similarity due to the WavLM
distillation.

! moshi must be installed inside this directory:

```bash
git clone git@github.com:kyutai-labs/moshi.git
pip install moshi
```
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
        self.mimi = init_mimi()
        self.conv_enc_state = None
        self.conv_dec_state = None
        self.tr_enc_states = [None, None, None, None]
        self.tr_dec_states = [None, None, None, None]

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
        source_feats, self.conv_enc_state = self.mimi.encoder(
            audio_in, self.conv_enc_state
        )
        source_feats, *self.tr_enc_states = self.mimi.encoder_transformer(
            source_feats, *self.tr_enc_states
        )
        source_feats = self.mimi.downsample(source_feats[0]).squeeze(0).T

        # convert the audio
        conv_feats = convert_vecs(source_feats, self.target_feats, self.n_neighbors)

        # decode the audio
        codes = self.mimi.quantizer.encode(conv_feats.unsqueeze(2))
        conv_feats = self.mimi.quantizer.decode(codes)
        conv_feats = self.mimi.upsample(conv_feats)
        conv_feats, *self.tr_dec_states = self.mimi.decoder_transformer(
            conv_feats, *self.tr_dec_states
        )
        audio_out, self.conv_dec_state = self.mimi.decoder(
            conv_feats[0], self.conv_dec_state
        )
        audio_out.squeeze_()

        # transform and return the converted audio
        audio_out = torch.clamp(audio_out, -1.0, 1.0)
        audio_out = (audio_out * 32768).to(torch.int16)
        return audio_out
