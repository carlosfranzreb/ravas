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

import numpy as np
import torch
from torch import Tensor
from torch.multiprocessing import Queue, Event
import onnxruntime as ort

from ...processor import AudioConverter
from ...utils import resolve_file_path
from ..knnvc.knnvc import convert_vecs
from .mimi import init_mimi


class MimiVC(AudioConverter):
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

        self.encoder = ort.InferenceSession(
            resolve_file_path(f"onnx/mimi_encoder.onnx")
        )
        self.encoder_transformer = ort.InferenceSession(
            resolve_file_path(f"onnx/mimi_encoder_transformer.onnx")
        )
        self.downsample = ort.InferenceSession(
            resolve_file_path(f"onnx/mimi_downsample.onnx")
        )
        self.upsample = ort.InferenceSession(
            resolve_file_path(f"onnx/mimi_upsample.onnx")
        )
        self.decoder = ort.InferenceSession(
            resolve_file_path(f"onnx/mimi_decoder.onnx")
        )
        self.decoder_transformer = ort.InferenceSession(
            resolve_file_path(f"onnx/mimi_decoder_transformer.onnx")
        )

        # load mimi and declare its states
        mimi = init_mimi()[0]
        self.quantizer = mimi.quantizer

        self.encoder_args = np.load(
            resolve_file_path(f"onnx/mimi_encoder_args.npy"), allow_pickle=True
        ).item()
        self.encoder_transformer_args = np.load(
            resolve_file_path(f"onnx/mimi_encoder_transformer_args.npy"),
            allow_pickle=True,
        ).item()
        self.downsample_args = np.load(
            resolve_file_path(f"onnx/mimi_downsample_args.npy"), allow_pickle=True
        ).item()
        self.upsample_args = np.load(
            resolve_file_path(f"onnx/mimi_upsample_args.npy"), allow_pickle=True
        ).item()
        self.decoder_args = np.load(
            resolve_file_path(f"onnx/mimi_decoder_args.npy"), allow_pickle=True
        ).item()
        self.decoder_transformer_args = np.load(
            resolve_file_path(f"onnx/mimi_decoder_transformer_args.npy"),
            allow_pickle=True,
        ).item()

    @torch.inference_mode()
    def convert_audio(self, audio_in: Tensor) -> Tensor:
        """
        Convert the audio to the target speaker.
        # TODO: avoid back and forth casting between torch and numpy
        """

        audio_in = (audio_in / 32768).to(torch.float32)
        if audio_in.shape[0] < 1920:
            audio_in = torch.nn.functional.pad(
                audio_in, (0, 1920 - audio_in.shape[0]), "constant", 0
            )

        # gather source features
        x = audio_in.unsqueeze(0).unsqueeze(0).numpy()
        x, self.encoder_args = run_onnx_model(self.encoder, self.encoder_args, x)

        x, self.encoder_transformer_args = run_onnx_model(
            self.encoder_transformer, self.encoder_transformer_args, x
        )

        x, self.downsample_args = run_onnx_model(
            self.downsample, self.downsample_args, x
        )

        source_feats = torch.from_numpy(x).squeeze(0).T

        # convert the audio
        conv_feats = convert_vecs(source_feats, self.target_feats, self.n_neighbors)
        codes = self.quantizer.encode(conv_feats.unsqueeze(2))
        conv_feats = self.quantizer.decode(codes)

        # decode the audio
        x = conv_feats.numpy()
        x, self.upsample_args = run_onnx_model(self.upsample, self.upsample_args, x)

        x, self.decoder_transformer_args = run_onnx_model(
            self.decoder_transformer, self.decoder_transformer_args, x
        )

        x, self.decoder_args = run_onnx_model(self.decoder, self.decoder_args, x)
        audio_out = torch.from_numpy(x)

        # transform and return the converted audio
        audio_out = torch.clamp(audio_out, -1.0, 1.0)
        audio_out = (audio_out * 32768).to(torch.int16)
        return audio_out


def run_onnx_model(model, model_args, new_input) -> tuple:
    """
    Run the model, update its state and return it along with the output.
    Uses the model's output names to correctly map outputs to inputs.
    """
    model_args["x"] = new_input
    model_out = model.run(None, model_args)

    # Update states using the output names
    # ! The names are not the same; this doesn't work
    for idx, key in enumerate(model_args):
        if idx == 0:
            out = model_out[idx]
        else:
            model_args[key] = model_out[idx]

    return out, model_args
