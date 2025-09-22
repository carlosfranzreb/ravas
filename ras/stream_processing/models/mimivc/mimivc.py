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
import queue

import torch
from torch import Tensor
from torch.multiprocessing import Queue, Event
import onnxruntime as ort

from ...processor import Converter
from ...utils import clear_queue, resolve_file_path
from ..knnvc.knnvc import convert_vecs


class MimiVC(Converter):
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

        # initialize Mimi
        input_size = config["input_size"]
        self.mimi_encoder = ort.InferenceSession(
            resolve_file_path(f"onnx/mimi_encoder_{input_size}.onnx")
        )
        self.mimi_decoder = ort.InferenceSession(
            resolve_file_path(f"onnx/mimi_decoder_{input_size}.onnx")
        )

        # load the target features
        self.target_feats = torch.load(self.target_feats_path)
        logging.info(f"Loaded {self.target_feats.shape[0]} target features")

    def convert(self) -> None:
        """
        Read the input queue, convert the data and put the converted data into the
        sync queue.
        TODO: this is the same as kNN-VC, merge this into a parent class.
        """
        pass

    @torch.inference_mode()
    def convert_audio(self, audio_in: Tensor) -> Tensor:
        """
        Convert the audio to the target speaker.
        """
        audio_in = (audio_in / 32768).to(torch.float32)

        # convert the audio
        source_feats = self.mimi_encoder.run(["output"], {"input": audio_in.numpy()})[0]
        source_feats = torch.tensor(source_feats, dtype=torch.float32)
        conv_feats = convert_vecs(source_feats, self.target_feats, self.n_neighbors)
        out = self.mimi_decoder.run(
            ["output"], {"input": conv_feats.unsqueeze(0).numpy()}
        )[0]
        out = torch.tensor(out, dtype=torch.float32).squeeze()

        # transform and return the converted audio
        audio_out = torch.clamp(audio_out, -1.0, 1.0)
        audio_out = (audio_out * 32768).to(torch.int16)
        return audio_out
