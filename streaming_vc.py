from queue import Queue

import torch
import torchaudio
import numpy as np

from LLVC.infer import load_model


class StreamingLLVC:
    def __init__(self, queue_size: int = 5):
        """
        Initialize the LLVC model.
        """
        self.model, self.model_sr = load_model(
            "checkpoints/llvc/models/checkpoints/llvc/G_500000.pth",
            "LLVC/experiments/llvc/config.json",
        )
        self.chunk_len = int(self.model.dec_chunk_size * self.model.L)
        self.enc_buf, self.dec_buf, self.out_buf = self.model.init_buffers(
            1, torch.device("cpu")
        )
        self.convnet_pre_ctx = None
        if hasattr(self.model, "convnet_pre"):
            self.convnet_pre_ctx = self.model.convnet_pre.init_ctx_buf(
                1, torch.device("cpu")
            )
        self.queue = Queue(maxsize=queue_size)
        self.idx = 0

    def __call__(self, frame: np.ndarray, audio_sr: int) -> np.ndarray:
        """
        1. Add the frame to the queue, resampled to the model's SR and as a tensor.
        2. Concatenate the queue into a single tensor and convert it.
        """
        max_value = np.max(np.abs(frame))
        frame = (frame / max_value).astype("float32")
        frame = torchaudio.transforms.Resample(
            orig_freq=audio_sr, new_freq=self.model_sr
        )(torch.from_numpy(frame))
        self.idx += 1
        self.queue.put(frame.squeeze(0))
        frames = torch.cat(list(self.queue.queue))
        converted = self.convert(frames)[:, -frame.shape[1] :]
        converted = torchaudio.transforms.Resample(
            orig_freq=self.model_sr, new_freq=audio_sr
        )(converted)

        out = (converted.numpy() * max_value).round().astype("int16")
        return out

    def convert(self, audio: torch.tensor) -> torch.tensor:
        n_samples = audio.shape[0]
        if n_samples % self.chunk_len != 0:
            pad_len = self.chunk_len - (len(audio) % self.chunk_len)
            audio = torch.nn.functional.pad(audio, (0, pad_len))

        audio = torch.cat((audio[self.model.L :], torch.zeros(self.model.L)))
        audio_chunks = torch.split(audio, self.chunk_len)

        # add lookahead context from prev chunk
        new_audio_chunks = []
        for i, a in enumerate(audio_chunks):
            if i == 0:
                front_ctx = torch.zeros(self.model.L * 2)
            else:
                front_ctx = audio_chunks[i - 1][-self.model.L * 2 :]
            new_audio_chunks.append(torch.cat([front_ctx, a]))
        audio_chunks = new_audio_chunks

        outputs = []
        with torch.inference_mode():
            for chunk in audio_chunks:
                (
                    output,
                    self.enc_buf,
                    self.dec_buf,
                    self.out_buf,
                    self.convnet_pre_ctx,
                ) = self.model(
                    chunk.unsqueeze(0).unsqueeze(0),
                    self.enc_buf,
                    self.dec_buf,
                    self.out_buf,
                    self.convnet_pre_ctx,
                    pad=(not self.model.lookahead),
                )
                outputs.append(output)
        outputs = torch.cat(outputs, dim=2)
        outputs = outputs[:, :, :n_samples].squeeze(0)
        return outputs
