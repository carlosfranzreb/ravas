"""
Run this script to compile the Mimi encoder and decoder.

TODO: skip the quantizer, but quantize the models.
"""

import os

import torch

from .mimi import init_mimi
from ...utils import resolve_file_path


def compile_onnx():
    os.makedirs(resolve_file_path("onnx/"), exist_ok=True)
    mimi = init_mimi()
    mimi.eval()
    mimi.requires_grad_(False)

    # compile the mimi encoder
    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: float tensor of shape [T]
        Output: float tensor of shape [C, T]
        """
        x.unsqueeze_(0).unsqueeze_(0)
        x = self.conv_encoder(x)
        x = self.transformer_encoder(x)[0]
        x = self._to_framerate(x).squeeze(0).T
        return x

    mimi_encoder = torch.nn.Module()
    mimi_encoder.conv_encoder = mimi._streaming_state.graphed_encoder
    mimi_encoder.transformer_encoder = mimi._streaming_state.graphed_tr_enc
    mimi_encoder._to_framerate = mimi._to_framerate
    mimi_encoder.forward = forward.__get__(mimi_encoder)

    encoder_in = torch.randn((1, 1, 1920), dtype=torch.float32)
    torch.onnx.export(
        mimi_encoder,
        encoder_in,
        resolve_file_path(f"onnx/mimi_encoder.onnx"),
        input_names=["input"],
        output_names=["output"],
    )

    # compile the mimi decoder
    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: float tensor of shape [C, T]
        Output: float tensor of shape [T]
        """
        x = self.quantizer.encode(x.unsqueeze(2))
        x = self.quantizer.decode(x)
        x = self._to_encoder_framerate(x)
        x = self.transformer_decoder(x)[0]
        x = self.conv_decoder(x).squeeze()
        return x

    mimi_decoder = torch.nn.Module()
    mimi_decoder._to_encoder_framerate = mimi._to_encoder_framerate
    mimi_decoder.quantizer = mimi.quantizer
    mimi_decoder.conv_decoder = mimi._streaming_state.graphed_decoder
    mimi_decoder.transformer_decoder = mimi._streaming_state.graphed_tr_dec
    mimi_decoder.forward = forward.__get__(mimi_decoder)

    decoder_in = torch.randn((1, 512), dtype=torch.float32)
    torch.onnx.export(
        mimi_decoder,
        decoder_in,
        resolve_file_path(f"onnx/mimi_decoder.onnx"),
        input_names=["input"],
        output_names=["output"],
    )


if __name__ == "__main__":
    compile_onnx()
