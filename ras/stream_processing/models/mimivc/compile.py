"""
Run this script to compile the Mimi encoder and decoder.

TODO: skip the quantizer, but quantize the models.
"""

import os

import torch
from torch import Tensor
import onnxruntime as ort

from .mimi import init_mimi
from ...utils import resolve_file_path


def compile_onnx():
    os.makedirs(resolve_file_path("onnx/"), exist_ok=True)
    mimi = init_mimi()
    mimi.eval()
    mimi.requires_grad_(False)

    # compile the mimi encoder
    @torch.inference_mode()
    def forward(self, x: Tensor) -> Tensor:
        """
        Input: float tensor of shape [T]
        Output: float tensor of shape [C, T]
        """
        x = x.unsqueeze(0).unsqueeze(0)
        x = self.conv_encoder(x)
        x = self.transformer_encoder(x)[0]
        x = self.downsample(x).squeeze(0).T
        return x

    mimi_encoder = torch.nn.Module()
    mimi_encoder.conv_encoder = mimi.encoder
    mimi_encoder.transformer_encoder = mimi.encoder_transformer
    mimi_encoder.downsample = mimi.downsample
    mimi_encoder.forward = forward.__get__(mimi_encoder)

    encoder_in = torch.randn((1920), dtype=torch.float32)
    torch.onnx.export(
        mimi_encoder,
        encoder_in,
        resolve_file_path(f"onnx/mimi_encoder.onnx"),
        input_names=["input"],
        output_names=["output"],
    )

    # compile the mimi decoder
    @torch.inference_mode()
    def forward(self, x: Tensor) -> Tensor:
        """
        Input: float tensor of shape [C, T]
        Output: float tensor of shape [T]
        """
        x = self.upsample(x)
        x = self.transformer_decoder(x)[0]
        x = self.conv_decoder(x).squeeze()
        return x

    mimi_decoder = torch.nn.Module()
    mimi_decoder.upsample = mimi.upsample
    mimi_decoder.conv_decoder = mimi.decoder
    mimi_decoder.transformer_decoder = mimi.decoder_transformer
    mimi_decoder.forward = forward.__get__(mimi_decoder)

    decoder_in = torch.randn((1, 512, 1), dtype=torch.float32)
    torch.onnx.export(
        mimi_decoder,
        decoder_in,
        resolve_file_path(f"onnx/mimi_decoder.onnx"),
        input_names=["input"],
        output_names=["output"],
    )

    # test that the compiled models output the same values as the torch ones
    onnx_encoder = ort.InferenceSession(resolve_file_path(f"onnx/mimi_encoder.onnx"))
    onnx_decoder = ort.InferenceSession(resolve_file_path(f"onnx/mimi_decoder.onnx"))

    for model_name, model_torch, model_onnx, model_input in zip(
        ["encoder", "decoder"],
        [mimi_encoder, mimi_decoder],
        [onnx_encoder, onnx_decoder],
        [encoder_in, decoder_in],
    ):
        torch_out = model_torch(model_input)
        onnx_out = model_onnx.run(["output"], {"input": model_input.numpy()})[0]
        onnx_out = torch.from_numpy(onnx_out)
        print(model_name, torch.allclose(torch_out, onnx_out, atol=1e-6))


if __name__ == "__main__":
    compile_onnx()
