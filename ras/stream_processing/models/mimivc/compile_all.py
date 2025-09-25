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

    # TODO: the current states are not tensors
    inputs = {
        "encoder": torch.randn((1, 1, 1920), dtype=torch.float32),
        "encoder_state": None,
        "encoder_transformer": torch.randn((1, 1, 1920), dtype=torch.float32),
        "encoder_transformer_state": None,
        "downsample": torch.randn((1, 1, 1920), dtype=torch.float32),
        "upsample": torch.randn((1, 1, 1920), dtype=torch.float32),
        "decoder": torch.randn((1, 1, 1920), dtype=torch.float32),
        "decoder_state": None,
        "decoder_transformer": torch.randn((1, 1, 1920), dtype=torch.float32),
        "decoder_transformer_state": None,
    }

    encoder_in = torch.randn((1, 1, 1920), dtype=torch.float32)
    state = None
    for method in ["encoder", "encoder_transformer", "downsample"]:
        dump_file = resolve_file_path(f"onnx/mimi_all_{method}.onnx")
        torch.onnx.export(
            getattr(mimi, method),
            encoder_in,
            dump_file,
            input_names=["input"],
            output_names=["output"],
        )

        # test that the compiled models output the same values as the torch ones
        model_onnx = ort.InferenceSession(dump_file)
        torch_out = getattr(mimi, method)(encoder_in)
        if isinstance(torch_out, list):
            torch_out = torch_out[0]
        onnx_out = model_onnx.run(["output"], {"input": encoder_in.numpy()})[0]
        onnx_out = torch.from_numpy(onnx_out)
        print(method, torch.allclose(torch_out, onnx_out, atol=1e-6))

        # update encoder_in for next method
        encoder_in = torch_out


if __name__ == "__main__":
    compile_onnx()
