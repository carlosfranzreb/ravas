"""
Run this script to compile the Mimi encoder and decoder.
"""

import os

import torch
from torch import Tensor
import onnxruntime as ort
import numpy as np

from .mimi import init_mimi
from .mimi_functional.quantization import SplitResidualVectorQuantizer

from ...utils import resolve_file_path


class Quantization(torch.nn.Module):
    def __init__(self, quantizer: SplitResidualVectorQuantizer):
        super().__init__()
        self.quantizer = quantizer

    def forward(self, x: Tensor) -> list[Tensor]:
        """Project the semantic and acoustic tokens and add them."""
        semantic_feats = self.quantizer.rvq_first.output_proj(
            self.quantizer.rvq_first.input_proj(x)
        )
        acoustic_feats = self.quantizer.rvq_rest.output_proj(
            self.quantizer.rvq_rest.input_proj(x)
        )
        x = semantic_feats + acoustic_feats

        return [x]


def compile_onnx():
    os.makedirs(resolve_file_path("onnx/"), exist_ok=True)
    mimi = init_mimi()[0]
    mimi.quantization = Quantization(mimi.quantizer)
    mimi.eval()
    mimi.requires_grad_(False)

    x = torch.randn((1, 1, 1920), dtype=torch.float32)
    for method in [
        "encoder",
        "encoder_transformer",
        "downsample",
        "quantization",
        "upsample",
        "decoder_transformer",
        "decoder",
    ]:
        dump_file = resolve_file_path(f"onnx/mimi_{method}.onnx")
        dump_file_args = resolve_file_path(f"onnx/mimi_{method}_args.npy")

        # get the streaming state
        if method != "quantization":
            state = getattr(mimi, method)._init_streaming_state()
            input_tuple = (x, *state)
        else:
            input_tuple = (x,)

        torch.onnx.export(
            getattr(mimi, method),
            input_tuple,
            dump_file,
            dynamo=True,
        )

        # create and dump the onnx inputs
        model_onnx = ort.InferenceSession(dump_file)
        input_onnx = {
            onnx_input.name: input_tuple[idx].numpy()
            for idx, onnx_input in enumerate(model_onnx.get_inputs())
        }
        np.save(dump_file_args, input_onnx)

        # ensure that the inputs and outputs have the same shape and different sizes
        onnx_out = model_onnx.run(None, input_onnx)
        for idx, value_in in enumerate(input_onnx.values()):
            if idx == 0:
                continue

            value_out = onnx_out[idx]
            assert [value_out.shape, value_in.shape]
            if value_in.size > 0:
                assert not np.allclose(value_out, value_in, atol=1e-2)

        # compare the outputs of the torch and onnx models
        torch_out = getattr(mimi, method)(*input_tuple)
        if method != "quantization":
            print(f"\tComparing outputs {method}")
            compare_outputs(torch_out, onnx_out)
        else:
            torch_out = [torch.randn((1, 512, 1), dtype=torch.float32)]

        # update input for next module
        x = torch_out[0]


def compare_outputs(torch_out: list[Tensor], onnx_out: list[Tensor]):
    for out_idx in range(len(torch_out)):
        onnx_tensor = torch.from_numpy(onnx_out[out_idx])
        for atol in torch.logspace(-8, -1, steps=8):
            if torch.allclose(torch_out[out_idx], onnx_tensor, atol=atol):
                break

        print(out_idx, "1e{}".format(int(torch.log10(atol).item())))


if __name__ == "__main__":
    compile_onnx()
