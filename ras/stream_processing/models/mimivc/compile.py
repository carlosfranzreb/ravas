"""
Run this script to compile the Mimi encoder and decoder.

TODO: the models are not accurately compiled. Their error tolerances are not
below 1e-4, whereas the encoders have tolerances of 1e-7. There must be something
wrong there.

On another note, the current compiled implementation is slower than the pytorch
one. Maybe the encoders and decoders have to be grouped together for better
performance.
"""

import os

import torch
from torch import Tensor
import onnxruntime as ort
import numpy as np

from .mimi import init_mimi
from ...utils import resolve_file_path


def compile_onnx():
    os.makedirs(resolve_file_path("onnx/"), exist_ok=True)
    mimi = init_mimi()[0]
    mimi.eval()
    mimi.requires_grad_(False)

    x = torch.randn((1, 1, 1920), dtype=torch.float32)
    for method in [
        "encoder",
        "encoder_transformer",
        "downsample",
        "upsample",
        "decoder_transformer",
        "decoder",
    ]:
        dump_file = resolve_file_path(f"onnx/mimi_{method}.onnx")
        dump_file_args = resolve_file_path(f"onnx/mimi_{method}_args.npy")

        # get the streaming state
        if "sample" not in method:
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
        print(f"\tComparing outputs {method}")
        torch_out = getattr(mimi, method)(*input_tuple)
        compare_outputs(torch_out, onnx_out)

        # update input for next module
        x = torch_out[0]
        if method == "downsample":
            x = mimi.quantizer.encode(x.unsqueeze(0))
            x = mimi.quantizer.decode(x)
        elif method == "upsample":
            x = x.unsqueeze(0)


def compare_outputs(torch_out: list, onnx_out: list):
    for out_idx in range(len(torch_out)):
        onnx_tensor = torch.from_numpy(onnx_out[out_idx])
        for atol in torch.logspace(-8, -1, steps=8):
            if torch.allclose(torch_out[out_idx], onnx_tensor, atol=atol):
                break

        print(out_idx, "1e{}".format(int(torch.log10(atol).item())))


if __name__ == "__main__":
    compile_onnx()
