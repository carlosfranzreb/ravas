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
    mimi = init_mimi()[0]
    mimi.eval()
    mimi.requires_grad_(False)
    batch_size = 1

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

        if "sample" not in method:
            state = getattr(mimi, method)._init_streaming_state(batch_size)

        if "transformer" in method:
            input_names = ["input", "offsets", "layer_states"]
            output_names = ["output", "new_offsets", "new_layer_states"]
            input_tuple = (x, *state)

            # Prepare inputs for ONNX model, used for testing after compilation
            input_onnx = {"input": x.numpy(), "offsets": state[0].numpy()}
            state_flat = flatten_state(state[1], key_prefix="layer_states_")
            input_onnx.update({key: value.numpy() for key, value in state_flat.items()})

        elif "sample" in method:
            input_names = ["input"]
            output_names = ["output"]
            input_tuple = (x,)

            # Prepare inputs for ONNX model, used for testing after compilation
            input_onnx = {"input": x.numpy()}

        else:  # seanet models
            input_names = ["input", "state"]
            output_names = ["output", "new_state"]
            input_tuple = (x, state)

            # Prepare inputs for ONNX model, used for testing after compilation
            input_onnx = {"input": x.numpy()}
            state_flat = flatten_state(state)
            input_onnx.update({key: value.numpy() for key, value in state_flat.items()})

        torch.onnx.export(
            getattr(mimi, method),
            input_tuple,
            dump_file,
            input_names=input_names,
            output_names=output_names,
            dynamo=True,
        )

        # test that the compiled models output the same values as the torch ones
        model_onnx = ort.InferenceSession(dump_file)
        torch_out = getattr(mimi, method)(*input_tuple)
        torch_out = torch_out[0]
        if isinstance(torch_out, list):
            torch_out = torch_out[0]

        # Run ONNX model
        onnx_out = model_onnx.run(output_names, input_onnx)
        onnx_out = onnx_out[0]
        onnx_out = torch.from_numpy(onnx_out)
        print(method, torch.allclose(torch_out, onnx_out, atol=1e-4))

        # update input for next module
        x = torch_out
        if method == "downsample":
            x = mimi.quantizer.encode(x.unsqueeze(0))
            x = mimi.quantizer.decode(x)
        elif method == "upsample":
            x = x.unsqueeze(0)


def flatten_state(
    state: list, key_prefix: str = "states_", master: bool = True
) -> dict:
    """
    Flatten the state into a dictionary, as expected by the ONNX compiler.

    Input elements that are tensors are called 'states_{idx}', where `idx` refers to the
    element's list index. For input elements that are lists, its inputs are enumerated
    and called 'states_{idx}_{idx2}', where `idx2` is the index of the sub-element in
    its parent list. This is implemented recursively for arbitrary depths.

    The ONNX compiler expects the first element to be called like the original argument
    if it is a tensor. This is the only exception to the recursion explained above.
    """
    out = dict()
    for elem_idx, elem in enumerate(state):
        if isinstance(elem, Tensor):
            out[f"{key_prefix}{elem_idx}"] = elem
        elif isinstance(elem, list):
            out.update(
                flatten_state(elem, key_prefix=f"{key_prefix}{elem_idx}_", master=False)
            )
        else:
            raise ValueError("Invalid element type")

    # fix first key naming
    if master:
        for key in out:
            if key.endswith("_0") and not any(c in "123456789" for c in key):
                first_value = out[key]
                break

        del out[key]

        if key.startswith("layer_states"):
            out["layer_states"] = first_value
        else:  # seanet
            out["state"] = first_value

    return out


if __name__ == "__main__":
    compile_onnx()
