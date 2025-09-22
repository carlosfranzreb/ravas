"""
Run this script to compile Mimi.
TODO
"""

import json
import os
from argparse import ArgumentParser

import torch

from .hifigan import Generator, AttrDict
from .wavlm.model import WavLM, WavLMConfig
from ...utils import resolve_file_path


def compile_onnx(
    input_size: int,
    wavlm_ckpt: str,
    wavlm_layer: int,
    hifigan_cfg: str,
    hifigan_ckpt: str,
):
    os.makedirs(resolve_file_path("onnx/"), exist_ok=True)

    # initialize the WavLM model
    ckpt = torch.load(wavlm_ckpt, map_location="cpu")
    wavlm = WavLM(WavLMConfig(ckpt["cfg"]))
    wavlm.load_state_dict(ckpt["model"])
    wavlm.eval()

    # wrap the `extract_features` method
    wavlm.encoder.layers = wavlm.encoder.layers[:wavlm_layer]
    wavlm.forward = wavlm.extract_features

    # compile the WavLM model
    wavlm_in = torch.randn((1, input_size), dtype=torch.float32)
    torch.onnx.export(
        wavlm,
        wavlm_in,
        resolve_file_path(f"onnx/wavlm_{input_size}.onnx"),
        input_names=["input"],
        output_names=["output"],
    )

    # initialize the HiFiGAN model
    hifigan = Generator(AttrDict(json.load(open(hifigan_cfg))))
    hifigan.load_state_dict(torch.load(hifigan_ckpt, map_location="cpu")["generator"])
    hifigan.eval()
    hifigan.remove_weight_norm()

    # compile the HiFiGAN model
    hifigan_input_size = (input_size // 320) - 1
    hifigan_in = torch.randn((1, hifigan_input_size, 1024), dtype=torch.float32)
    torch.onnx.export(
        hifigan,
        hifigan_in,
        resolve_file_path(f"onnx/hifigan_{input_size}.onnx"),
        input_names=["input"],
        output_names=["output"],
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Compile the WavLM and HiFi-GAN models.")
    parser.add_argument(
        "--input_size", type=int, help="No. of samples fed to the WavLM model."
    )
    parser.add_argument("--wavlm_ckpt", help="Path to the WavLM checkpoint.")
    parser.add_argument(
        "--wavlm_layer", type=int, help="WavLM layer to extract features from."
    )
    parser.add_argument("--hifigan_cfg", help="Path to the HiFi-GAN config.")
    parser.add_argument("--hifigan_ckpt", help="Path to the HiFi-GAN checkpoint.")
    args = parser.parse_args()
    compile_onnx(
        args.input_size,
        args.wavlm_ckpt,
        args.wavlm_layer,
        args.hifigan_cfg,
        args.hifigan_ckpt,
    )
