import argparse
import os
from glob import glob

import torch
import torchaudio

from ..knnvc.wavlm.model import WavLM, WavLMConfig


def compute_feats(ls_dir: str, wavlm_ckpt: str, wavlm_layer: int) -> str:
    """
    Given the LibriSpeech directory of a speaker, return all the WavLm features that
    can be computed for that speaker, stored in a file.
    """

    # initialize the WavLM model
    ckpt = torch.load(wavlm_ckpt, map_location="cpu")
    wavlm = WavLM(WavLMConfig(ckpt["cfg"]))
    wavlm.load_state_dict(ckpt["model"])
    wavlm.eval()

    target_feats = list()
    for chapter in os.listdir(ls_dir):
        for audiofile in glob(os.path.join(ls_dir, chapter, "*.flac")):
            audio = torchaudio.load(audiofile)[0]
            feats = wavlm.extract_features(audio, output_layer=wavlm_layer)[0]
            target_feats.append(feats.squeeze(0))
    target_feats = torch.cat(target_feats, dim=0)
    print(f"Computed {target_feats.shape[0]} features for {ls_dir}")

    # dump the features
    dump_file = os.path.join("target_feats", os.path.basename(ls_dir) + ".pt")
    os.makedirs("target_feats", exist_ok=True)
    torch.save(target_feats, dump_file)

    return dump_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ls_dir", type=str, required=True)
    parser.add_argument("--wavlm_ckpt", type=str, required=True)
    parser.add_argument("--wavlm_layer", type=int, default=6)
    args = parser.parse_args()
    compute_feats(args.ls_dir, args.wavlm_ckpt, args.wavlm_layer)
