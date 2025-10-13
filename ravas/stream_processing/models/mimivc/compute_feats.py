import argparse
import os
from glob import glob
from collections import deque

import torch
from torch import Tensor
import torchaudio
from tqdm import tqdm

from .mimi import init_mimi, SAMPLE_RATE, FRAME_SIZE
from ...utils import resolve_file_path


def compute_feats(ls_dir: str) -> str:
    """
    Given the LibriSpeech directory of a speaker, return all the Mimi features
    that can be computed for that speaker, stored in a file.
    """
    target_feats = list()
    for chapter in os.listdir(ls_dir):
        for audiofile in tqdm(glob(os.path.join(ls_dir, chapter, "*.flac"))):
            feats = get_feats(audiofile)
            target_feats.append(feats.squeeze(0))

    target_feats = torch.vstack(target_feats)
    print(f"Computed {target_feats.shape[0]} features for {ls_dir}")

    # dump the features
    save_dir = resolve_file_path("target_feats/mimivc")
    dump_file = os.path.join(save_dir, os.path.split(ls_dir)[-1] + ".pt")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(target_feats, dump_file)

    return dump_file


def get_chunks(audio_path: str):
    in_pcms, sr = torchaudio.load(audio_path)

    if sr != SAMPLE_RATE:
        in_pcms = torchaudio.functional.resample(in_pcms, sr, SAMPLE_RATE)

    batch_size = 1
    in_pcms = in_pcms[None, 0:1].expand(batch_size, -1, -1)
    chunks = deque(
        [
            chunk
            for chunk in in_pcms.split(FRAME_SIZE, dim=2)
            if chunk.shape[-1] == FRAME_SIZE
        ]
    )

    return chunks


def get_feats(audiofile: str) -> Tensor:
    mimi, enc_state, tr_enc_state, downsample_state, _, _, _ = init_mimi()
    embs = list()

    for chunk in get_chunks(audiofile):
        source_feats, *enc_state = mimi.encoder(chunk, *enc_state)
        source_feats, *tr_enc_state = mimi.encoder_transformer(
            source_feats, *tr_enc_state
        )
        source_feats, *downsample_state = mimi.downsample(
            source_feats, *downsample_state
        )
        embs.append(source_feats.squeeze())

    embs = torch.vstack(embs)
    return embs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ls_dir", type=str)
    args = parser.parse_args()
    compute_feats(args.ls_dir)
