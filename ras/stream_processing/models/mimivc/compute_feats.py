import argparse
import os
from glob import glob
from collections import deque

import torch
import torchaudio

from .mimi import init_mimi, SAMPLE_RATE, FRAME_SIZE
from ...utils import resolve_file_path


def compute_feats(ls_dir: str) -> str:
    """
    Given the LibriSpeech directory of a speaker, return all the Mimi features
    that can be computed for that speaker, stored in a file.
    """

    mimi = init_mimi()

    target_feats = list()
    for chapter in os.listdir(ls_dir):
        for audiofile in glob(os.path.join(ls_dir, chapter, "*.flac")):
            feats = get_feats(mimi, audiofile)
            target_feats.append(feats.squeeze(0))

    target_feats = torch.vstack(target_feats)
    print(f"Computed {target_feats.shape[0]} features for {ls_dir}")

    # dump the features
    save_dir = resolve_file_path("target_feats/")
    dump_file = os.path.join(save_dir, os.path.basename(ls_dir) + ".pt")
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


def get_feats(mimi, audiofile: str):
    embs = list()
    chunks = get_chunks(audiofile)

    while True:
        if not chunks:
            break

        chunk = chunks.popleft()
        emb = mimi._streaming_state.graphed_encoder(chunk).clone()
        emb = mimi._streaming_state.graphed_tr_enc(emb)[0]
        emb = mimi._to_framerate(emb)
        embs.append(emb.squeeze())

    embs = torch.vstack(embs)
    return embs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ls_dir", type=str)
    args = parser.parse_args()
    compute_feats(args.ls_dir)
