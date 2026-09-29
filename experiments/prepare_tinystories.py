"""Tokenise TinyStories (Eldan & Li, 2023) for small-model fluency tests.

TinyStories is ~2.7M short stories written by GPT-3.5/4 with a vocabulary a
young child understands. Models of 1-33M parameters trained on it write
fluent English, so it separates "can this architecture learn to write" from
"does it have enough data and parameters for the open web".

  * trains a byte-level BPE tokenizer (default 4,096 tokens) on training stories,
  * train.npy / val.npy: TinyStoriesV2-GPT4 train / valid, stories separated
    by <|endoftext|>, uint16,
  * train_counts.npy: how often each token id occurs in train.npy.

    python -m experiments.prepare_tinystories --out /content/ts/tok
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from .prepare_ultrafineweb import EOT, encode, train_tokenizer

REPO = "roneneldan/TinyStories"
FILES = {"train": "TinyStoriesV2-GPT4-train.txt", "val": "TinyStoriesV2-GPT4-valid.txt"}


def stories(split: str):
    path = hf_hub_download(REPO, FILES[split], repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for s in text.split(EOT):
        s = s.strip()
        if s:
            yield s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--tokenizer_stories", type=int, default=200_000)
    ap.add_argument("--train_tokens", type=int, default=10**10)  # default: all of it
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tok_path = os.path.join(a.out, "tokenizer.json")
    if os.path.exists(tok_path):
        tok = Tokenizer.from_file(tok_path)
    else:
        print("training BPE tokenizer", flush=True)
        it = stories("train")
        tok = train_tokenizer((next(it) for _ in range(a.tokenizer_stories)), a.vocab)
        tok.save(tok_path)

    print("train split", flush=True)
    train = encode(tok, stories("train"), a.train_tokens, chunk=16384)
    np.save(os.path.join(a.out, "train.npy"), train)
    np.save(os.path.join(a.out, "train_counts.npy"), np.bincount(train, minlength=tok.get_vocab_size()))
    print("validation split", flush=True)
    val = encode(tok, stories("val"), 10**10, chunk=16384)
    np.save(os.path.join(a.out, "val.npy"), val)

    meta = {"vocab_size": tok.get_vocab_size(), "train_tokens": int(len(train)), "val_tokens": int(len(val)),
            "source": f"hf://datasets/{REPO}: {FILES['train']}, {FILES['val']}"}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(meta)


if __name__ == "__main__":
    main()
