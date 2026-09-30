"""Tokenise Ultra-FineWeb (English) for the real-text study.

  * trains a byte-level BPE tokenizer (default 16,384 tokens) on documents
    from the training file,
  * train.npy: documents from ultrafineweb-en part 1 (or the files given by
    --train_parts, encoded in parallel processes) until --train_tokens,
  * val.npy: held-out documents from a different file (part 2),
  * ood.npy: Tiny Shakespeare (a different domain),
  * train_counts.npy: how often each token id occurs in train.npy.

Documents are separated by <|endoftext|>. Arrays are uint16. The dataset
is streamed: documents are read over HTTP as they are tokenised and
nothing from Ultra-FineWeb is stored locally except the token arrays.

    python -m experiments.prepare_ultrafineweb --out /root/ufw/tok
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import urllib.request

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PART = "hf://datasets/openbmb/Ultra-FineWeb/data/ultrafineweb_en/ultrafineweb-en-part-{:04d}-of-2048.parquet"
SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
EOT = "<|endoftext|>"


def docs(part: int):
    """Stream the documents of one Ultra-FineWeb file (nothing is downloaded
    to disk)."""
    ds = load_dataset("parquet", data_files=PART.format(part), split="train", streaming=True)
    for row in ds.select_columns(["content"]):
        if row["content"]:
            yield row["content"]


def train_tokenizer(texts, vocab: int) -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def encode(tok: Tokenizer, texts, limit: int, chunk: int = 2048) -> np.ndarray:
    eot = tok.token_to_id(EOT)
    out, n, buf = [], 0, []

    def flush():
        nonlocal n
        for e in tok.encode_batch(buf):
            ids = np.asarray(e.ids + [eot], np.uint16)
            out.append(ids)
            n += len(ids)
        buf.clear()

    for t in texts:
        buf.append(t)
        if len(buf) == chunk:
            flush()
            print(f"  {n / 1e6:.1f}M tokens", flush=True)
            if n >= limit:
                break
    if buf and n < limit:
        flush()
    return np.concatenate(out)[:limit]


def _encode_part(args):
    tok_path, part, limit = args
    arr = encode(Tokenizer.from_file(tok_path), docs(part), limit)
    print(f"part {part}: {len(arr) / 1e6:.1f}M tokens", flush=True)
    return arr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--tokenizer_docs", type=int, default=100_000)
    ap.add_argument("--train_tokens", type=int, default=100_000_000)
    ap.add_argument("--val_tokens", type=int, default=2_000_000)
    ap.add_argument("--train_parts", default="1",
                    help="comma-separated Ultra-FineWeb files for train.npy (part 2 is the held-out split)")
    a = ap.parse_args()
    parts = [int(p) for p in a.train_parts.split(",")]
    if 2 in parts:
        raise ValueError("part 2 is the held-out split")
    os.makedirs(a.out, exist_ok=True)

    tok_path = os.path.join(a.out, "tokenizer.json")
    if os.path.exists(tok_path):
        tok = Tokenizer.from_file(tok_path)
    else:
        print("training BPE tokenizer", flush=True)
        it = docs(1)
        tok = train_tokenizer((next(it) for _ in range(a.tokenizer_docs)), a.vocab)
        tok.save(tok_path)

    print(f"train split (parts {parts})", flush=True)
    if len(parts) == 1:
        train = encode(tok, docs(parts[0]), a.train_tokens)
    else:  # one process per file; each file is encoded up to the whole budget, then trimmed
        with mp.get_context("spawn").Pool(len(parts)) as pool:
            train = np.concatenate(pool.map(_encode_part, [(tok_path, p, a.train_tokens) for p in parts]))
        train = train[:a.train_tokens]
    np.save(os.path.join(a.out, "train.npy"), train)
    np.save(os.path.join(a.out, "train_counts.npy"), np.bincount(train, minlength=tok.get_vocab_size()))
    print("held-out split (part 2)", flush=True)
    val = encode(tok, docs(2), a.val_tokens)
    np.save(os.path.join(a.out, "val.npy"), val)
    print("out-of-domain split (Tiny Shakespeare)", flush=True)
    text = urllib.request.urlopen(SHAKESPEARE).read().decode()
    ood = np.asarray(tok.encode(text).ids, np.uint16)
    np.save(os.path.join(a.out, "ood.npy"), ood)

    meta = {"vocab_size": tok.get_vocab_size(), "train_tokens": int(len(train)), "val_tokens": int(len(val)),
            "ood_tokens": int(len(ood)), "train_source": [PART.format(p) for p in parts], "val_source": PART.format(2),
            "ood_source": SHAKESPEARE,
            "bytes_per_token_ood": len(text.encode()) / len(ood)}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(meta)


if __name__ == "__main__":
    main()
    # the streaming reader can leave non-daemon threads behind, which kept
    # the process alive after everything was written (seen on Colab)
    sys.stdout.flush()
    os._exit(0)
