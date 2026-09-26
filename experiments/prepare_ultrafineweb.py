"""Tokenise Ultra-FineWeb (English) for the real-text study.

  * trains a byte-level BPE tokenizer (default 16,384 tokens) on documents
    from the training file,
  * train.npy: documents from ultrafineweb-en part 1 until --train_tokens,
  * val.npy: held-out documents from a different file (part 2),
  * ood.npy: Tiny Shakespeare (a different domain),
  * train_counts.npy: how often each token id occurs in train.npy.

Documents are separated by <|endoftext|>. Arrays are uint16.

    python -m experiments.prepare_ultrafineweb --out /root/ufw/tok
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

REPO = "openbmb/Ultra-FineWeb"
PART = "data/ultrafineweb_en/ultrafineweb-en-part-{:04d}-of-2048.parquet"
SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
EOT = "<|endoftext|>"


def docs(part: int, cache: str):
    path = hf_hub_download(REPO, PART.format(part), repo_type="dataset", local_dir=cache)
    f = pq.ParquetFile(path)
    for batch in f.iter_batches(batch_size=4096, columns=["content"]):
        yield from (t for t in batch.column(0).to_pylist() if t)


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="/root/ufw")
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--tokenizer_docs", type=int, default=100_000)
    ap.add_argument("--train_tokens", type=int, default=100_000_000)
    ap.add_argument("--val_tokens", type=int, default=2_000_000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tok_path = os.path.join(a.out, "tokenizer.json")
    if os.path.exists(tok_path):
        tok = Tokenizer.from_file(tok_path)
    else:
        print("training BPE tokenizer", flush=True)
        it = docs(1, a.cache)
        tok = train_tokenizer((next(it) for _ in range(a.tokenizer_docs)), a.vocab)
        tok.save(tok_path)

    print("train split (part 1)", flush=True)
    train = encode(tok, docs(1, a.cache), a.train_tokens)
    np.save(os.path.join(a.out, "train.npy"), train)
    np.save(os.path.join(a.out, "train_counts.npy"), np.bincount(train, minlength=tok.get_vocab_size()))
    print("held-out split (part 2)", flush=True)
    val = encode(tok, docs(2, a.cache), a.val_tokens)
    np.save(os.path.join(a.out, "val.npy"), val)
    print("out-of-domain split (Tiny Shakespeare)", flush=True)
    text = urllib.request.urlopen(SHAKESPEARE).read().decode()
    ood = np.asarray(tok.encode(text).ids, np.uint16)
    np.save(os.path.join(a.out, "ood.npy"), ood)

    meta = {"vocab_size": tok.get_vocab_size(), "train_tokens": int(len(train)), "val_tokens": int(len(val)),
            "ood_tokens": int(len(ood)), "train_source": PART.format(1), "val_source": PART.format(2),
            "ood_source": SHAKESPEARE,
            "bytes_per_token_ood": len(text.encode()) / len(ood)}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(meta)


if __name__ == "__main__":
    main()
