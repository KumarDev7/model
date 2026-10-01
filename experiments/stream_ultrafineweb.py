"""Tokenise Ultra-FineWeb into shards while training runs, so training never
repeats data and the disk never has to hold the corpus.

Parts (Ultra-FineWeb English files, 2048 in all, part 2 is the held-out
split) are streamed over HTTP and tokenised in a fixed order into

    <out>/stream.json       parts order, tokenizer, vocab
    <out>/p0001-0000.npy    uint16 tokens, about --shard_tokens each, cut at a document end
    <out>/p0001.progress    shards written and source rows read so far (restart point)
    <out>/p0001.done        the part is finished: its number of shards

memory_pool_model.data.StreamingTokenDataset (train --task stream) reads the
shards in this order and deletes them once a checkpoint is past them. This
script stays at most --max_ready_gb of unread shards ahead (the oldest
unfinished part is always allowed to write, so the reader can't wait on a
full disk). Run it again after a crash and it continues where it stopped:
rows already in shards are skipped, not tokenised again, and the shards come
out identical. On a new VM (empty --out), pass the training checkpoint with
--from_checkpoint so parts the trainer has finished are skipped.

    python -m experiments.stream_ultrafineweb --out /data/stream --tokenizer /data/tok/tokenizer.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

EOT = "<|endoftext|>"
HELD_OUT = 2  # val.npy of prepare_ultrafineweb


def parse_parts(spec: str):
    """"1,3-6" -> [1, 3, 4, 5, 6]"""
    out = []
    for item in spec.split(","):
        lo, _, hi = item.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def hf_rows(part: int):
    """Raw `content` of every row of one Ultra-FineWeb file (empty ones too, so
    row counts are stable for skipping)."""
    from datasets import load_dataset

    from .prepare_ultrafineweb import PART

    ds = load_dataset("parquet", data_files=PART.format(part), split="train", streaming=True)
    for row in ds.select_columns(["content"]):
        yield row["content"]


def _write_atomic(path: str, write) -> None:
    with open(path + ".tmp", "wb") as f:
        write(f)
    os.replace(path + ".tmp", path)


def _write_json(path: str, obj) -> None:
    _write_atomic(path, lambda f: f.write(json.dumps(obj).encode()))


def ready_bytes(out: str) -> int:
    return sum(os.path.getsize(p) for p in glob.glob(os.path.join(out, "p????-????.npy")))


def oldest_unfinished(out: str, parts) -> int | None:
    for p in parts:
        if not os.path.exists(os.path.join(out, f"p{p:04d}.done")):
            return p
    return None


def produce_part(out: str, part: int, tok, rows, parts, shard_tokens: int, max_ready: int,
                 chunk_docs: int = 1024, poll: float = 5.0) -> int:
    """Tokenise one part into shards, continuing from its .progress file.
    rows: iterator of raw row texts. Returns the number of shards."""
    done = os.path.join(out, f"p{part:04d}.done")
    if os.path.exists(done):
        with open(done) as f:
            return json.load(f)["shards"]
    prog_path = os.path.join(out, f"p{part:04d}.progress")
    prog = {"shards": 0, "rows": 0}
    if os.path.exists(prog_path):
        with open(prog_path) as f:
            prog = json.load(f)
    eot = tok.token_to_id(EOT)
    rows = itertools.islice(rows, prog["rows"], None)  # rows already in shards: read, not tokenised
    n_rows, pieces, n_tok, t0 = prog["rows"], [], 0, time.time()

    def flush_shard():
        nonlocal pieces, n_tok
        arr = np.concatenate(pieces)
        while part != oldest_unfinished(out, parts) and ready_bytes(out) + arr.nbytes > max_ready:
            time.sleep(poll)  # far enough ahead of the trainer
        name = os.path.join(out, f"p{part:04d}-{prog['shards']:04d}.npy")
        _write_atomic(name, lambda f: np.save(f, arr))
        prog.update(shards=prog["shards"] + 1, rows=n_rows)
        _write_json(prog_path, prog)
        print(f"part {part}: shard {prog['shards'] - 1} ({len(arr) / 1e6:.1f}M tokens, "
              f"{len(arr) / max(time.time() - t0, 1e-9) / 1e3:.0f}k tok/s)", flush=True)
        pieces, n_tok = [], 0

    while True:
        batch = list(itertools.islice(rows, chunk_docs))
        if not batch:
            break
        n_rows += len(batch)
        texts = [t for t in batch if t]
        for e in tok.encode_batch(texts):
            ids = np.asarray(e.ids + [eot], np.uint16)
            pieces.append(ids)
            n_tok += len(ids)
        if n_tok >= shard_tokens:
            flush_shard()
            t0 = time.time()
    if pieces:
        flush_shard()
    _write_json(done, {"shards": prog["shards"], "rows": n_rows})
    if os.path.exists(prog_path):
        os.remove(prog_path)
    print(f"part {part}: done, {prog['shards']} shards, {n_rows} rows", flush=True)
    return prog["shards"]


def _worker(args):
    out, part, tok_path, parts, shard_tokens, max_ready, retries = args
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tok_path)
    for attempt in range(retries + 1):
        try:
            return part, produce_part(out, part, tok, hf_rows(part), parts, shard_tokens, max_ready)
        except Exception as e:  # network errors: continue from the last shard
            if attempt == retries:
                raise
            wait = min(600, 10 * 2 ** attempt)
            print(f"part {part}: {type(e).__name__}: {e}; retry in {wait}s", flush=True)
            time.sleep(wait)


def _load_tokenizer(path: str):
    """(file bytes, vocab size)"""
    from tokenizers import Tokenizer

    with open(path, "rb") as f:
        raw = f.read()
    return raw, Tokenizer.from_str(raw.decode()).get_vocab_size()


def _run_jobs(jobs, workers: int) -> None:
    with mp.get_context("spawn").Pool(workers) as pool:
        for _ in pool.imap(_worker, jobs):
            pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json of prepare_ultrafineweb (same as val.npy)")
    ap.add_argument("--parts", default="1,3-2047", help="Ultra-FineWeb files in reading order, e.g. 1,3-200")
    ap.add_argument("--shard_tokens", type=int, default=50_000_000)
    ap.add_argument("--max_ready_gb", type=float, default=10.0, help="unread shards kept ahead of training")
    ap.add_argument("--workers", type=int, default=2, help="parts tokenised at the same time")
    ap.add_argument("--retries", type=int, default=8)
    ap.add_argument("--from_checkpoint", help="<save>.state.json of the training run: skip the parts it has finished")
    a = ap.parse_args()
    parts = parse_parts(a.parts)
    if HELD_OUT in parts:
        raise ValueError(f"part {HELD_OUT} is the held-out split")
    os.makedirs(a.out, exist_ok=True)
    tok_bytes, vocab = _load_tokenizer(a.tokenizer)
    meta = {"parts": parts, "vocab_size": vocab,
            "tokenizer_sha256": hashlib.sha256(tok_bytes).hexdigest(), "shard_tokens": a.shard_tokens}
    meta_path = os.path.join(a.out, "stream.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            old = json.load(f)
        if {k: old[k] for k in ("parts", "vocab_size", "tokenizer_sha256")} != \
                {k: meta[k] for k in ("parts", "vocab_size", "tokenizer_sha256")}:
            raise ValueError(f"{meta_path} was written with other parts or another tokenizer")
        meta = old  # keep its shard size: shards already written must not change
    else:
        _write_json(meta_path, meta)
        with open(os.path.join(a.out, "tokenizer.json"), "wb") as f:
            f.write(tok_bytes)
    if a.from_checkpoint:
        with open(a.from_checkpoint) as f:
            st = json.load(f)["data_state"]
        # earliest part the trainer still needs: its current block can start in the part before st["part"]
        reader_part = min([st["part"]] + [parts.index(int(n[1:5])) for n in st["block"]])
        for p in parts[:reader_part]:  # marked done, so the budget rule sees the right oldest part
            done = os.path.join(a.out, f"p{p:04d}.done")
            if not os.path.exists(done):
                _write_json(done, {"shards": 0, "skipped": True})
        print(f"trainer is in part {parts[reader_part]}: {reader_part} parts skipped", flush=True)
    todo = [p for p in parts if not os.path.exists(os.path.join(a.out, f"p{p:04d}.done"))]
    print(f"{len(parts) - len(todo)} of {len(parts)} parts done; {a.workers} workers", flush=True)
    jobs = [(a.out, p, a.tokenizer, parts, meta["shard_tokens"], int(a.max_ready_gb * 1e9), a.retries) for p in todo]
    _run_jobs(jobs, a.workers)
    print("STREAM_DONE", flush=True)


if __name__ == "__main__":
    main()
    # the streaming reader can leave non-daemon threads behind (see prepare_ultrafineweb)
    sys.stdout.flush()
    os._exit(0)
