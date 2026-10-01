"""Datasets: a synthetic knowledge base, byte-level text, pre-tokenised text,
and a stream of token shards that is read once."""

from __future__ import annotations

import glob
import json
import os
import re
import time
from typing import Dict, Iterator

import numpy as np

Batch = Dict[str, np.ndarray]

PAD, BOS, SEP = 0, 1, 2


class FactDataset:
    """Synthetic knowledge base of (entity, relation) -> attribute facts.

    Entities are multi-token names, so their token embeddings cannot store
    per-entity knowledge: the facts have to live in the network weights or,
    ideally, in the memory pool. A sequence is a list of facts:

        BOS  n1 n2 n3 rel attr SEP  n1 n2 n3 rel attr SEP ...

    Only the attribute tokens are predictable, so only they are scored.
    """

    def __init__(
        self,
        num_entities: int = 4096,
        num_relations: int = 4,
        num_attributes: int = 256,
        name_alphabet: int = 16,
        name_len: int = 3,
        facts_per_seq: int = 10,
        seed: int = 0,
    ):
        if num_entities > name_alphabet**name_len:
            raise ValueError("name_alphabet ** name_len must be >= num_entities")
        rng = np.random.default_rng(seed)
        self.name_len = name_len
        self.facts_per_seq = facts_per_seq
        self.name_offset = 3
        self.rel_offset = self.name_offset + name_alphabet
        self.attr_offset = self.rel_offset + num_relations
        self.vocab_size = self.attr_offset + num_attributes

        codes = rng.choice(name_alphabet**name_len, size=num_entities, replace=False)
        digits = (codes[:, None] // name_alphabet ** np.arange(name_len)[::-1]) % name_alphabet
        self.names = digits + self.name_offset  # [E, name_len]
        self.table = rng.integers(0, num_attributes, size=(num_entities, num_relations))
        self.num_relations = num_relations
        self.fact_len = name_len + 3  # name + rel + attr + SEP
        self.seq_len = 1 + facts_per_seq * self.fact_len

    @property
    def num_facts(self) -> int:
        return self.table.size

    def _encode(self, ent: np.ndarray, rel: np.ndarray) -> Batch:
        B, F = ent.shape
        facts = np.concatenate(
            [
                self.names[ent],  # [B, F, name_len]
                (rel + self.rel_offset)[..., None],
                (self.table[ent, rel] + self.attr_offset)[..., None],
                np.full((B, F, 1), SEP),
            ],
            axis=-1,
        ).reshape(B, -1)
        seq = np.concatenate([np.full((B, 1), BOS), facts], axis=1).astype(np.int32)
        targets = seq[:, 1:]
        mask = (targets >= self.attr_offset).astype(np.float32)
        return {"inputs": seq[:, :-1], "targets": targets, "mask": mask}

    def sample(self, rng: np.random.Generator, batch_size: int) -> Batch:
        shape = (batch_size, self.facts_per_seq)
        ent = rng.integers(0, len(self.names), size=shape)
        rel = rng.integers(0, self.num_relations, size=shape)
        return self._encode(ent, rel)

    def eval_batches(self, batch_size: int) -> Iterator[Batch]:
        """Every fact exactly once (last sequence padded by repeating facts)."""
        ent, rel = np.divmod(np.arange(self.num_facts), self.num_relations)
        per_batch = batch_size * self.facts_per_seq
        for start in range(0, self.num_facts, per_batch):
            e, r = ent[start : start + per_batch], rel[start : start + per_batch]
            valid = len(e)
            pad = (-valid) % self.facts_per_seq
            e, r = np.pad(e, (0, pad), mode="wrap"), np.pad(r, (0, pad), mode="wrap")
            batch = self._encode(e.reshape(-1, self.facts_per_seq), r.reshape(-1, self.facts_per_seq))
            if pad:  # don't double count the wrapped facts
                flat_mask = batch["mask"].reshape(-1)
                attr_pos = np.flatnonzero(flat_mask)
                flat_mask[attr_pos[valid:]] = 0.0
            yield batch


class TextDataset:
    """Byte-level language modelling on an arbitrary text file."""

    def __init__(self, path: str, seq_len: int = 128, eval_fraction: float = 0.05):
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8).astype(np.int32)
        split = int(len(data) * (1 - eval_fraction))
        self.train, self.eval = data[:split], data[split:]
        self.seq_len = seq_len
        self.vocab_size = 256

    def _windows(self, data: np.ndarray, starts: np.ndarray) -> Batch:
        idx = starts[:, None] + np.arange(self.seq_len + 1)
        seq = data[idx]
        return {
            "inputs": seq[:, :-1],
            "targets": seq[:, 1:],
            "mask": np.ones(seq[:, 1:].shape, np.float32),
        }

    def sample(self, rng: np.random.Generator, batch_size: int) -> Batch:
        starts = rng.integers(0, len(self.train) - self.seq_len - 1, size=batch_size)
        return self._windows(self.train, starts)

    def eval_batches(self, batch_size: int, max_batches: int = 20) -> Iterator[Batch]:
        starts = np.arange(0, len(self.eval) - self.seq_len - 1, self.seq_len)
        for i in range(0, min(len(starts), batch_size * max_batches), batch_size):
            yield self._windows(self.eval, starts[i : i + batch_size])


class TokenDataset:
    """Pre-tokenised text: flat .npy token arrays (e.g. uint16 BPE ids), read
    through a memory map so large corpora don't have to fit in RAM.

    Training windows are sampled at random offsets of `train_path`; evaluation
    walks `eval_path` (held-out documents) in consecutive windows."""

    def __init__(self, train_path: str, eval_path: str, vocab_size: int, seq_len: int = 256,
                 eval_windows: int = 640):
        self.train = np.load(train_path, mmap_mode="r")
        self.eval = np.load(eval_path, mmap_mode="r")
        self.seq_len, self.vocab_size, self.eval_windows = seq_len, vocab_size, eval_windows

    @staticmethod
    def windows(data: np.ndarray, starts: np.ndarray, seq_len: int) -> Batch:
        seq = np.stack([np.asarray(data[s : s + seq_len + 1]) for s in starts]).astype(np.int32)
        return {"inputs": seq[:, :-1], "targets": seq[:, 1:],
                "mask": np.ones(seq[:, 1:].shape, np.float32)}

    def sample(self, rng: np.random.Generator, batch_size: int) -> Batch:
        starts = rng.integers(0, len(self.train) - self.seq_len - 1, size=batch_size)
        return self.windows(self.train, starts, self.seq_len)

    def eval_batches(self, batch_size: int) -> Iterator[Batch]:
        starts = np.arange(0, len(self.eval) - self.seq_len - 1, self.seq_len)[: self.eval_windows]
        for i in range(0, len(starts), batch_size):
            yield self.windows(self.eval, starts[i : i + batch_size], self.seq_len)


class StreamingTokenDataset:
    """Token shards written by experiments.stream_ultrafineweb, each token
    trained on once.

    Shards are read in the producer's fixed order (stream.json parts, then
    shard number), `block_shards` at a time. The windows of a block
    (seq_len + 1 tokens, consecutive windows share one token, so every token
    is a target once) are visited in a random order seeded by the block
    number. The position (block shards + offset) goes into the training
    checkpoint, so a resumed run continues exactly where it stopped.

    Shards before the current block are deleted after each checkpoint
    (immediately when training saves no checkpoint), so disk holds the
    current block plus what the producer has tokenised ahead. If the next
    shard isn't written yet, sample() waits for it (time in wait_seconds)."""

    _NAME = re.compile(r"p(\d{4})-(\d{4})\.npy$")

    def __init__(self, stream_dir: str, eval_path: str, vocab_size: int, seq_len: int = 256,
                 eval_windows: int = 640, block_shards: int = 4, seed: int = 0, keep_consumed: bool = False,
                 poll_seconds: float = 5.0):
        with open(os.path.join(stream_dir, "stream.json")) as f:
            meta = json.load(f)
        if meta["vocab_size"] != vocab_size:
            raise ValueError(f"stream vocab {meta['vocab_size']} != model vocab {vocab_size}")
        self.dir, self.parts = stream_dir, list(meta["parts"])
        self.eval = np.load(eval_path, mmap_mode="r")
        self.seq_len, self.vocab_size, self.eval_windows = seq_len, vocab_size, eval_windows
        self.block_shards, self.seed, self.keep, self.poll = block_shards, seed, keep_consumed, poll_seconds
        self.defer_delete = False  # set by train.run when it checkpoints
        self.wait_seconds = 0.0
        # next shard to fetch (index into parts, shard number), current block, offset in it
        self._st = {"part": 0, "shard": 0, "block_id": -1, "block": [], "pos": 0, "tokens": 0}
        self._open = None

    # ---------------------------------------------------------------- state
    def state_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, list) else v) for k, v in self._st.items()}

    def load_state_dict(self, st: dict) -> None:
        self._st = {k: (list(v) if isinstance(v, list) else v) for k, v in st.items()}
        self._open = None

    def progress(self) -> str:
        cur = self._st["block"][-1][:-4] if self._st["block"] else "-"
        return f"data {cur} tokens={self._st['tokens'] / 1e9:.3f}B wait={self.wait_seconds:.0f}s"

    # --------------------------------------------------------------- shards
    def _path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def _next_shard(self) -> str | None:
        """Name of the next shard in stream order, waiting until it's written;
        None when every part is done and read."""
        st, waited = self._st, 0.0
        while st["part"] < len(self.parts):
            part = self.parts[st["part"]]
            name = f"p{part:04d}-{st['shard']:04d}.npy"
            if os.path.exists(self._path(name)):
                st["shard"] += 1
                return name
            done = self._path(f"p{part:04d}.done")
            if os.path.exists(done):
                with open(done) as f:
                    n = json.load(f)["shards"]
                if st["shard"] < n:
                    raise FileNotFoundError(f"{name} is missing (deleted?) but part {part} has {n} shards")
                st["part"], st["shard"] = st["part"] + 1, 0
                continue
            if waited == 0.0 or waited % 60 < self.poll:
                print(f"waiting for {name} from the stream producer ({waited:.0f}s)", flush=True)
            time.sleep(self.poll)
            waited += self.poll
            self.wait_seconds += self.poll
        return None

    def _order(self, name: str):
        m = self._NAME.search(name)
        part, k = int(m.group(1)), int(m.group(2))
        return (self.parts.index(part) if part in self.parts else -1, k)

    def _delete_before_block(self) -> None:
        if self.keep or not self._st["block"]:
            return
        first = self._order(self._st["block"][0])
        for path in glob.glob(self._path("p????-????.npy")):
            if self._order(path) < first:
                os.remove(path)

    def checkpoint_saved(self) -> None:
        """The checkpoint is past every block before the current one."""
        self._delete_before_block()

    def _open_block(self) -> None:
        names = self._st["block"]
        maps = [np.load(self._path(n), mmap_mode="r") for n in names]
        starts = [np.arange(0, len(m) - self.seq_len, self.seq_len) for m in maps]
        file_of = np.concatenate([np.full(len(s), i, np.int32) for i, s in enumerate(starts)])
        start = np.concatenate(starts)
        order = np.random.default_rng([self.seed, self._st["block_id"]]).permutation(len(start))
        self._open = (maps, file_of[order], start[order])

    def _advance_block(self) -> None:
        names = []
        while len(names) < self.block_shards:
            name = self._next_shard()
            if name is None:
                break
            names.append(name)
        if not names:
            raise RuntimeError("the token stream is exhausted (every part has been read)")
        self._st.update(block=names, block_id=self._st["block_id"] + 1, pos=0)
        if not self.defer_delete:
            self._delete_before_block()
        self._open_block()

    # -------------------------------------------------------------- batches
    def sample(self, rng: np.random.Generator, batch_size: int) -> Batch:
        """Next batch_size windows of the stream (rng is unused: the order
        is fixed by the block seed, so it survives a resume)."""
        rows = []
        while len(rows) < batch_size:
            if self._open is None and self._st["block"]:
                self._open_block()
            if self._open is None or self._st["pos"] >= len(self._open[1]):
                self._advance_block()
            maps, file_of, start = self._open
            pos = self._st["pos"]
            take = min(batch_size - len(rows), len(start) - pos)
            for f, s in zip(file_of[pos: pos + take], start[pos: pos + take]):
                rows.append(np.asarray(maps[f][s: s + self.seq_len + 1]))
            self._st["pos"] = pos + take
        self._st["tokens"] += batch_size * self.seq_len
        seq = np.stack(rows).astype(np.int32)
        return {"inputs": seq[:, :-1], "targets": seq[:, 1:], "mask": np.ones(seq[:, 1:].shape, np.float32)}

    def eval_batches(self, batch_size: int) -> Iterator[Batch]:
        starts = np.arange(0, len(self.eval) - self.seq_len - 1, self.seq_len)[: self.eval_windows]
        for i in range(0, len(starts), batch_size):
            yield TokenDataset.windows(self.eval, starts[i : i + batch_size], self.seq_len)
