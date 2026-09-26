"""Datasets: a synthetic knowledge base, byte-level text, and pre-tokenised text."""

from __future__ import annotations

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
