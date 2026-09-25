"""Generalisation tests: does the model do more than recall what it was trained on?

The knowledge base mixes rules and random facts:
  * relation 0 follows a rule: the attribute is fixed by the first two name
    tokens (shared by ~8 entities). 10% of entities are exceptions with a
    random attribute.
  * relations 1-3 are random per entity (pure memorisation).
  * every relation has two surface tokens: canonical `r` and a paraphrase `r'`.

Training sees 80% of entities. Only half of those ever appear with the
paraphrase tokens. Tests:
  memorised      trained entities, random relations, canonical token
  paraphrase     trained entities never shown with r', asked with r'
  rule_unseen    held-out entities, rule relation -> can it apply the rule?
  exceptions     trained exception entities, rule relation -> overrides kept?
  unknowable     held-out entities, random relations -> must be chance;
                 we report how confident the model is anyway (hallucination)

    python -m experiments.generalization
"""

from __future__ import annotations

import json
import os
from typing import Dict

import jax
import numpy as np

from memory_pool_model.config import ModelConfig, TrainConfig
from memory_pool_model.data import BOS, SEP
from memory_pool_model.train import Trainer

from .knowledge_tests import RESULTS, _forward, _save, _train


class RuleFacts:
    def __init__(self, entities=2048, relations=4, attributes=256, alphabet=16, name_len=3,
                 exception_rate=0.1, train_frac=0.8, para_frac=0.5, facts_per_seq=10, seed=0):
        rng = np.random.default_rng(seed)
        self.R, self.facts_per_seq, self.name_len = relations, facts_per_seq, name_len
        self.name_off = 3
        self.rel_off = self.name_off + alphabet
        self.para_off = self.rel_off + relations
        self.attr_off = self.para_off + relations
        self.vocab_size = self.attr_off + attributes
        self.fact_len = name_len + 3
        self.seq_len = 1 + facts_per_seq * self.fact_len

        codes = rng.choice(alphabet**name_len, size=entities, replace=False)
        digits = (codes[:, None] // alphabet ** np.arange(name_len)[::-1]) % alphabet
        self.names = digits + self.name_off
        self.table = rng.integers(0, attributes, size=(entities, relations))
        rule_map = rng.permutation(attributes)[: alphabet * alphabet]  # (n1, n2) -> attribute
        self.rule_attr = rule_map[digits[:, 0] * alphabet + digits[:, 1]]
        self.exception = rng.random(entities) < exception_rate
        self.table[:, 0] = np.where(self.exception, self.table[:, 0], self.rule_attr)
        # guarantee exceptions actually differ from the rule
        clash = self.exception & (self.table[:, 0] == self.rule_attr)
        self.table[clash, 0] = (self.rule_attr[clash] + 1) % attributes

        perm = rng.permutation(entities)
        n_train = int(train_frac * entities)
        self.train_ents = np.sort(perm[:n_train])
        self.test_ents = np.sort(perm[n_train:])
        self.para_ents = np.sort(rng.choice(self.train_ents, int(para_frac * n_train), replace=False))
        self.no_para_ents = np.setdiff1d(self.train_ents, self.para_ents)

    def encode(self, ent, rel, para):
        """ent, rel, para: [B, F] arrays."""
        B, F = ent.shape
        rel_tok = np.where(para, rel + self.para_off, rel + self.rel_off)
        facts = np.concatenate([
            self.names[ent], rel_tok[..., None],
            (self.table[ent, rel] + self.attr_off)[..., None], np.full((B, F, 1), SEP),
        ], axis=-1).reshape(B, -1)
        seq = np.concatenate([np.full((B, 1), BOS), facts], 1).astype(np.int32)
        targets = seq[:, 1:]
        return {"inputs": seq[:, :-1], "targets": targets, "mask": (targets >= self.attr_off).astype(np.float32)}

    def sample(self, rng, batch_size):
        shape = (batch_size, self.facts_per_seq)
        ent = rng.choice(self.train_ents, shape)
        rel = rng.integers(0, self.R, shape)
        para = np.isin(ent, self.para_ents) & (rng.random(shape) < 0.5)
        return self.encode(ent, rel, para)


def score(mcfg, params, ds: RuleFacts, ent, rel, para, rows_per_batch=256) -> Dict[str, float]:
    fwd = _forward(mcfg)
    F = ds.facts_per_seq
    n = len(ent)
    pad = (-n) % F
    e, r, p = (np.pad(a, (0, pad), mode="wrap").reshape(-1, F) for a in (ent, rel, para))
    correct, conf = [], []
    for s in range(0, len(e), rows_per_batch):
        b = ds.encode(e[s : s + rows_per_batch], r[s : s + rows_per_batch], p[s : s + rows_per_batch])
        rows_n = len(b["inputs"])
        n_rows = max(8, 1 << (rows_n - 1).bit_length())
        logits = np.asarray(fwd(params, np.pad(b["inputs"], ((0, n_rows - rows_n), (0, 0))))[0])[:rows_n]
        pos = 1 + np.arange(F) * ds.fact_len + ds.name_len
        lg = logits[:, pos]  # [rows, F, V]
        tgt = b["targets"][:, pos]
        probs = np.asarray(jax.nn.softmax(lg[..., ds.attr_off :], -1))
        correct.append((lg.argmax(-1) == tgt).reshape(-1))
        conf.append(probs.max(-1).reshape(-1))
    correct, conf = np.concatenate(correct)[:n], np.concatenate(conf)[:n]
    return {"accuracy": float(correct.mean()), "confidence": float(conf.mean()),
            "confident_wrong": float(np.mean(~correct & (conf > 0.5))), "n": int(n)}


def splits(ds: RuleFacts) -> Dict[str, tuple]:
    def grid(ents, rels, para):
        e, r = np.meshgrid(ents, rels, indexing="ij")
        return e.reshape(-1), r.reshape(-1), np.full(e.size, para)

    rand = np.arange(1, ds.R)
    rule_test = ds.test_ents[~ds.exception[ds.test_ents]]
    exc_train = ds.train_ents[ds.exception[ds.train_ents]]
    return {
        "memorised": grid(ds.train_ents, rand, False),
        "paraphrase_seen": grid(ds.para_ents, rand, True),
        "paraphrase_transfer": grid(ds.no_para_ents, rand, True),
        "rule_unseen_entities": grid(rule_test, [0], False),
        "exceptions_memorised": grid(exc_train, [0], False),
        "unknowable": grid(ds.test_ents, rand, False),
    }


def run(steps: int = 3000, every: int = 500) -> Dict:
    ds = RuleFacts()
    sp = splits(ds)
    out = {"entities_train": len(ds.train_ents), "entities_test": len(ds.test_ents),
           "exception_rate": float(ds.exception.mean()), "steps": steps, "models": {}}
    base = dict(vocab_size=ds.vocab_size, max_len=ds.seq_len)
    for kind, mcfg in {"memory": ModelConfig(**base),
                       "dense_matched": ModelConfig(**base, use_memory=False, ffn_mult=9)}.items():
        trainer = Trainer(mcfg, TrainConfig(steps=steps))
        state = trainer.init(jax.random.PRNGKey(0))
        curve = []
        evals = {k: (lambda p, v=v: score(mcfg, p, ds, *v)["accuracy"]) for k, v in sp.items()}
        print(f"[generalization] {kind}", flush=True)
        state = _train(trainer, state, ds.sample, steps, 0, evals, every, True, curve)
        final = {k: score(mcfg, state.params, ds, *v) for k, v in sp.items()}
        out["models"][kind] = {"final": final, "curve": curve}
        print(kind, json.dumps(final, indent=1), flush=True)
    return out


if __name__ == "__main__":
    _save("generalization.json", run())
