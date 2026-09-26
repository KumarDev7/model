"""Real text: does the pool collapse, and does the model generalise?

Data: Ultra-FineWeb English tokenised with a 16k BPE (prepare_ultrafineweb.py).
Every arm has the same backbone (d_model 256, 4 layers, FFN x4) and the
same token budget: 6,000 steps x 32 x 256 tokens = 49M tokens, about 1.6
passes over a 30M-token training set, so memorisation shows up as a
train/held-out gap.

  dense       backbone alone
  pool        + 262k-vector pool read in layers 1 and 3 (defaults: pool read
              replaces the FFN there, no-pool penalty, temperature fix)
  pool_ffn    pool added next to the FFN (memory_ffn=True)
  pool_nopen  pool without the no-pool penalty
  pool_old    pool with the old temperature settings (collapse check)

    python -m experiments.text_study train --data /root/ufw/tok30 --out experiments/results/text --gpus 0,1 --per_gpu 2
    python -m experiments.text_study analyze --data /root/ufw/tok30 --out experiments/results/text

The analysis measures, for held-out documents (same distribution), training
windows, and Tiny Shakespeare (another domain): loss / perplexity with the
pool and with it removed; loss by how often the target token occurs in the
training set; and pool usage per memory layer under inference routing
(coverage, active share, evenness, heaviest slot, sub-keys used, top-1
weight), plus how much held-out and out-of-domain text share pool vectors.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from memory_pool_model.config import ModelConfig
from memory_pool_model.data import TokenDataset
from memory_pool_model.model import MemoryPoolLM

SEQ = 256
BACKBONE = ["--d_model", "256", "--n_layers", "4", "--n_heads", "8", "--ffn_mult", "4",
            "--max_len", str(SEQ), "--batch_size", "32", "--lr", "1e-3", "--warmup_steps", "500"]
POOL = ["--memory_layers", "1,3", "--n_sub_keys", "512", "--d_key", "128", "--d_value", "256"]
ARMS = {
    "pool": POOL,
    "pool_ffn": POOL + ["--memory_ffn", "true"],
    "pool_nopen": POOL + ["--nopool_true_coef", "0.0"],
    "pool_old": POOL + ["--balance_temperature_grad", "true", "--min_temperature", "1.0"],
    "dense": ["--use_memory", "false"],
}
FREQ_BUCKETS = [(0, 10), (10, 100), (100, 1_000), (1_000, 10_000), (10_000, 100_000), (100_000, 10**12)]


def train(data, out, gpus, per_gpu, steps, only):
    ckpt = os.path.join(out, "ckpt")
    os.makedirs(ckpt, exist_ok=True)
    vocab = json.load(open(os.path.join(data, "meta.json")))["vocab_size"]
    todo = [a for a in ARMS if (not only or a in only)
            and not os.path.exists(os.path.join(ckpt, a + ".msgpack.json"))]
    slots = [g for g in gpus for _ in range(per_gpu)]
    running = {}
    while todo or running:
        free = [s for s in range(len(slots)) if s not in running]
        while todo and free:
            s, arm = free.pop(0), todo.pop(0)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(slots[s]),
                   "XLA_PYTHON_CLIENT_MEM_FRACTION": f"{0.9 / per_gpu:.2f}"}
            cmd = [sys.executable, "-m", "memory_pool_model.train", "--task", "tokens",
                   "--train_tokens", os.path.join(data, "train.npy"),
                   "--eval_tokens", os.path.join(data, "val.npy"), "--vocab_size", str(vocab),
                   "--eval_windows", "256", *BACKBONE, *ARMS[arm], "--steps", str(steps),
                   "--log_every", "250", "--eval_every", "1000", "--checkpoint_every", "1000",
                   "--save", os.path.join(ckpt, arm + ".msgpack"), "--resume"]
            log = open(os.path.join(out, f"train_{arm}.log"), "a")
            print(f"[start] {arm} on gpu {slots[s]}", flush=True)
            running[s] = (arm, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env))
        time.sleep(10)
        for s, (arm, p) in list(running.items()):
            if p.poll() is not None:
                print(f"[done ] {arm} exit={p.returncode}", flush=True)
                del running[s]


# ----------------------------------------------------------------- analysis
def load(out, arm):
    path = os.path.join(out, "ckpt", arm + ".msgpack")
    meta = json.load(open(path + ".json"))
    m = dict(meta["model"])
    m["memory_layers"] = tuple(m["memory_layers"])
    mcfg = ModelConfig(**{**m, "pool_location": "device", "host_pool": ""})
    template = MemoryPoolLM(mcfg).init(jax.random.PRNGKey(0), jnp.zeros((1, 8), jnp.int32))["params"]
    params = serialization.from_bytes(template, open(path, "rb").read())
    return mcfg, jax.tree_util.tree_map(jnp.asarray, params), meta


def make_forward(mcfg):
    model = MemoryPoolLM(mcfg)

    @jax.jit
    def fwd(params, inputs, targets):
        logits, aux = model.apply({"params": params}, inputs)
        ce = -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[..., None], -1)[..., 0]
        acc = logits.argmax(-1) == targets
        return ce, acc, aux.get("slots", []), aux.get("weights", [])

    @jax.jit
    def fwd_off(params, inputs, targets):
        logits, _ = model.apply({"params": params}, inputs, pool_off=True)
        return -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[..., None], -1)[..., 0]

    return fwd, fwd_off


def windows(data, n, rng=None, batch=32):
    if rng is None:  # consecutive windows from the start
        starts = np.arange(0, len(data) - SEQ - 1, SEQ)[:n]
    else:
        starts = rng.integers(0, len(data) - SEQ - 1, size=n)
    for i in range(0, len(starts), batch):
        b = TokenDataset.windows(data, starts[i : i + batch], SEQ)
        if len(b["inputs"]) == batch:
            yield b


def usage(counts, n_sub, top1):
    p = counts / max(counts.sum(), 1.0)
    nz = p[p > 0]
    rows = np.nonzero(counts)[0]
    a, b = rows // n_sub, rows % n_sub
    return {"coverage": float(np.mean(counts > 0)),
            "active": float(np.mean(p > 0.1 / len(p))),
            "spread": float(np.exp(-np.sum(nz * np.log(nz))) / len(p)),
            "max_share_x_fair": float(p.max() * len(p)),
            "subkeys_used": float((len(np.unique(a)) + len(np.unique(b))) / (2 * n_sub)),
            "top1_weight": float(top1)}


def measure(mcfg, params, fwd, fwd_off, split, counts_train=None):
    L = len(mcfg.memory_layers) if mcfg.use_memory else 0
    ce_all, acc_all, off_all, tgt_all = [], [], [], []
    slot_counts = [np.zeros(mcfg.pool_size) for _ in range(L)]
    top1 = [[] for _ in range(L)]
    for b in split:
        ce, acc, slots, weights = fwd(params, b["inputs"], b["targets"])
        ce_all.append(np.asarray(ce).ravel())
        acc_all.append(np.asarray(acc).ravel())
        tgt_all.append(b["targets"].ravel())
        if L:
            off_all.append(np.asarray(fwd_off(params, b["inputs"], b["targets"])).ravel())
            for i in range(L):
                s = np.asarray(slots[i]).ravel()
                slot_counts[i] += np.bincount(s, minlength=mcfg.pool_size)
                top1[i].append(float(np.asarray(weights[i]).max(-1).mean()))
    ce, acc, tgt = map(np.concatenate, (ce_all, acc_all, tgt_all))
    r = {"tokens": int(len(ce)), "loss": float(ce.mean()), "ppl": float(np.exp(ce.mean())),
         "acc": float(acc.mean())}
    if L:
        off = np.concatenate(off_all)
        r["loss_pool_removed"] = float(off.mean())
        r["ppl_pool_removed"] = float(np.exp(off.mean()))
        r["usage"] = [usage(slot_counts[i], mcfg.n_sub_keys, np.mean(top1[i])) for i in range(L)]
    if counts_train is not None:
        f = counts_train[tgt]
        r["loss_by_train_freq"] = []
        for lo, hi in FREQ_BUCKETS:
            m = (f >= lo) & (f < hi)
            if m.sum():
                row = {"freq": f"{lo}-{hi}", "tokens": int(m.sum()), "loss": float(ce[m].mean())}
                if L:
                    row["loss_pool_removed"] = float(off[m].mean())
                r["loss_by_train_freq"].append(row)
    return r, slot_counts


def analyze(data, out, n_windows):
    train_arr = np.load(os.path.join(data, "train.npy"), mmap_mode="r")
    val_arr = np.load(os.path.join(data, "val.npy"), mmap_mode="r")
    ood_arr = np.load(os.path.join(data, "ood.npy"), mmap_mode="r")
    counts_train = np.load(os.path.join(data, "train_counts.npy"))
    results = []
    for arm in ARMS:
        if not os.path.exists(os.path.join(out, "ckpt", arm + ".msgpack.json")):
            continue
        mcfg, params, meta = load(out, arm)
        fwd, fwd_off = make_forward(mcfg)
        n_params = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
        n_pool = int(params["pool"]["values"].size) if mcfg.use_memory else 0
        row = {"arm": arm, "flags": " ".join(ARMS[arm]), "params_total": n_params,
               "params_backbone": n_params - n_pool - (
                   int(params["pool"]["sub_keys"].size) if mcfg.use_memory else 0),
               "pool_vectors": mcfg.pool_size if mcfg.use_memory else 0,
               "vectors_read_per_token": (mcfg.pool_heads * mcfg.top_k * len(mcfg.memory_layers)
                                          if mcfg.use_memory else 0),
               "train_seconds": meta.get("train_seconds"), "history": meta["history"]}
        if mcfg.use_memory:
            lt = float(params["pool"]["log_temperature"])
            row["temperature"] = float(np.exp(np.clip(lt, np.log(mcfg.min_temperature), np.log(100.0))))
        splits = {
            "heldout": windows(val_arr, n_windows),
            "train": windows(train_arr, n_windows, np.random.default_rng(0)),
            "ood_shakespeare": windows(ood_arr, n_windows),
        }
        per_split_counts = {}
        for name, split in splits.items():
            r, sc = measure(mcfg, params, fwd, fwd_off, split, counts_train if name == "heldout" else None)
            row[name] = r
            per_split_counts[name] = sc
        row["generalization_gap"] = row["heldout"]["loss"] - row["train"]["loss"]
        if mcfg.use_memory:
            row["shared_vectors_heldout_vs_ood"] = []
            for i in range(len(mcfg.memory_layers)):
                a, b = per_split_counts["heldout"][i], per_split_counts["ood_shakespeare"][i]
                row["shared_vectors_heldout_vs_ood"].append({
                    "cosine_of_usage": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)),
                    "ood_reads_on_vectors_unused_by_heldout": float(b[a == 0].sum() / max(b.sum(), 1))})
        print(json.dumps({k: v for k, v in row.items() if k != "history"}, default=float)[:1500], flush=True)
        results.append(row)
    path = os.path.join(out, "text_study.json")
    json.dump(results, open(path, "w"), indent=1, default=float)
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "analyze"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--per_gpu", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--windows", type=int, default=1024, help="windows of 256 tokens per split in analyze")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.data, a.out, a.gpus.split(","), a.per_gpu, a.steps, a.only)
    else:
        analyze(a.data, a.out, a.windows)
