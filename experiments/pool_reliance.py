"""Does the knowledge live in the pool? Compare training recipes.

Every arm trains the main setup (16,384 facts, 4,096-vector pool, 4,000
steps). "Pool dependence" = accuracy with the pool vs with it removed.

    KT_RESULTS=experiments/results/reliance python -m experiments.pool_reliance train --gpus 0,1 --per_gpu 3
    KT_RESULTS=experiments/results/reliance python -m experiments.pool_reliance analyze
    (add --round 2 to both for the second set of recipes)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import jax
import numpy as np

from . import knowledge_tests as kt

BASE = ["--num_entities", "4096", "--num_relations", "4", "--steps", "4000"]
ARMS = {
    "base": [],
    "kl": ["--nopool_kl_coef", "1.0"],
    "route": ["--route_through_pool", "true"],
    "noffn": ["--memory_ffn", "false"],
    "noffn_kl": ["--memory_ffn", "false", "--nopool_kl_coef", "1.0"],
    "noffn_kl_route": ["--memory_ffn", "false", "--nopool_kl_coef", "1.0", "--route_through_pool", "true"],
}
# Round 2: all without the memory-layer FFN (the round-1 winner).
NOFFN = ["--memory_ffn", "false"]
ARMS2 = {
    "ptrue": NOFFN + ["--nopool_true_coef", "1.0"],
    "warm_ptrue": NOFFN + ["--nopool_true_coef", "1.0", "--nopool_after_step", "1500"],
    "warm_route": NOFFN + ["--route_after_step", "1500"],
    "warm_route_ptrue": NOFFN + ["--route_after_step", "1500", "--nopool_true_coef", "1.0",
                                 "--nopool_after_step", "1500"],
    "freeze_1500": NOFFN + ["--freeze_backbone_after_step", "1500"],
    "freeze_500": NOFFN + ["--freeze_backbone_after_step", "500"],
}


def train(gpus, per_gpu, only):
    os.makedirs(kt.CKPT, exist_ok=True)
    todo = [a for a in ARMS if (not only or a in only)
            and not os.path.exists(os.path.join(kt.CKPT, a + ".msgpack.json"))]
    slots = [g for g in gpus for _ in range(per_gpu)]
    running = {}
    while todo or running:
        free = [s for s in range(len(slots)) if s not in running]
        while todo and free:
            s, arm = free.pop(0), todo.pop(0)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(slots[s]),
                   "XLA_PYTHON_CLIENT_MEM_FRACTION": f"{0.9 / per_gpu:.2f}"}
            cmd = [sys.executable, "-m", "memory_pool_model.train", *BASE, *ARMS[arm],
                   "--log_every", "500", "--eval_every", "1000",
                   "--save", os.path.join(kt.CKPT, arm + ".msgpack")]
            log = open(os.path.join(kt.RESULTS, f"train_{arm}.log"), "w")
            print(f"[start] {arm} on gpu {slots[s]}", flush=True)
            running[s] = (arm, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env))
        time.sleep(5)
        for s, (arm, p) in list(running.items()):
            if p.poll() is not None:
                print(f"[done ] {arm} exit={p.returncode}", flush=True)
                del running[s]


def analyze(out_name="reliance.json"):
    out = []
    for arm in ARMS:
        if not os.path.exists(os.path.join(kt.CKPT, arm + ".msgpack.json")):
            continue
        mcfg, ds, params, meta = kt.load(arm)
        ids = np.arange(ds.num_facts)
        full = kt.probe(mcfg, params, ds, ids)
        zeroed = kt.probe(mcfg, kt.zero_slots(params, np.arange(mcfg.pool_size)), ds, ids)
        half = kt.probe(mcfg, kt.zero_slots(
            params, np.random.default_rng(0).choice(mcfg.pool_size, mcfg.pool_size // 2, replace=False)), ds, ids)
        counts = np.bincount(full["slots"].reshape(-1), minlength=mcfg.pool_size).astype(float)
        p = counts / counts.sum()
        spread = float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))) / mcfg.pool_size)
        row = {
            "arm": arm, "flags": " ".join(ARMS[arm]) or "(defaults)",
            "accuracy": float(full["correct"].mean()),
            "accuracy_pool_zeroed": float(zeroed["correct"].mean()),
            "accuracy_half_pool_deleted": float(half["correct"].mean()),
            "pool_dependence": float(full["correct"].mean() - zeroed["correct"].mean()),
            "pool_active": float(np.mean(p > 0.1 / mcfg.pool_size)),
            "pool_spread": spread,
            "params": int(sum(x.size for x in jax.tree_util.tree_leaves(params))),
            "history": meta["history"],
            "train_seconds": meta.get("train_seconds"),
        }
        print({k: v for k, v in row.items() if k != "history"}, flush=True)
        out.append(row)
    kt._save(out_name, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "analyze"])
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--per_gpu", type=int, default=1)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--round", type=int, default=1)
    a = ap.parse_args()
    if a.round == 2:
        ARMS.clear()
        ARMS.update(ARMS2)
    if a.cmd == "train":
        train(a.gpus.split(","), a.per_gpu, a.only)
    else:
        analyze("reliance.json" if a.round == 1 else f"reliance{a.round}.json")
