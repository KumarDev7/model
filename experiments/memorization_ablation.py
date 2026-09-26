"""Why the pool model memorises 99.8% and not 100%: ablate the suspected causes.

Every arm trains the fact task (16,384 facts, 4,096-vector pool) with the
default recipe and changes one thing:

  anneal     routing noise fades out over 60-90% of training (clean last 10%)
  revive50   dead-key revival stops at 50% of training instead of 80%
  both       anneal + revive50
  both_lr    both + pool learning rate x2 (pool_lr_mult 6)
  long       default recipe, 6,000 steps
  both_long  both, 6,000 steps

Collapse is checked on every arm: pool coverage / active / spread measured
on all 16,384 facts with clean (inference) routing, plus how many sub-keys
are ever picked.

    KT_RESULTS=experiments/results/memorization python -m experiments.memorization_ablation train --gpus 0,1 --per_gpu 3
    KT_RESULTS=experiments/results/memorization python -m experiments.memorization_ablation analyze
    (--gpus cpu runs on the CPU; --round 2: temperature fixes; --round 3: fix + anneal / longer)
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

BASE = ["--num_entities", "4096", "--num_relations", "4"]
ANNEAL = ["--noise_anneal_start", "0.6", "--noise_anneal_end", "0.9"]
REVIVE50 = ["--revive_until", "0.5"]
# Round 1 ran before the temperature fix; OLD reproduces its settings.
OLD = ["--balance_temperature_grad", "true", "--min_temperature", "1.0"]
ARMS = {k: v + OLD for k, v in {
    "base": ["--steps", "4000"],
    "anneal": ["--steps", "4000", *ANNEAL],
    "revive50": ["--steps", "4000", *REVIVE50],
    "both": ["--steps", "4000", *ANNEAL, *REVIVE50],
    "both_lr": ["--steps", "4000", *ANNEAL, *REVIVE50, "--pool_lr_mult", "6.0"],
    "long": ["--steps", "6000"],
    "both_long": ["--steps", "6000", *ANNEAL, *REVIVE50],
}.items()}
# Round 2. Round 1 showed the cause: in 6 of 7 runs the learnable routing
# temperature fell to its floor (1.0) and stayed there; routing then
# concentrated and learning slowed. Two seeds per arm (runs also differ
# from GPU nondeterminism, so one run per arm is not enough).
SGT = ["--balance_temperature_grad", "false", "--min_temperature", "1.0"]
FLOOR = ["--balance_temperature_grad", "true", "--min_temperature", "10.0"]
ARMS2 = {}
for name, flags, seeds in (("base", OLD, (1, 2)), ("sgt", SGT, (0, 1)), ("floor10", FLOOR, (0, 1)),
                           ("fix", ["--balance_temperature_grad", "false", "--min_temperature", "10.0"], (0, 1))):
    for seed in seeds:
        ARMS2[f"{name}_s{seed}"] = ["--steps", "4000", "--seed", str(seed), *flags]
# Round 3: the fix (now the default) + noise fade-out, and + longer training.
ARMS3 = {}
for seed in (0, 1):
    ARMS3[f"fix_anneal_s{seed}"] = ["--steps", "4000", "--seed", str(seed), *ANNEAL]
    ARMS3[f"fix_long_s{seed}"] = ["--steps", "6000", "--seed", str(seed)]


def _done(arm):
    return os.path.exists(os.path.join(kt.CKPT, arm + ".msgpack.json"))


def train(gpus, per_gpu, only):
    os.makedirs(kt.CKPT, exist_ok=True)
    todo = [a for a in ARMS if (not only or a in only) and not _done(a)]
    slots = [g for g in gpus for _ in range(per_gpu)]
    running = {}
    while todo or running:
        free = [s for s in range(len(slots)) if s not in running]
        while todo and free:
            s, arm = free.pop(0), todo.pop(0)
            env = dict(os.environ)
            if slots[s] == "cpu":
                env["JAX_PLATFORMS"] = "cpu"
            else:
                env.update(CUDA_VISIBLE_DEVICES=str(slots[s]),
                           XLA_PYTHON_CLIENT_MEM_FRACTION=f"{0.9 / per_gpu:.2f}")
            save = os.path.join(kt.CKPT, arm + ".msgpack")
            # --resume: a restarted container continues each arm from its last checkpoint
            cmd = [sys.executable, "-m", "memory_pool_model.train", *BASE, *ARMS[arm],
                   "--log_every", "500", "--eval_every", "500", "--checkpoint_every", "500",
                   "--save", save, "--resume"]
            log = open(os.path.join(kt.RESULTS, f"train_{arm}.log"), "a")
            print(f"[start] {arm} on {slots[s]}", flush=True)
            running[s] = (arm, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env))
        time.sleep(5)
        for s, (arm, p) in list(running.items()):
            if p.poll() is not None:
                print(f"[done ] {arm} exit={p.returncode}", flush=True)
                del running[s]


def analyze(out_name="memorization.json"):
    out = []
    for arm in ARMS:
        if not _done(arm):
            continue
        mcfg, ds, params, meta = kt.load(arm)
        ids = np.arange(ds.num_facts)
        full = kt.probe(mcfg, params, ds, ids)
        zeroed = kt.probe(mcfg, kt.zero_slots(params, np.arange(mcfg.pool_size)), ds, ids)
        slots = full["slots"]  # [facts, heads, k], clean routing
        counts = np.bincount(slots.reshape(-1), minlength=mcfg.pool_size).astype(float)
        p = counts / counts.sum()
        n = mcfg.n_sub_keys
        subkeys_used = np.mean([len(np.unique(slots[:, h] // n)) / n for h in range(slots.shape[1])]
                               + [len(np.unique(slots[:, h] % n)) / n for h in range(slots.shape[1])])
        last = meta["history"][-1]
        row = {
            "arm": arm, "flags": " ".join(ARMS[arm]),
            "facts_wrong": int((~full["correct"]).sum()),
            "accuracy": float(full["correct"].mean()),
            "mean_p_true": float(full["p_true"].mean()),
            "accuracy_pool_removed": float(zeroed["correct"].mean()),
            "eval_ce": last["ce"],
            "pool_coverage": float(np.mean(counts > 0)),
            "pool_active": float(np.mean(p > 0.1 / mcfg.pool_size)),
            "pool_spread": float(np.exp(-np.sum(p[p > 0] * np.log(p[p > 0]))) / mcfg.pool_size),
            "max_slot_share": float(p.max()),
            "subkeys_used": float(subkeys_used),
            "temperature": float(np.exp(np.clip(np.asarray(params["pool"]["log_temperature"]),
                                                np.log(mcfg.min_temperature), np.log(100.0)))),
            "top1_weight": float(full["weights"].max(-1).mean()),
            "history": meta["history"],
            "train_seconds": meta.get("train_seconds"),
        }
        print({k: v for k, v in row.items() if k != "history"}, flush=True)
        out.append(row)
    kt._save(out_name, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "analyze"])
    ap.add_argument("--gpus", default="0", help="comma-separated GPU ids, or 'cpu'")
    ap.add_argument("--per_gpu", type=int, default=1)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--round", type=int, default=1)
    a = ap.parse_args()
    if a.round > 1:
        ARMS.clear()
        ARMS.update(ARMS2 if a.round == 2 else ARMS3)
    if a.cmd == "train":
        train(a.gpus.split(","), a.per_gpu, a.only)
    else:
        analyze("memorization.json" if a.round == 1 else f"memorization{a.round}.json")
