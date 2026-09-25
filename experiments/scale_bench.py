"""Scaling benchmark: step time and peak device memory vs pool size.

Each configuration runs in its own process so peak-memory numbers are clean.

    python -m experiments.scale_bench                      # pool-size sweep, sparse vs dense
    python -m experiments.scale_bench --devices 1 2        # data-parallel throughput
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

CHILD = r"""
import json, os, sys, time
import jax, jax.numpy as jnp, numpy as np
from memory_pool_model.config import ModelConfig, TrainConfig
from memory_pool_model.train import Trainer
cfg = json.loads(sys.argv[1])
n_dev = cfg["devices"]
devices = jax.devices()[:n_dev]
mcfg = ModelConfig(vocab_size=256, max_len=cfg["seq"], d_model=cfg["d_model"], n_layers=cfg["layers"],
                   n_heads=8, memory_layers=tuple(cfg["memory_layers"]), memory_ffn=False,
                   n_sub_keys=cfg["n_sub"], pool_heads=4, d_key=128, d_value=cfg["d_value"], top_k=16,
                   pool_location=cfg.get("location", "device"), pool_dir=cfg.get("pool_dir", ""))
tcfg = TrainConfig(steps=1000, batch_size=cfg["batch"], sparse_pool_updates=cfg["mode"] == "sparse")
mesh = None
if n_dev > 1:
    from jax.sharding import Mesh
    mesh = Mesh(np.array(devices), ("data",))
tr = Trainer(mcfg, tcfg, mesh=mesh, donate=True)
with jax.default_device(devices[0]):
    st = tr.place_state(tr.init(jax.random.PRNGKey(0)))
rng = np.random.default_rng(0)
def batch():
    x = rng.integers(0, 256, size=(cfg["batch"], cfg["seq"] + 1)).astype(np.int32)
    return tr.place_batch({"inputs": x[:, :-1], "targets": x[:, 1:], "mask": np.ones((cfg["batch"], cfg["seq"]), np.float32)})
for i in range(3):  # compile + warm up
    st, m, _ = tr.train_step(st, batch(), jax.random.PRNGKey(i))
jax.block_until_ready(st)
t = time.time()
for i in range(cfg["steps"]):
    st, m, _ = tr.train_step(st, batch(), jax.random.PRNGKey(100 + i))
jax.block_until_ready(st)
dt = (time.time() - t) / cfg["steps"]
peak = max((d.memory_stats() or {}).get("peak_bytes_in_use", 0) for d in devices)
print("RESULT " + json.dumps({**cfg, "step_ms": dt * 1e3, "tokens_per_s": cfg["batch"] * cfg["seq"] / dt,
      "peak_gb": peak / 1e9, "pool_slots": cfg["n_sub"] ** 2,
      "loss": float(m["loss"]), "rows_updated": int(m.get("rows_updated", -1)),
      "host_pool_gb": tr.host.nbytes() / 1e9 if tr.host else 0.0}), flush=True)
"""


def run_one(cfg, timeout=1800):
    env = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    try:
        r = subprocess.run([sys.executable, "-c", CHILD, json.dumps(cfg)], capture_output=True,
                           text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {**cfg, "error": "timeout"}
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    err = (r.stderr.strip().splitlines() or ["?"])[-1]
    return {**cfg, "error": "OOM" if "RESOURCE_EXHAUSTED" in r.stderr or "out of memory" in r.stderr.lower() else err[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sub", type=int, nargs="*", default=[64, 256, 1024, 2048])
    ap.add_argument("--modes", nargs="*", default=["sparse", "dense"])
    ap.add_argument("--devices", type=int, nargs="*", default=[1])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--d_value", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--memory_layers", default="1,3")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--location", default="device", choices=["device", "host"])
    ap.add_argument("--pool_dir", default="", help="host pool on SSD under this dir (empty = RAM)")
    ap.add_argument("--out", default="experiments/results/scale_bench.json")
    a = ap.parse_args()
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    for n_sub in a.n_sub:
        for mode in a.modes:
            for dev in a.devices:
                cfg = dict(n_sub=n_sub, mode=mode, devices=dev, batch=a.batch, seq=a.seq, d_model=a.d_model,
                           d_value=a.d_value, layers=a.layers,
                           memory_layers=[int(x) for x in a.memory_layers.split(",")], steps=a.steps,
                           location=a.location,
                           pool_dir=os.path.join(a.pool_dir, f"train_n{n_sub}") if a.pool_dir else "")
                t = time.time()
                res = run_one(cfg)
                print(json.dumps(res), f"({time.time() - t:.0f}s)", flush=True)
                rows.append(res)
                os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
                json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
