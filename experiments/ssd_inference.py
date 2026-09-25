"""Generation latency with the knowledge pool on GPU, in host RAM, or on SSD.

The backbone, router and product keys are on the GPU; with the pool on SSD
each generated token reads only the rows its router picks
(pool_heads * top_k per memory layer).

    python -m experiments.ssd_inference --n_sub 2048 --where gpu ram ssd_warm ssd_cold
    python -m experiments.ssd_inference --n_sub 8192 --where ssd_warm ssd_cold   # pool larger than RAM

Latency does not depend on what the pool has learned, so weights are random.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from memory_pool_model.config import ModelConfig
from memory_pool_model.generate import Generator
from memory_pool_model.host_pool import HostPool
from memory_pool_model.model import MemoryPoolLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sub", type=int, default=2048)
    ap.add_argument("--d_value", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--memory_layers", default="1,3")
    ap.add_argument("--where", nargs="*", default=["gpu", "ram", "ssd_warm", "ssd_cold"])
    ap.add_argument("--pool_dir", default="/root/pool_ssd")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", default="experiments/results/ssd_inference.json")
    a = ap.parse_args()

    base = ModelConfig(vocab_size=256, max_len=a.tokens + 8, d_model=a.d_model, n_layers=a.layers, n_heads=8,
                       memory_layers=tuple(int(x) for x in a.memory_layers.split(",")),
                       n_sub_keys=a.n_sub, pool_heads=4, d_key=128, d_value=a.d_value, top_k=16)
    n, rows_per_token = base.pool_size, base.pool_heads * base.top_k * len(base.memory_layers)
    print(f"pool {n:,} x {a.d_value} (fp16 {n * a.d_value * 2 / 1e9:.1f} GB), "
          f"{rows_per_token} rows/token = {rows_per_token * a.d_value * 2 / 1024:.0f} KB/token", flush=True)

    # backbone + router params (identical for every placement)
    host_cfg = dataclasses.replace(base, pool_location="host")
    ssd = ram = None
    results = json.load(open(a.out)) if os.path.exists(a.out) else []
    prompt = np.random.default_rng(0).integers(0, 256, size=(a.batch, 4)).astype(np.int32)

    for where in a.where:
        if where == "gpu":
            cfg = base
            params = MemoryPoolLM(cfg).init(jax.random.PRNGKey(1), jnp.zeros((1, 8), jnp.int32))["params"]
        else:
            if where == "ram":
                if ram is None:
                    t = time.time()
                    ram = HostPool(n, a.d_value, path=None, trainable=False, dtype=np.float16)
                    print(f"  built RAM pool in {time.time() - t:.0f}s", flush=True)
                pool = ram
            else:
                if ssd is None:
                    t = time.time()
                    ssd = HostPool(n, a.d_value, path=os.path.join(a.pool_dir, f"n{a.n_sub}_d{a.d_value}"),
                                   trainable=False, dtype=np.float16, name=f"ssd{a.n_sub}")
                    ssd.flush()
                    print(f"  SSD pool ready in {time.time() - t:.0f}s", flush=True)
                pool = ssd
            cfg = dataclasses.replace(host_cfg, host_pool=pool.name)
            params = MemoryPoolLM(cfg).init(jax.random.PRNGKey(1), jnp.zeros((1, 8), jnp.int32))["params"]

        gen = Generator(cfg, params, batch_size=a.batch)
        gen.generate(prompt, n_new=4)  # compile
        if where == "ssd_cold":
            # evict before every token: each read goes to the SSD itself
            times = []
            for _ in range(a.tokens // 8):
                ssd.evict_page_cache()
                r = gen.generate(prompt, n_new=8)
                times.extend(r["step_seconds"][prompt.shape[1] - 1:])
            times = np.array(times)
        else:
            if where == "ssd_warm":
                gen.generate(prompt, n_new=a.tokens)  # warm the page cache on this path
            r = gen.generate(prompt, n_new=a.tokens)
            times = r["step_seconds"][prompt.shape[1] - 1:]
        row = {"where": where, "pool_slots": n, "d_value": a.d_value, "batch": a.batch,
               "rows_per_token": rows_per_token, "pool_gb_fp16": n * a.d_value * 2 / 1e9,
               "ms_p50": float(np.median(times) * 1e3), "ms_p90": float(np.percentile(times, 90) * 1e3),
               "tokens_per_s": float(a.batch / np.mean(times))}
        print(json.dumps(row), flush=True)
        results.append(row)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
