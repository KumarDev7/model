"""End to end: a trained model generates text with its pool moved to SSD.

Loads a checkpoint saved by memory_pool_model.train (pool on the device),
writes the pool values to an fp16 memory-mapped file, and generates the same
continuation with the pool (a) on the GPU and (b) read from the SSD file,
with the OS page cache dropped first. Reports whether the text matches and
the per-token latency of each.

    python -m experiments.ssd_generate --ckpt experiments/results/final/text_dp.msgpack \
        --pool_dir /root/pool_ssd/text --prompt "ROMEO:"
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from memory_pool_model.config import ModelConfig
from memory_pool_model.generate import Generator
from memory_pool_model.host_pool import HostPool
from memory_pool_model.model import MemoryPoolLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pool_dir", required=True)
    ap.add_argument("--prompt", default="ROMEO:")
    ap.add_argument("--n_new", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    meta = json.load(open(a.ckpt + ".json"))
    m = dict(meta["model"])
    m["memory_layers"] = tuple(m["memory_layers"])
    mcfg = ModelConfig(**{**m, "pool_location": "device", "host_pool": ""})
    template = MemoryPoolLM(mcfg).init(jax.random.PRNGKey(0), jnp.zeros((1, 8), jnp.int32))["params"]
    params = serialization.from_bytes(template, open(a.ckpt, "rb").read())
    values = np.asarray(params["pool"]["values"])

    # knowledge -> SSD (fp16 memmap); backbone/router/keys stay on the GPU
    ssd = HostPool(mcfg.pool_size, mcfg.d_value, path=a.pool_dir, trainable=False, dtype=np.float16,
                   name="ssd-generate")
    ssd.values[:] = values.astype(np.float16)
    ssd.flush()
    host_cfg = dataclasses.replace(mcfg, pool_location="host", host_pool=ssd.name)
    host_params = {**params, "pool": {k: v for k, v in params["pool"].items() if k != "values"}}
    gpu_params = jax.tree_util.tree_map(jnp.asarray, params)
    gpu_params["pool"]["values"] = jnp.asarray(values.astype(np.float16).astype(np.float32))  # same fp16 knowledge

    prompt = np.frombuffer(a.prompt.encode(), np.uint8).astype(np.int32)[None]
    out = {}
    for where, cfg, p in (("gpu", mcfg, gpu_params), ("ssd", host_cfg, host_params)):
        gen = Generator(cfg, p)
        gen.generate(prompt, n_new=4)  # compile
        if where == "ssd":
            ssd.evict_page_cache()
        r = gen.generate(prompt, n_new=a.n_new)
        text = bytes(r["tokens"][0].astype(np.uint8).tolist()).decode("utf-8", "replace")
        t = r["step_seconds"][prompt.shape[1] - 1:]
        out[where] = {"text": a.prompt + text, "ms_p50": float(np.median(t) * 1e3),
                      "ms_p90": float(np.percentile(t, 90) * 1e3), "tokens": r["tokens"][0].tolist()}
        print(f"--- pool on {where.upper()}: {out[where]['ms_p50']:.2f} ms/token (p90 {out[where]['ms_p90']:.2f})")
        print(out[where]["text"])
    same = out["gpu"]["tokens"] == out["ssd"]["tokens"]
    print(f"\nidentical output: {same}")
    summary = {"pool_slots": mcfg.pool_size, "d_value": mcfg.d_value,
               "pool_file_gb": ssd.values.nbytes / 1e9,
               "backbone_params": int(sum(x.size for k, v in params.items() if k != "pool"
                                          for x in jax.tree_util.tree_leaves(v))),
               "rows_per_token": mcfg.pool_heads * mcfg.top_k * len(mcfg.memory_layers),
               "identical_output": same, **{k: {kk: vv for kk, vv in v.items() if kk != "tokens"} for k, v in out.items()}}
    if a.out:
        json.dump(summary, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
