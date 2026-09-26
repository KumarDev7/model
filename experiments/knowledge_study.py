"""Does the pool store knowledge from real text? (facts-in-text benchmark)

Data: experiments/facts_in_text.py (fictional people's facts written as
sentences, mixed into Ultra-FineWeb). All arms: 4,000 steps x 8,192 tokens
(~0.95 passes, ~37 sightings of each bio).

  dense, dense_2x, dense_4x   backbone alone at 1x / ~2x / ~4x size
  pool                        + 262k-vector pool next to the FFN, cosine routing, 16 per head
  pool_qs                     + routing scores scaled by query length (sharper reads)
  pool_qs_k4                  + only 4 vectors per head
  pool_qs_k4_nopen            same without the no-pool penalty

Commands (run on the GPU machine):
  train     train the arms
  analyze   fact recall (seen / new wordings, pool on / off), held-out and
            Shakespeare perplexity, pool usage and read sharpness
  update    add set B's facts to a trained model (full fine-tune, pool
            vectors only, pool vectors only + replay of set A) and measure
            what is learnt and what is forgotten
  cache     how many pool reads a RAM cache of the hottest vectors absorbs
  ssd       per-token latency with the pool read from an fp16 file on disk
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

from memory_pool_model.config import TrainConfig
from memory_pool_model.data import TokenDataset
from memory_pool_model.train import Trainer, split_values

from . import facts_in_text as fit
from . import text_study as ts

BACKBONE = ts.BACKBONE
POOL = ["--memory_layers", "1,3", "--n_sub_keys", "512", "--d_key", "128", "--d_value", "256",
        "--memory_ffn", "true"]
QS = ["--router_query_scale", "true"]
ARMS = {
    "pool": POOL,
    "pool_qs": POOL + QS,
    "pool_qs_k4": POOL + QS + ["--top_k", "4"],
    "pool_qs_k4_nopen": POOL + QS + ["--top_k", "4", "--nopool_true_coef", "0.0"],
    "dense": ["--use_memory", "false"],
    "dense_2x": ["--use_memory", "false", "--d_model", "384"],
    "dense_4x": ["--use_memory", "false", "--d_model", "512", "--n_layers", "6"],
}
PILOT = {k: ARMS[k] for k in ("pool", "pool_qs", "pool_qs_k4")}


def load_facts(data):
    from tokenizers import Tokenizer

    return (json.load(open(os.path.join(data, "facts.json"))),
            Tokenizer.from_file(os.path.join(data, "tokenizer.json")))


def save(out, name, obj):
    path = os.path.join(out, name)
    json.dump(obj, open(path, "w"), indent=1, default=float)
    print(f"wrote {path}", flush=True)


# ------------------------------------------------------------------ analyze
def analyze(data, out, n_windows, only=None):
    facts, tok = load_facts(data)
    val = np.load(os.path.join(data, "val.npy"), mmap_mode="r")
    ood = np.load(os.path.join(data, "ood.npy"), mmap_mode="r")
    rows = []
    for arm in ARMS:
        if (only and arm not in only) or not os.path.exists(os.path.join(out, "ckpt", arm + ".msgpack.json")):
            continue
        mcfg, params, meta = ts.load(out, arm)
        fwd, fwd_off = ts.make_forward(mcfg)
        n_params = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
        n_pool = int(params["pool"]["values"].size) if mcfg.use_memory else 0
        row = {"arm": arm, "flags": " ".join(ARMS[arm]), "params_total": n_params,
               "params_without_pool_values": n_params - n_pool,
               "vectors_read_per_token": (mcfg.pool_heads * mcfg.top_k * len(mcfg.memory_layers)
                                          if mcfg.use_memory else 0),
               "train_seconds": meta.get("train_seconds"), "history": meta["history"]}
        if mcfg.use_memory:
            lt = float(params["pool"]["log_temperature"])
            row["temperature"] = float(np.exp(np.clip(lt, np.log(mcfg.min_temperature), np.log(mcfg.max_temperature))))
        row["heldout"], _ = ts.measure(mcfg, params, fwd, fwd_off, ts.windows(val, n_windows))
        row["ood_shakespeare"], _ = ts.measure(mcfg, params, fwd, fwd_off, ts.windows(ood, n_windows // 2))
        row["facts_A"] = fit.recall_report(mcfg, params, tok, facts["A"], facts)
        row["facts_B_untrained"] = fit.recall_report(mcfg, params, tok, facts["B"][:300], facts)
        print(json.dumps({k: v for k, v in row.items() if k != "history"}, default=float)[:1800], flush=True)
        rows.append(row)
    save(out, "knowledge_study.json", rows)


# ------------------------------------------------------------------- update
def finetune(mcfg, params, tcfg, data_arr, steps, seed=0):
    tr = Trainer(mcfg, tcfg, donate=True)
    state = tr.init(jax.random.PRNGKey(seed))
    state = state.replace(params=jax.tree_util.tree_map(jnp.asarray, params))
    if tr.sparse:
        state = state.replace(opt_state=tr.optimizer.init(split_values(state.params)[1]))
    else:
        state = state.replace(opt_state=tr.optimizer.init(state.params))
    ds = TokenDataset.__new__(TokenDataset)
    ds.train, ds.seq_len = data_arr, ts.SEQ
    rng = np.random.default_rng(seed)
    t = time.time()
    for step in range(1, steps + 1):
        state, _, _ = tr.train_step(state, ds.sample(rng, tcfg.batch_size), jax.random.PRNGKey(step))
    jax.block_until_ready(state.params)
    return state.params, time.time() - t


def update(data, out, arms, steps):
    facts, tok = load_facts(data)
    b_mix = np.load(os.path.join(data, "b_mix.npy"))
    a_docs = np.load(os.path.join(data, "a_docs.npy"))
    replay = fit.shuffle_mix(np.random.default_rng(1), b_mix, a_docs[: len(b_mix) // 4])
    val = np.load(os.path.join(data, "val.npy"), mmap_mode="r")
    rows = []
    for arm in arms:
        mcfg, params, meta = ts.load(out, arm)
        base = TrainConfig(**{k: v for k, v in meta["train"].items() if k in TrainConfig.__dataclass_fields__})
        base = dataclasses.replace(base, steps=steps, warmup_steps=50, revive_every=0, checkpoint_every=0)
        modes = {"full_finetune": (base, b_mix)}
        if mcfg.use_memory:
            modes["pool_vectors_only"] = (dataclasses.replace(base, pool_values_only=True), b_mix)
            modes["pool_vectors_only_replay"] = (dataclasses.replace(base, pool_values_only=True), replay)
        before_a = fit.recall_report(mcfg, params, tok, facts["A"][:500], facts)
        for mode, (tcfg, arr) in modes.items():
            new, secs = finetune(mcfg, params, tcfg, arr, steps)
            fwd, fwd_off = ts.make_forward(mcfg)
            r = {"arm": arm, "mode": mode, "steps": steps, "seconds": secs,
                 "A_before": before_a,
                 "A_after": fit.recall_report(mcfg, new, tok, facts["A"][:500], facts),
                 "B_after": fit.recall_report(mcfg, new, tok, facts["B"], facts),
                 "heldout_ppl_after": ts.measure(mcfg, new, fwd, fwd_off, ts.windows(val, 256))[0]["ppl"]}
            print(json.dumps({k: (v["seen_wording"]["exact"], v["new_wording"]["exact"]) if isinstance(v, dict)
                              and "seen_wording" in v else v for k, v in r.items()}, default=float), flush=True)
            rows.append(r)
    save(out, "knowledge_update.json", rows)


# -------------------------------------------------------------------- cache
def cache(data, out, arm, n_windows):
    """Static cache of the vectors read most on training text; hit rate on
    held-out text and on Shakespeare."""
    mcfg, params, _ = ts.load(out, arm)
    fwd, fwd_off = ts.make_forward(mcfg)
    train = np.load(os.path.join(data, "train.npy"), mmap_mode="r")
    _, hot = ts.measure(mcfg, params, fwd, fwd_off, ts.windows(train, n_windows, np.random.default_rng(5)))
    res = {"arm": arm, "pool_vectors": mcfg.pool_size, "layers": []}
    for name, arr in (("heldout", np.load(os.path.join(data, "val.npy"), mmap_mode="r")),
                      ("ood_shakespeare", np.load(os.path.join(data, "ood.npy"), mmap_mode="r"))):
        _, counts = ts.measure(mcfg, params, fwd, fwd_off, ts.windows(arr, n_windows))
        for li, (h, c) in enumerate(zip(hot, counts)):
            order = np.argsort(-h)
            row = {"split": name, "layer": li, "reads": float(c.sum())}
            for frac in (0.01, 0.05, 0.10, 0.25, 0.50):
                cached = order[: int(frac * len(order))]
                row[f"hit_rate_cache_{int(frac * 100)}pct"] = float(c[cached].sum() / c.sum())
            res["layers"].append(row)
    print(json.dumps(res), flush=True)
    save(out, "knowledge_cache.json", res)


# ---------------------------------------------------------------------- ssd
def ssd(data, out, arm, pool_dir, tokens=64):
    from memory_pool_model.generate import Generator
    from memory_pool_model.host_pool import HostPool

    mcfg, params, _ = ts.load(out, arm)
    values = np.asarray(params["pool"]["values"])
    pool = HostPool(mcfg.pool_size, mcfg.d_value, path=pool_dir, trainable=False, dtype=np.float16,
                    name="ssd-knowledge")
    pool.values[:] = values.astype(np.float16)
    pool.flush()
    host_cfg = dataclasses.replace(mcfg, pool_location="host", host_pool=pool.name)
    host_params = {**params, "pool": {k: v for k, v in params["pool"].items() if k != "values"}}
    gpu_params = jax.tree_util.tree_map(jnp.asarray, params)
    gpu_params["pool"]["values"] = jnp.asarray(values.astype(np.float16).astype(np.float32))
    val = np.load(os.path.join(data, "val.npy"), mmap_mode="r")
    rows = []
    for batch in (1, 8, 32):
        prompt = np.stack([np.asarray(val[i * 997 : i * 997 + 16]) for i in range(batch)]).astype(np.int32)
        outs = {}
        for where, cfg, p in (("gpu", mcfg, gpu_params), ("ssd_cold", host_cfg, host_params),
                              ("ssd_warm", host_cfg, host_params)):
            gen = Generator(cfg, p, batch_size=batch)
            gen.generate(prompt, n_new=4)
            if where == "ssd_cold":
                times = []
                for _ in range(tokens // 16):
                    pool.evict_page_cache()
                    r = gen.generate(prompt, n_new=16)
                    times.extend(r["step_seconds"][prompt.shape[1] - 1:])
            else:
                if where == "ssd_warm":
                    gen.generate(prompt, n_new=tokens)
                r = gen.generate(prompt, n_new=tokens)
                times = r["step_seconds"][prompt.shape[1] - 1:]
                outs[where] = r["tokens"]
            times = np.asarray(times)
            rows.append({"batch": batch, "where": where, "ms_per_step_p50": float(np.median(times) * 1e3),
                         "tokens_per_s": float(batch / np.mean(times))})
            print(rows[-1], flush=True)
        rows.append({"batch": batch, "identical_output_gpu_vs_ssd": bool(np.array_equal(outs["gpu"], outs["ssd_warm"]))})
    save(out, "knowledge_ssd.json", {"arm": arm, "pool_vectors": mcfg.pool_size, "pool_file_mb": values.size * 2 / 1e6,
                                     "vectors_read_per_token": mcfg.pool_heads * mcfg.top_k * len(mcfg.memory_layers),
                                     "rows": rows})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "pilot", "analyze", "update", "cache", "ssd"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--per_gpu", type=int, default=1)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--windows", type=int, default=512)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--arm", default="pool")
    ap.add_argument("--pool_dir", default="/root/pool_ssd/knowledge")
    a = ap.parse_args()
    if a.cmd in ("train", "pilot"):
        arms = ARMS if a.cmd == "train" else PILOT
        ts.launch(arms, ts.token_task_args(a.data, a.steps) + BACKBONE, a.out, a.gpus.split(","), a.per_gpu, a.only)
    elif a.cmd == "analyze":
        analyze(a.data, a.out, a.windows, a.only)
    elif a.cmd == "update":
        update(a.data, a.out, a.only or ["pool", "dense"], a.steps)
    elif a.cmd == "cache":
        cache(a.data, a.out, a.arm, a.windows)
    else:
        ssd(a.data, a.out, a.arm, a.pool_dir)
