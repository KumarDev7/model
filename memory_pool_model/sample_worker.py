"""Generate text from a training snapshot on the CPU.

Started by train.run every --sample_every steps (background.SampleLauncher)
while the accelerator keeps training. Each prompt gets a greedy continuation
with the pool as trained and one with its reads shuffled (each read returns
another slot's vector): if the two are the same, the model isn't using what
the pool returns. One JSON line per snapshot goes to --out.

    python -m memory_pool_model.sample_worker --params <save>.sample_params.msgpack \\
        --config <save>.config.json --tokenizer tokenizer.json --step 5000 --out <save>.samples.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import time


def _pin_cpus(n: int) -> None:
    """Use the last n cores, away from the training process's input pipeline."""
    if n <= 0 or not hasattr(os, "sched_setaffinity"):
        return
    cores = sorted(os.sched_getaffinity(0))
    if n < len(cores):
        os.sched_setaffinity(0, cores[-n:])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", required=True)
    ap.add_argument("--config", required=True, help="<save>.config.json of the training run")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", help="text file, one prompt per line (default: a fixed general set)")
    ap.add_argument("--new_tokens", type=int, default=48)
    ap.add_argument("--cpus", type=int, default=0, help="cores to use (0 = all)")
    ap.add_argument("--no_shuffled", action="store_true", help="skip the pool-shuffled continuations")
    ap.add_argument("--nice", type=int, default=10, help="lower this process's priority (training comes first)")
    a = ap.parse_args()
    if a.nice:
        os.nice(a.nice)
    _pin_cpus(a.cpus)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    t_start = time.time()

    import jax
    import numpy as np
    from flax import serialization
    from tokenizers import Tokenizer

    from .background import DEFAULT_PROMPTS
    from .config import ModelConfig
    from .generate import Generator

    with open(a.config) as f:
        m = dict(json.load(f)["model"])
    m["memory_layers"] = tuple(m["memory_layers"])
    # float32 on the CPU (bfloat16 is emulated there and slower); no device meshes
    m.update(pool_location="device", host_pool="", dp_mesh="", pool_mesh="", compute_dtype="float32")
    mcfg = ModelConfig(**{k: v for k, v in m.items() if k in ModelConfig.__dataclass_fields__})
    with open(a.params, "rb") as f:
        params = serialization.msgpack_restore(f.read())
    # on the device once: numpy leaves would be copied again at every generated token
    params = jax.tree_util.tree_map(
        lambda x: jax.device_put(np.asarray(x, np.float32) if np.issubdtype(np.asarray(x).dtype, np.floating) else x),
        params)
    tok = Tokenizer.from_file(a.tokenizer)
    eot = tok.token_to_id("<|endoftext|>")
    if a.prompts:
        with open(a.prompts) as f:
            prompts = [line.rstrip("\n") for line in f if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS
    modes = {"text": Generator(mcfg, params)}
    if mcfg.use_memory and not a.no_shuffled:
        modes["text_pool_shuffled"] = Generator(mcfg, params, shuffle_pool=True)

    samples, n_tok, t_gen = [], 0, time.time()
    for p in prompts:
        ids = tok.encode(p).ids[-(mcfg.max_len - a.new_tokens):]
        rec = {"prompt": p}
        for key, gen in modes.items():
            out = gen.generate(np.asarray([ids]), a.new_tokens)
            toks = [int(t) for t in out["tokens"][0]]
            if eot in toks:
                toks = toks[: toks.index(eot)]
            n_tok += a.new_tokens
            rec[key] = tok.decode(toks)
            if key == "text":
                # mean probability the model gave its own greedy tokens: low = unsure
                lg = out["logits"][0, len(ids) - 1:]
                pr = np.exp(lg - lg.max(-1, keepdims=True))
                pr /= pr.sum(-1, keepdims=True)
                rec["mean_top_prob"] = round(float(pr.max(-1).mean()), 4)
        samples.append(rec)
    gen_s = time.time() - t_gen
    if "text_pool_shuffled" in modes:
        same = sum(s["text"] == s["text_pool_shuffled"] for s in samples)
    else:
        same = None
    rec = {"type": "samples", "step": a.step, "time": round(time.time(), 3), "seconds": round(time.time() - t_start, 1),
           "tokens_per_s": round(n_tok / max(gen_s, 1e-9), 1), "same_with_pool_shuffled": same, "samples": samples}
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"=== samples at step {a.step} ({rec['seconds']}s, {rec['tokens_per_s']} tok/s on CPU)", flush=True)
    for s in samples:
        print(f"  PROMPT  {s['prompt']!r}\n  MODEL   {s['text']!r}", flush=True)
        if "text_pool_shuffled" in s:
            print(f"  SHUFFLE {s['text_pool_shuffled']!r}", flush=True)


if __name__ == "__main__":
    main()
