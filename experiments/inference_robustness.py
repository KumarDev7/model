"""Does recall survive the inference shortcuts a low-memory engine wants?

For a trained checkpoint: fact recall (Q/A for people trained with Q/A and
people seen only in bios, bio wordings) and held-out loss, with the pool's
values stored as float32 / bfloat16 / float16 / int8 (per-row absmax scale)
and with fewer vectors read per head than in training (top_k override).

    python -m experiments.inference_robustness --data <fit_qa> --out <results dir> --arm pool_4x_fp16
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from . import facts_in_text as fit
from . import knowledge_study as ks
from . import knowledge_verify as kv
from . import text_study as ts


def quantize(values: np.ndarray, kind: str) -> np.ndarray:
    """Round-trip the pool table through a storage format (returns float32)."""
    if kind == "float32":
        return values
    if kind in ("bfloat16", "float16"):
        return np.asarray(jnp.asarray(values).astype(kind).astype(jnp.float32))
    if kind.startswith("int"):
        bits = int(kind[3:])
        q = 2 ** (bits - 1) - 1
        scale = np.abs(values).max(axis=1, keepdims=True) / q
        scale[scale == 0] = 1.0
        return (np.clip(np.round(values / scale), -q, q) * scale).astype(np.float32)
    raise ValueError(kind)


def evaluate(mcfg, params, tok, facts, val, n_people, n_windows):
    pr = kv.Prober(mcfg, params)
    row = {}
    nq = facts.get("qa_trained_people", 0)
    groups = {"bio_wording": (facts["A"][:n_people], facts["train_templates"])}
    if nq:
        groups["qa_trained"] = (facts["A"][:nq][:n_people], facts["qa_templates"])
        groups["qa_bio_only"] = (facts["A"][nq:][:n_people], facts["qa_templates"])
    for name, (people, templates) in groups.items():
        row[name] = float(pr.run(fit.probe_items(tok, people, templates), batch=128)["hit"].mean())
    f = kv.make_mode_loss(mcfg)
    row["heldout_loss"] = float(np.mean([float(f(params, b["inputs"], b["targets"], "normal"))
                                         for b in ts.windows(val, n_windows)]))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="pool_4x_fp16")
    ap.add_argument("--people", type=int, default=300)
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--formats", nargs="*", default=["float32", "bfloat16", "float16", "int8", "int4"])
    ap.add_argument("--top_ks", nargs="*", type=int, default=[16, 8, 4, 2, 1])
    a = ap.parse_args()
    facts, tok = ks.load_facts(a.data)
    val = np.load(os.path.join(a.data, "val.npy"), mmap_mode="r")
    mcfg, params, _ = ts.load(a.out, a.arm)
    values = np.asarray(params["pool"]["values"])
    rows = []
    for kind in a.formats:  # pool storage format, reads as trained
        p = {**params, "pool": {**params["pool"], "values": jnp.asarray(quantize(values, kind))}}
        r = {"pool_format": kind, "top_k": mcfg.top_k, **evaluate(mcfg, p, tok, facts, val, a.people, a.windows)}
        print(json.dumps(r), flush=True)
        rows.append(r)
    for k in a.top_ks:  # fewer vectors read per head at inference (float32 pool)
        if k == mcfg.top_k or k > mcfg.top_k:
            continue
        m = dataclasses.replace(mcfg, top_k=k)
        r = {"pool_format": "float32", "top_k": k, **evaluate(m, params, tok, facts, val, a.people, a.windows)}
        print(json.dumps(r), flush=True)
        rows.append(r)
    ks.save(a.out, f"inference_robustness_{a.arm}.json", rows)


if __name__ == "__main__":
    main()
