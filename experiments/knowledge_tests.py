"""Accuracy, capacity, vector-sharing and retention tests for the memory-pool LM.

    python -m experiments.knowledge_tests train      # train all models (skips existing)
    python -m experiments.knowledge_tests analyze    # probe the main model
    python -m experiments.knowledge_tests retention  # add-new-facts / forgetting test
    python -m experiments.knowledge_tests summary    # print a table of everything

Results are written to experiments/results/*.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from typing import Callable, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from memory_pool_model.config import ModelConfig, TrainConfig
from memory_pool_model.data import FactDataset
from memory_pool_model.model import MemoryPoolLM
from memory_pool_model.train import Trainer, _path_is

RESULTS = os.environ.get("KT_RESULTS", os.path.join(os.path.dirname(__file__), "results"))
CKPT = os.path.join(RESULTS, "ckpt")

# --------------------------------------------------------------------------
# Training runs (each is a CLI invocation of memory_pool_model.train)
# --------------------------------------------------------------------------
MAIN_DATA = ["--num_entities", "4096", "--num_relations", "4"]  # 16,384 facts
SWEEP_DATA = ["--num_relations", "4", "--name_len", "4"]  # entity count varies

RUNS: Dict[str, List[str]] = {
    # Main model: 4,096-slot pool, 16,384 facts (4 facts per slot).
    "main": MAIN_DATA + ["--steps", "4000"],
    # Same backbone, no pool.
    "dense_small": MAIN_DATA + ["--steps", "4000", "--use_memory", "false"],
    # No pool, FFN widened so total params ~= main model (833k vs 850k).
    "dense_matched": MAIN_DATA + ["--steps", "4000", "--use_memory", "false", "--ffn_mult", "9"],
}
# Capacity sweep: fixed 1,024-slot pool, growing number of facts.
CAP_STEPS = {4096: 3000, 8192: 3500, 16384: 5000, 32768: 5000}
for n_facts, steps in sorted(CAP_STEPS.items(), reverse=True):
    RUNS[f"cap_{n_facts}"] = SWEEP_DATA + [
        "--num_entities", str(n_facts // 4), "--n_sub_keys", "32", "--steps", str(steps),
    ]
    RUNS[f"cap_{n_facts}_dense"] = SWEEP_DATA + [
        "--num_entities", str(n_facts // 4), "--use_memory", "false", "--steps", str(steps),
    ]


def train_all(parallel: int, only: List[str] | None) -> None:
    os.makedirs(CKPT, exist_ok=True)
    todo = [n for n in RUNS if (not only or n in only) and not os.path.exists(os.path.join(CKPT, n + ".msgpack.json"))]
    # Slowest (memory, most steps) first.
    todo.sort(key=lambda n: ("dense" in n, -int(RUNS[n][RUNS[n].index("--steps") + 1])))
    env = {**os.environ, "XLA_FLAGS": os.environ.get("XLA_FLAGS", "")}
    running: List[subprocess.Popen] = []
    while todo or running:
        while todo and len(running) < parallel:
            name = todo.pop(0)
            cmd = [sys.executable, "-m", "memory_pool_model.train", *RUNS[name],
                   "--log_every", "500", "--eval_every", "1000",
                   "--save", os.path.join(CKPT, name + ".msgpack")]
            log = open(os.path.join(RESULTS, f"train_{name}.log"), "w")
            print(f"[start] {name}: {' '.join(cmd[3:])}", flush=True)
            running.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env))
            running[-1].name = name  # type: ignore[attr-defined]
        time.sleep(5)
        for p in list(running):
            if p.poll() is not None:
                print(f"[done ] {p.name} exit={p.returncode}", flush=True)  # type: ignore[attr-defined]
                running.remove(p)


# --------------------------------------------------------------------------
# Loading and probing
# --------------------------------------------------------------------------
def load(name: str):
    path = os.path.join(CKPT, name + ".msgpack")
    meta = json.load(open(path + ".json"))
    m = dict(meta["model"])
    m["memory_layers"] = tuple(m["memory_layers"])
    mcfg = ModelConfig(**m)
    ds = FactDataset(**meta["dataset"])
    template = MemoryPoolLM(mcfg).init(jax.random.PRNGKey(0), jnp.zeros((1, mcfg.max_len), jnp.int32))["params"]
    params = jax.tree_util.tree_map(jnp.asarray, serialization.from_bytes(template, open(path, "rb").read()))
    return mcfg, ds, params, meta


def encode_facts(ds: FactDataset, fact_ids: np.ndarray):
    """Pack facts into sequences; return batch + (row, position) of each fact's answer."""
    F = ds.facts_per_seq
    n = len(fact_ids)
    padded = np.pad(fact_ids, (0, (-n) % F), mode="wrap")
    ent, rel = np.divmod(padded, ds.num_relations)
    batch = ds._encode(ent.reshape(-1, F), rel.reshape(-1, F))
    j = np.arange(n)
    rows, slot_in_row = j // F, j % F
    # Input position holding the relation token; its output predicts the attribute.
    pos = 1 + slot_in_row * ds.fact_len + ds.name_len
    return batch, rows, pos


_FWD_CACHE: Dict = {}


def _forward(mcfg: ModelConfig):
    if mcfg not in _FWD_CACHE:
        model = MemoryPoolLM(mcfg)

        @jax.jit
        def fwd(params, inputs):
            logits, aux = model.apply({"params": params}, inputs)
            if aux:
                return logits, aux["slots"][0], aux["weights"][0]
            B, T = inputs.shape
            return logits, jnp.zeros((B, T, 1, 1), jnp.int32), jnp.zeros((B, T, 1, 1))

        _FWD_CACHE[mcfg] = fwd
    return _FWD_CACHE[mcfg]


def probe(mcfg, params, ds, fact_ids, rows_per_batch: int = 256) -> Dict[str, np.ndarray]:
    """Per-fact correctness, p(correct answer), fetched slots and weights."""
    fwd = _forward(mcfg)
    fact_ids = np.asarray(fact_ids)
    per = rows_per_batch * ds.facts_per_seq
    out = {k: [] for k in ("correct", "p_true", "slots", "weights")}
    for s in range(0, len(fact_ids), per):
        ids = fact_ids[s : s + per]
        batch, rows, pos = encode_facts(ds, ids)
        inputs = batch["inputs"]
        # pad rows to a power of two so jit compiles only a few shapes
        n_rows = max(8, 1 << (len(inputs) - 1).bit_length())
        inputs = np.pad(inputs, ((0, n_rows - len(inputs)), (0, 0)))
        logits, slots, weights = fwd(params, inputs)
        logits = np.asarray(logits)[rows, pos]
        target = batch["targets"][rows, pos]
        probs = jax.nn.softmax(logits, -1)
        out["correct"].append(logits.argmax(-1) == target)
        out["p_true"].append(np.asarray(probs)[np.arange(len(ids)), target])
        out["slots"].append(np.asarray(slots)[rows, pos])
        out["weights"].append(np.asarray(weights)[rows, pos])
    return {k: np.concatenate(v) for k, v in out.items()}


def zero_slots(params, rows: np.ndarray):
    values = jnp.asarray(params["pool"]["values"]).at[jnp.asarray(rows)].set(0.0, mode="drop")
    return {**params, "pool": {**params["pool"], "values": values}}


# --------------------------------------------------------------------------
# Analysis of the main model
# --------------------------------------------------------------------------
def analyze(name: str = "main", n_targeted: int = 400, seed: int = 0) -> Dict:
    rng = np.random.default_rng(seed)
    mcfg, ds, params, meta = load(name)
    N, P = ds.num_facts, mcfg.pool_size
    all_ids = np.arange(N)
    attrs = ds.table.reshape(-1)  # attribute of fact id (ent * R + rel)
    res: Dict = {"name": name, "num_facts": int(N), "pool_size": int(P),
                 "fetch_per_token": mcfg.pool_heads * mcfg.top_k}

    t0 = time.time()
    base = probe(mcfg, params, ds, all_ids)
    acc = base["correct"].mean()
    res["accuracy"] = {
        "top1": float(acc),
        "facts_correct": int(base["correct"].sum()),
        "mean_p_true": float(base["p_true"].mean()),
        "confident_correct_p>0.9": float(np.mean(base["correct"] & (base["p_true"] > 0.9))),
        "per_relation": [float(base["correct"][r :: ds.num_relations].mean()) for r in range(ds.num_relations)],
        "chance": 1.0 / (ds.vocab_size - ds.attr_offset),
    }
    print("accuracy", res["accuracy"], f"{time.time() - t0:.0f}s", flush=True)

    # ---- 1. how many vectors does one fact need? (inference-time top-k) ----
    res["topk_at_inference"] = []
    for k in (1, 2, 4, 8, 16, 32, 64):
        if k > mcfg.n_sub_keys:
            continue
        p = probe(dataclasses.replace(mcfg, top_k=k), params, ds, all_ids)
        res["topk_at_inference"].append({"top_k": k, "vectors_per_token": k * mcfg.pool_heads,
                                         "accuracy": float(p["correct"].mean())})
    print("topk", res["topk_at_inference"], flush=True)

    # ---- 2. mixing: how concentrated are the weights for one fact? ----
    w = base["weights"]  # [N, H, k] sorted by score (desc)
    pr = 1.0 / np.sum(w**2, axis=-1)  # participation ratio per head
    res["mixing"] = {
        "top1_weight_mean": float(w[..., 0].mean()),
        "top2_weight_mean": float(w[..., :2].sum(-1).mean()),
        "top4_weight_mean": float(w[..., :4].sum(-1).mean()),
        "effective_vectors_per_head_mean": float(pr.mean()),
        "effective_vectors_per_head_pcts": [float(x) for x in np.percentile(pr, [10, 50, 90])],
        "effective_vectors_per_fact_mean": float(pr.sum(-1).mean()),
    }

    # ---- 3. sharing: how many facts live in the same vector? ----
    primary = base["slots"][..., 0]  # [N, H] top-1 slot per head
    H = primary.shape[1]
    fact_of = np.repeat(all_ids, H)
    slot_flat = primary.reshape(-1)
    pairs = np.unique(np.stack([slot_flat, fact_of], 1), axis=0)  # unique (slot, fact)
    facts_per_slot = np.bincount(pairs[:, 0], minlength=P)
    used = facts_per_slot > 0
    groups: Dict[int, np.ndarray] = {}
    order = np.argsort(pairs[:, 0], kind="stable")
    sp = pairs[order]
    bounds = np.flatnonzero(np.diff(sp[:, 0])) + 1
    for chunk in np.split(sp, bounds):
        groups[int(chunk[0, 0])] = chunk[:, 1]
    distinct_attr_ratio, purity = [], []
    for s, f in groups.items():
        if len(f) >= 2:
            a = attrs[f]
            distinct_attr_ratio.append(len(np.unique(a)) / len(a))
            purity.append(np.bincount(a).max() / len(a))
    res["sharing"] = {
        "slots_primary_for_any_fact": float(used.mean()),
        "facts_per_used_slot_mean": float(facts_per_slot[used].mean()),
        "facts_per_used_slot_pcts": [float(x) for x in np.percentile(facts_per_slot[used], [10, 50, 90, 99])],
        "facts_per_used_slot_max": int(facts_per_slot.max()),
        "hist": {str(b): int(c) for b, c in zip(*np.unique(np.minimum(facts_per_slot[used], 64), return_counts=True))},
        "distinct_answers_fraction_in_shared_slots": float(np.mean(distinct_attr_ratio)),
        "answer_purity_in_shared_slots": float(np.mean(purity)),
    }

    # accuracy as a function of how crowded the fact's primary vectors are
    load_ = facts_per_slot[primary].mean(-1)
    buckets = [(1, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10**9)]
    res["accuracy_by_slot_load"] = []
    for lo, hi in buckets:
        m = (load_ >= lo) & (load_ < hi)
        if m.sum() >= 20:
            res["accuracy_by_slot_load"].append({"facts_per_vector": f"{lo}-{hi if hi < 10**9 else '+'}",
                                                 "n_facts": int(m.sum()),
                                                 "accuracy": float(base["correct"][m].mean())})

    # ---- 4. group test: are all facts that share one vector recalled together? ----
    correct = base["correct"]
    res["shared_vector_groups"] = []
    for g in (2, 3, 4, 6, 8, 12, 16):
        rows = [f for f in groups.values() if len(f) >= g]
        if len(rows) < 20:
            continue
        # take exactly g facts with pairwise-different answers from each group
        sel = []
        for f in rows:
            _, first = np.unique(attrs[f], return_index=True)
            f = f[np.sort(first)]
            if len(f) >= g:
                sel.append(rng.choice(f, g, replace=False))
        if len(sel) < 20:
            continue
        sel = np.array(sel)
        all_ok = correct[sel].all(1).mean()
        rand = rng.choice(N, size=sel.shape)
        res["shared_vector_groups"].append({
            "facts_in_vector": g, "n_groups": len(sel),
            "all_correct": float(all_ok),
            "all_correct_random_groups": float(correct[rand].all(1).mean()),
            "independent_expectation": float(acc**g),
        })

    # ---- 5. triplet test: fact C overlaps vectors with A (head i) and B (head j) ----
    trip = []
    for _ in range(20000):
        c = rng.integers(N)
        hi, hj = rng.choice(H, 2, replace=False)
        ga, gb = groups[int(primary[c, hi])], groups[int(primary[c, hj])]
        ga, gb = ga[ga != c], gb[gb != c]
        if len(ga) == 0 or len(gb) == 0:
            continue
        a, b = rng.choice(ga), rng.choice(gb)
        if a == b or len({attrs[a], attrs[b], attrs[c]}) < 3:
            continue
        trip.append((a, b, c))
        if len(trip) >= 5000:
            break
    trip = np.array(trip)
    res["triplets"] = {
        "n": int(len(trip)),
        "all_three_correct": float(correct[trip].all(1).mean()),
        "c_correct": float(correct[trip[:, 2]].mean()),
        "independent_expectation": float(acc**3),
    }

    # ---- 6. random ablation: delete a fraction of the pool ----
    res["random_ablation"] = []
    for frac in (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0):
        accs = []
        for s in range(1 if frac in (0.0, 1.0) else 3):
            dead = np.random.default_rng(100 + s).choice(P, int(round(frac * P)), replace=False)
            accs.append(probe(mcfg, zero_slots(params, dead), ds, all_ids)["correct"].mean())
        res["random_ablation"].append({"fraction_deleted": frac, "accuracy": float(np.mean(accs))})
    print("ablation", res["random_ablation"], flush=True)

    # ---- 7. targeted ablation: delete the vectors one fact relies on ----
    ok_ids = rng.choice(np.flatnonzero(correct), n_targeted, replace=False)
    levels = (1, 2, 4, mcfg.top_k)
    tgt = {r: [] for r in levels}
    neigh_before, neigh_after, ctrl_before, ctrl_after = [], [], [], []
    for c in ok_ids:
        for r in levels:
            rows = np.unique(base["slots"][c, :, :r])
            if r == 1:
                neigh = np.unique(np.concatenate([groups[int(s)] for s in rows]))
                neigh = neigh[neigh != c][:40]
                ctrl = rng.choice(N, 20, replace=False)
                ids = np.concatenate([[c], neigh, ctrl])
                p = probe(mcfg, zero_slots(params, rows), ds, ids)
                tgt[r].append(p["correct"][0])
                neigh_before.append(correct[neigh]); neigh_after.append(p["correct"][1 : 1 + len(neigh)])
                ctrl_before.append(correct[ctrl]); ctrl_after.append(p["correct"][1 + len(neigh) :])
            else:
                tgt[r].append(probe(mcfg, zero_slots(params, rows), ds, np.array([c]))["correct"][0])
    res["targeted_ablation"] = {
        "n_facts": int(n_targeted),
        "fact_survives": [{"vectors_deleted_per_head": r, "vectors_deleted": r * H,
                           "accuracy": float(np.mean(tgt[r]))} for r in levels],
        "neighbours_sharing_deleted_top1": {
            "before": float(np.concatenate(neigh_before).mean()),
            "after": float(np.concatenate(neigh_after).mean()),
            "n": int(sum(len(x) for x in neigh_before)),
        },
        "random_control_facts": {
            "before": float(np.concatenate(ctrl_before).mean()),
            "after": float(np.concatenate(ctrl_after).mean()),
        },
    }
    print("targeted", res["targeted_ablation"], flush=True)

    # ---- 8. pool usage while answering facts (clean router) ----
    counts = np.bincount(base["slots"].reshape(-1), minlength=P).astype(np.float64)
    p_ = counts / counts.sum()
    ent = -np.sum(p_[p_ > 0] * np.log(p_[p_ > 0]))
    res["pool_usage_on_facts"] = {
        "coverage_fetched_at_least_once": float(np.mean(counts > 0)),
        "active_10pct_of_fair_share": float(np.mean(p_ > 0.1 / P)),
        "spread": float(np.exp(ent) / P),
        "effective_vectors": float(np.exp(ent)),
    }
    res["seconds"] = time.time() - t0
    return res


def compare_dense(names=("main", "dense_small", "dense_matched")) -> List[Dict]:
    rows = []
    for n in names:
        path = os.path.join(CKPT, n + ".msgpack")
        if not os.path.exists(path):
            continue
        mcfg, ds, params, meta = load(n)
        p = probe(mcfg, params, ds, np.arange(ds.num_facts))
        n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
        n_pool = sum(x.size for x in jax.tree_util.tree_leaves(params.get("pool", {})))
        rows.append({"name": n, "params": int(n_params), "pool_params": int(n_pool),
                     "accuracy": float(p["correct"].mean()), "mean_p_true": float(p["p_true"].mean()),
                     "train_seconds": meta.get("train_seconds"), "history": meta["history"]})
    return rows


def capacity() -> List[Dict]:
    rows = []
    for n_facts in (4096, 8192, 16384, 32768):
        row = {"facts": n_facts}
        for kind, key in (("memory", f"cap_{n_facts}"), ("dense", f"cap_{n_facts}_dense")):
            path = os.path.join(CKPT, key + ".msgpack")
            if not os.path.exists(path):
                continue
            mcfg, ds, params, meta = load(key)
            p = probe(mcfg, params, ds, np.arange(ds.num_facts))
            row[f"{kind}_accuracy"] = float(p["correct"].mean())
            row[f"{kind}_facts_stored"] = int(p["correct"].sum())
            if kind == "memory":
                row["pool_size"] = mcfg.pool_size
                row["facts_per_vector"] = n_facts / mcfg.pool_size
                counts = np.bincount(p["slots"].reshape(-1), minlength=mcfg.pool_size)
                row["pool_coverage"] = float(np.mean(counts > 0))
                row["pool_active"] = float(np.mean(counts / counts.sum() > 0.1 / mcfg.pool_size))
                row["history"] = meta["history"]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Retention: learn facts A, then learn new facts B. Is A kept?
# --------------------------------------------------------------------------
class FactSubset:
    def __init__(self, ds: FactDataset, entities: np.ndarray):
        self.ds, self.entities = ds, entities
        R = ds.num_relations
        self.fact_ids = (entities[:, None] * R + np.arange(R)).reshape(-1)

    def sample(self, rng, batch_size):
        shape = (batch_size, self.ds.facts_per_seq)
        return self.ds._encode(rng.choice(self.entities, shape), rng.integers(0, self.ds.num_relations, shape))


def freeze_except_pool_values() -> optax.GradientTransformation:
    def update(updates, state, params=None):
        return jax.tree_util.tree_map_with_path(
            lambda p, u: u if _path_is(p, "pool", "values") else jnp.zeros_like(u), updates), state
    return optax.GradientTransformation(lambda _: optax.EmptyState(), update)


def _train(trainer: Trainer, state, sample: Callable, steps: int, seed: int,
           evals: Dict[str, Callable], every: int, revive: bool, log: List, offset: int = 0):
    rng = jax.random.PRNGKey(seed)
    np_rng = np.random.default_rng(seed)
    for step in range(1, steps + 1):
        rng, r1, r2 = jax.random.split(rng, 3)
        state, metrics, q = trainer.train_step(state, sample(np_rng, trainer.tcfg.batch_size), r1)
        if revive and trainer.mcfg.use_memory and step % trainer.tcfg.revive_every == 0 \
                and step <= trainer.tcfg.revive_until * steps:
            state, _ = trainer.revive(state, q, r2)
        if step % every == 0 or step == steps:
            row = {"step": offset + step, **{k: float(f(state.params)) for k, f in evals.items()}}
            log.append(row)
            print("   ", row, flush=True)
    return state


def retention(entities: int = 2048, steps_a: int = 2500, steps_b: int = 1200, every: int = 200) -> Dict:
    ds = FactDataset(num_entities=entities, num_relations=4)
    perm = np.random.default_rng(1).permutation(entities)
    A, B = FactSubset(ds, np.sort(perm[: entities // 2])), FactSubset(ds, np.sort(perm[entities // 2 :]))
    base = dict(vocab_size=ds.vocab_size, max_len=ds.seq_len)
    configs = {
        "memory": ModelConfig(**base),
        "dense_matched": ModelConfig(**base, use_memory=False, ffn_mult=9),
    }
    out: Dict = {"facts_A": len(A.fact_ids), "facts_B": len(B.fact_ids), "steps_A": steps_a, "steps_B": steps_b,
                 "curves": {}}

    def acc_fn(mcfg, subset):
        return lambda params: probe(mcfg, params, ds, subset.fact_ids)["correct"].mean()

    for kind, mcfg in configs.items():
        tcfg = TrainConfig(steps=steps_a)
        trainer = Trainer(mcfg, tcfg)
        state = trainer.init(jax.random.PRNGKey(0))
        log: List = []
        print(f"[retention] {kind}: phase A", flush=True)
        evals = {"acc_A": acc_fn(mcfg, A), "acc_B": acc_fn(mcfg, B)}
        state = _train(trainer, state, A.sample, steps_a, 0, evals, every, True, log)
        phase_b = {"dense_matched": ["full"], "memory": ["full", "pool_values_only"]}[kind]
        for mode in phase_b:
            tcfg_b = TrainConfig(steps=steps_b, warmup_steps=50)
            tr_b = Trainer(mcfg, tcfg_b)
            if mode == "pool_values_only":
                tr_b.optimizer = optax.chain(tr_b.optimizer, freeze_except_pool_values())
            st = state.replace(opt_state=tr_b.optimizer.init(state.params))
            log_b = [dict(r) for r in log]
            print(f"[retention] {kind}/{mode}: phase B", flush=True)
            _train(tr_b, st, B.sample, steps_b, 1, evals, every, False, log_b, offset=steps_a)
            out["curves"][f"{kind}/{mode}"] = log_b
    return out


# --------------------------------------------------------------------------
def _save(name: str, obj) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"wrote {os.path.join(RESULTS, name)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "analyze", "retention", "summary"])
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    if args.cmd == "train":
        train_all(args.parallel, args.only)
    elif args.cmd == "analyze":
        _save("analysis_main.json", analyze("main"))
        _save("dense_comparison.json", compare_dense())
        _save("capacity.json", capacity())
    elif args.cmd == "retention":
        _save("retention.json", retention())
    else:
        for f in sorted(os.listdir(RESULTS)):
            if f.endswith(".json"):
                print(f"== {f}")
                print(json.dumps(json.load(open(os.path.join(RESULTS, f))), indent=1)[:4000])


if __name__ == "__main__":
    main()
