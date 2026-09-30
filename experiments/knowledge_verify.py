"""Is the knowledge in the pool, and does the backbone use it? (facts in text)

Runs on models trained by experiments/knowledge_study.py (fictional people's
facts written as bios inside Ultra-FineWeb text, see facts_in_text.py) and
answers five questions with tests nothing in training optimises for:

  1. recall      Fact recall with the pool read normally, with every read
                 returning another slot's vector (routing unchanged) and with
                 the pool removed; dense models of 1x / 2x / 4x size alongside.
  2. targeted    For each fact, zero only the pool vectors read while its
                 answer is predicted. Controls: zero another person's vectors
                 for the same relation (same count, same kind of context), or
                 as many random vectors. If the fact lives in its own vectors,
                 only the first breaks it, and other facts survive.
  3. usage       Collapse check under inference routing, on web text, on the
                 bios and on the recall prompts: vectors read at least once,
                 share with a fair share of reads, evenness, heaviest vector,
                 sub-keys used, read sharpness, routing temperature.
  4. reliance    Loss on web text, known people's bios and unknown people's
                 bios with the pool normal / shuffled / removed, and greedy
                 answers to "<name> was born in" for each condition.
  5. write       New people (set B) written into a trained model by updating
                 only the pool vectors: backbone, router and keys frozen. If
                 the frozen backbone then answers about B, it knows how to
                 read knowledge from the pool. Full fine-tuning (pool and
                 dense models) for comparison, and set-A forgetting.

    python -m experiments.knowledge_verify --data /content/ufw/fit --out experiments/results/knowledge
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from flax import serialization

from memory_pool_model.config import ModelConfig, TrainConfig
from memory_pool_model.generate import Generator
from memory_pool_model.model import MemoryPoolLM
from memory_pool_model.train import Trainer

from . import facts_in_text as fit
from . import knowledge_study as ks
from . import text_study as ts

L = 32  # probe length (prompt + answer)
MODES = ("normal", "shuffled", "removed")


# ------------------------------------------------------------------ probes
class Prober:
    """Teacher-forced exact match of an answer, with any set of pool
    vectors zeroed ("dead"), and the slots read at the answer positions."""

    def __init__(self, mcfg, params, dead_cap: int = 8192):
        self.mcfg, self.params, self.dead_cap = mcfg, params, dead_cap
        self.model = MemoryPoolLM(mcfg)
        self.N = mcfg.pool_size if mcfg.use_memory else 0

    @functools.partial(jax.jit, static_argnums=(0, 6))
    def _f(self, params, dead, inp, tgt, am, mode):
        if self.mcfg.use_memory:
            values = params["pool"]["values"].at[dead].set(0.0, mode="drop")
            params = {**params, "pool": {**params["pool"], "values": values}}
        logits, aux = self.model.apply({"params": params}, inp, pool_off=mode == "removed",
                                       shuffle_pool=mode == "shuffled")
        lp = jnp.take_along_axis(jax.nn.log_softmax(logits), tgt[..., None], -1)[..., 0]
        hit = jnp.all((logits.argmax(-1) == tgt) | (am == 0), -1)
        return hit, jnp.sum(lp * am, -1), aux.get("slots", []), aux.get("weights", [])

    def run(self, items, dead=None, mode="normal", batch=64, want_slots=False):
        """items: (prompt ids, answer ids, ...). Returns hits [n], logprob [n]
        and, with want_slots, per item an array [n_answer_tokens, layers, H, k]
        of slots read while predicting the answer, plus their weights."""
        if mode != "normal" and not self.mcfg.use_memory:
            return None
        d = np.full(self.dead_cap, self.N, np.int32)
        if dead is not None and len(dead):
            if len(dead) > self.dead_cap:
                raise ValueError("too many dead vectors")
            d[: len(dead)] = dead
        d = jnp.asarray(d)
        hits, lps, slots, weights = [], [], [], []
        for s in range(0, len(items), batch):
            chunk = items[s : s + batch]
            inp = np.zeros((batch, L), np.int32)
            tgt = np.zeros((batch, L), np.int32)
            am = np.zeros((batch, L), np.float32)
            spans = []
            for i, it in enumerate(chunk):
                pi, ai = it[0], it[1]
                seq = (pi + ai)[: L + 1]
                inp[i, : len(seq) - 1] = seq[:-1]
                tgt[i, : len(seq) - 1] = seq[1:]
                am[i, len(pi) - 1 : len(seq) - 1] = 1.0
                spans.append((len(pi) - 1, len(seq) - 1))
            h, lp, sl, w = self._f(self.params, d, inp, tgt, am, mode)
            hits.append(np.asarray(h)[: len(chunk)])
            lps.append(np.asarray(lp)[: len(chunk)])
            if want_slots and sl:
                sl = np.stack([np.asarray(x) for x in sl], 2)  # [B, T, layers, H, k]
                w = np.stack([np.asarray(x) for x in w], 2)
                for i, (a, b) in enumerate(spans):
                    slots.append(sl[i, a:b])
                    weights.append(w[i, a:b])
        out = {"hit": np.concatenate(hits), "logprob": np.concatenate(lps)}
        if want_slots:
            out["slots"], out["weights"] = slots, weights
        return out


def fact_items(tok, people, templates):
    """probe_items grouped by fact: {(person index, relation): [items]}."""
    groups = {}
    for pi_, p in enumerate(people):
        for it in fit.probe_items(tok, [p], templates):
            groups.setdefault((pi_, it[2]), []).append(it)
    return groups


# ------------------------------------------------------------------ 1. recall
def recall_table(pr: Prober, tok, facts, n_b: int = 300):
    res = {}
    for split, people in (("A_trained", facts["A"]), ("B_never_seen", facts["B"][:n_b])):
        for wording, templates in (("seen_wording", facts["train_templates"]),
                                   ("new_wording", facts["test_templates"])):
            items = fit.probe_items(tok, people, templates)
            rels = np.array([it[2] for it in items])
            row = {"items": len(items)}
            for mode in MODES:
                r = pr.run(items, mode=mode, batch=256)
                if r is None:
                    continue
                row[mode] = float(r["hit"].mean())
                row[f"{mode}_logprob"] = float(r["logprob"].mean())
                if mode == "normal":
                    row["by_relation"] = {rel: float(r["hit"][rels == rel].mean()) for rel in fit.RELS}
            res[f"{split}/{wording}"] = row
    # chance: the most common value of each relation, as a guess
    return res


# --------------------------------------------------------------- 2. targeted
def targeted_ablation(pr: Prober, tok, facts, n_facts=200, levels=(1, 4, None), n_collateral=24, seed=0):
    rng = np.random.default_rng(seed)
    k = pr.mcfg.top_k
    seen = fact_items(tok, facts["A"], facts["train_templates"])
    new = fact_items(tok, facts["A"], facts["test_templates"])
    keys = [key for key in seen if key in new]
    # facts recalled with every training wording
    order = rng.permutation(len(keys))
    chosen, reads = [], {}
    for j in order:
        key = keys[j]
        r = pr.run(seen[key], want_slots=True, batch=8)
        if r["hit"].all():
            chosen.append(key)
            reads[key] = (r["slots"], r["weights"])
        if len(chosen) >= n_facts:
            break
    by_rel = {}
    for key in chosen:
        by_rel.setdefault(key[1], []).append(key)

    def rows_of(key, r):
        sl = reads[key][0]
        return np.unique(np.concatenate([s[..., :r].reshape(-1) for s in sl]))

    res = {"n_facts": len(chosen), "levels": []}
    for lev in levels:
        r = k if lev is None else lev
        acc = {c: {"seen": [], "new": []} for c in ("own", "other_person_same_relation", "random")}
        n_rows, overlap, coll_before, coll_after = [], [], [], []
        for key in chosen:
            own = rows_of(key, r)
            peers = [q for q in by_rel[key[1]] if q[0] != key[0]]
            other = rows_of(peers[rng.integers(len(peers))], r) if peers else own
            n_rows.append(len(own))
            overlap.append(len(np.intersect1d(own, other)) / max(len(own), 1))
            rand = rng.choice(pr.N, len(own), replace=False)
            # collateral: one seen-wording item of random other facts
            others = [keys[i] for i in rng.choice(len(keys), n_collateral, replace=False) if keys[i] != key]
            coll_items = [seen[q][0] for q in others]
            for cond, dead in (("own", own), ("other_person_same_relation", other), ("random", rand)):
                items = seen[key] + new[key] + (coll_items if cond == "own" else [])
                h = pr.run(items, dead=dead, batch=64)["hit"]
                ns = len(seen[key])
                acc[cond]["seen"].append(h[:ns].mean())
                acc[cond]["new"].append(h[ns : ns + len(new[key])].mean())
                if cond == "own":
                    coll_after.append(h[ns + len(new[key]) :].mean())
            coll_before.append(pr.run(coll_items, batch=64)["hit"].mean())
        new_before = np.mean([pr.run(new[key], batch=8)["hit"].mean() for key in chosen]) if lev == levels[0] else None
        row = {"vectors_zeroed_per_head_per_read": r,
               "vectors_zeroed_mean": float(np.mean(n_rows)),
               "share_of_pool": float(np.mean(n_rows) / pr.N),
               "overlap_own_vs_other_person": float(np.mean(overlap)),
               "recall_after": {c: {w: float(np.mean(v)) for w, v in d.items()} for c, d in acc.items()},
               "other_facts_before": float(np.mean(coll_before)),
               "other_facts_after_own_zeroed": float(np.mean(coll_after))}
        if new_before is not None:
            res["new_wording_recall_before"] = float(new_before)
        res["levels"].append(row)
        print("  targeted", json.dumps(row), flush=True)
    res["seen_wording_recall_before"] = 1.0  # facts were chosen this way
    return res


# ------------------------------------------------------------------ 3. usage
def usage_block(counts, top1, n_sub):
    return ts.usage(counts, n_sub, top1)


def pool_usage(pr: Prober, tok, facts, split_windows, n_facts_items=4000):
    """Inference-routing usage per memory layer on text splits and on the
    answer positions of the recall prompts."""
    mcfg, params = pr.mcfg, pr.params
    fwd, _ = ts.make_forward(mcfg)
    res = {"temperature": float(np.exp(np.clip(float(params["pool"]["log_temperature"]),
                                               np.log(mcfg.min_temperature), np.log(mcfg.max_temperature))))}
    for name, split in split_windows.items():
        counts = [np.zeros(mcfg.pool_size) for _ in mcfg.memory_layers]
        top1 = [[] for _ in mcfg.memory_layers]
        for b in split:
            _, _, slots, weights = fwd(params, b["inputs"], b["targets"])
            for i in range(len(counts)):
                counts[i] += np.bincount(np.asarray(slots[i]).ravel(), minlength=mcfg.pool_size)
                top1[i].append(float(np.asarray(weights[i]).max(-1).mean()))
        res[name] = [usage_block(c, np.mean(t), mcfg.n_sub_keys) for c, t in zip(counts, top1)]
        allc = sum(counts)
        res[name + "_all_layers"] = usage_block(allc, np.mean([np.mean(t) for t in top1]), mcfg.n_sub_keys)
    items = fit.probe_items(tok, facts["A"], facts["train_templates"])
    items = [items[i] for i in np.random.default_rng(0).permutation(len(items))[:n_facts_items]]
    r = pr.run(items, want_slots=True, batch=256)
    sl = np.concatenate([s.reshape(-1, *s.shape[1:]) for s in r["slots"]])  # [n, layers, H, k]
    w = np.concatenate([x.reshape(-1, *x.shape[1:]) for x in r["weights"]])
    res["fact_answers"] = []
    for li in range(sl.shape[1]):
        c = np.bincount(sl[:, li].ravel(), minlength=mcfg.pool_size).astype(float)
        u = usage_block(c, w[:, li].max(-1).mean(), mcfg.n_sub_keys)
        pr_ = 1.0 / np.sum(w[:, li] ** 2, -1)
        u["effective_vectors_per_head"] = float(pr_.mean())
        res["fact_answers"].append(u)
    return res


# --------------------------------------------------------------- 4. reliance
def make_mode_loss(mcfg):
    model = MemoryPoolLM(mcfg)

    @functools.partial(jax.jit, static_argnums=(3,))
    def f(params, inputs, targets, mode):
        logits, _ = model.apply({"params": params}, inputs, pool_off=mode == "removed",
                                shuffle_pool=mode == "shuffled")
        return -jnp.take_along_axis(jax.nn.log_softmax(logits), targets[..., None], -1)[..., 0].mean()

    return f


def reliance(pr: Prober, tok, facts, split_arrays, n_windows=256, n_people=40):
    mcfg, params = pr.mcfg, pr.params
    f = make_mode_loss(mcfg)
    res = {"loss": {}}
    for name, arr in split_arrays.items():
        row = {}
        for mode in MODES if mcfg.use_memory else ("normal",):
            ls = [float(f(params, b["inputs"], b["targets"], mode)) for b in ts.windows(arr, n_windows)]
            row[mode] = float(np.mean(ls))
            row[f"{mode}_ppl"] = float(np.exp(row[mode]))
        res["loss"][name] = row
    # greedy answers
    people = facts["A"][:n_people]
    res["greedy_born_in"] = {}
    for mode in MODES if mcfg.use_memory else ("normal",):
        gen = Generator(mcfg, params, batch_size=1, shuffle_pool=mode == "shuffled", pool_off=mode == "removed")
        right, samples = 0, []
        for p in people:
            ids = np.asarray([tok.encode(f"{p['name']} was born in").ids], np.int32)
            out = tok.decode(gen.generate(ids, n_new=4)["tokens"][0].tolist()).strip()
            right += out.split(" ")[0].strip(".,") == p["born"]
            if len(samples) < 8:
                samples.append({"name": p["name"], "truth": p["born"], "generated": out})
        res["greedy_born_in"][mode] = {"correct": right / len(people), "samples": samples}
    return res


# ------------------------------------------------------------------ 5. write
def write_test(out, data, arm, steps, facts, tok, n_eval_a=500):
    """Add set B to a trained model; see the module docstring."""
    mcfg, params, meta = ts.load(out, arm)
    b_mix = np.load(os.path.join(data, "b_mix.npy"))
    a_docs = np.load(os.path.join(data, "a_docs.npy"))
    replay = fit.shuffle_mix(np.random.default_rng(1), b_mix, a_docs[: len(b_mix) // 4])
    val = np.load(os.path.join(data, "val.npy"), mmap_mode="r")
    base = TrainConfig(**{k: v for k, v in meta["train"].items() if k in TrainConfig.__dataclass_fields__})
    base = dataclasses.replace(base, steps=steps, warmup_steps=50, revive_every=0, checkpoint_every=0)
    modes = {"full_finetune": (base, b_mix)}
    if mcfg.use_memory:
        modes["pool_vectors_only"] = (dataclasses.replace(base, pool_values_only=True), b_mix)
        modes["pool_vectors_only_with_replay"] = (dataclasses.replace(base, pool_values_only=True), replay)
    items_a = fit.probe_items(tok, facts["A"][: n_eval_a // 4], facts["train_templates"])
    items_b = fit.probe_items(tok, facts["B"], facts["train_templates"])
    items_b_new = fit.probe_items(tok, facts["B"], facts["test_templates"])
    f = make_mode_loss(mcfg)

    def summary(p):
        pr = Prober(mcfg, p)
        row = {}
        for name, items in (("A", items_a), ("B", items_b), ("B_new_wording", items_b_new)):
            row[name] = {m: float(pr.run(items, mode=m, batch=256)["hit"].mean())
                         for m in (MODES if mcfg.use_memory else ("normal",))}
        row["heldout_ppl"] = float(np.exp(np.mean([float(f(p, b["inputs"], b["targets"], "normal"))
                                                   for b in ts.windows(val, 128)])))
        return row

    res = {"arm": arm, "steps": steps, "before": summary(params), "after": {}}
    for mode, (tcfg, arr) in modes.items():
        new, secs = ks.finetune(mcfg, params, tcfg, arr, steps)
        row = summary(new)
        row["seconds"] = secs
        if mode.startswith("pool_vectors_only"):
            # everything but the value table must be bit-identical
            same = jax.tree_util.tree_map(lambda a, b: bool(np.array_equal(np.asarray(a), np.asarray(b))),
                                          {k: v for k, v in params.items() if k != "pool"},
                                          {k: v for k, v in new.items() if k != "pool"})
            row["backbone_unchanged"] = all(jax.tree_util.tree_leaves(same)) and all(
                np.array_equal(np.asarray(params["pool"][k]), np.asarray(new["pool"][k]))
                for k in params["pool"] if k != "values")
            changed = np.any(np.asarray(params["pool"]["values"]) != np.asarray(new["pool"]["values"]), axis=1)
            row["pool_vectors_changed"] = float(changed.mean())
        res["after"][mode] = row
        print("  write", mode, json.dumps(row), flush=True)
    return res


# ----------------------------------------------------------------- monitor
def load_training_state(save_path):
    """Params and step of the resumable checkpoint <save_path>.state of a
    run that may still be training (configs from <save_path>.config.json)."""
    cfg = json.load(open(save_path + ".config.json"))
    m = dict(cfg["model"])
    m["memory_layers"] = tuple(m["memory_layers"])
    mcfg = ModelConfig(**{**m, "pool_location": "device", "host_pool": ""})
    t = {k: v for k, v in cfg["train"].items() if k in TrainConfig.__dataclass_fields__}
    tcfg = TrainConfig(**{**t, "data_parallel": False})
    template = Trainer(mcfg, tcfg).init(jax.random.PRNGKey(0))
    with open(save_path + ".state", "rb") as f:
        state = serialization.from_bytes(template, f.read())
    return mcfg, jax.tree_util.tree_map(jnp.asarray, state.params), int(state.step)


def monitor(data, save_path, log_path, n_people=300, n_windows=64):
    """One snapshot of a running training: fact recall (pool normal /
    shuffled / removed), held-out loss and pool usage. Appends a JSON line."""
    facts, tok = ks.load_facts(data)
    mcfg, params, step = load_training_state(save_path)
    pr = Prober(mcfg, params)
    row = {"step": step, "time": time.time()}
    people = [facts["A"][i] for i in np.random.default_rng(0).permutation(len(facts["A"]))[:n_people]]
    for wording, templates in (("seen", facts["train_templates"]), ("new", facts["test_templates"])):
        items = fit.probe_items(tok, people, templates)
        for mode in MODES if mcfg.use_memory else ("normal",):
            row[f"recall_{wording}_{mode}"] = float(pr.run(items, mode=mode, batch=128)["hit"].mean())
    nq = facts.get("qa_trained_people", 0)
    if nq:  # Q/A: people trained with Q/A vs people only seen in bios
        for which, grp in (("trained", facts["A"][:nq][:n_people]), ("heldout", facts["A"][nq:][:n_people])):
            items = fit.probe_items(tok, grp, facts["qa_templates"])
            for mode in MODES if mcfg.use_memory else ("normal",):
                row[f"qa_{which}_{mode}"] = float(pr.run(items, mode=mode, batch=128)["hit"].mean())
        items = fit.probe_items(tok, facts["B"][:n_people], facts["qa_templates"])
        row["qa_never_seen"] = float(pr.run(items, batch=128)["hit"].mean())
    items_b = fit.probe_items(tok, facts["B"][:n_people // 2], facts["train_templates"])
    row["recall_never_seen"] = float(pr.run(items_b, batch=128)["hit"].mean())
    val = np.load(os.path.join(data, "val.npy"), mmap_mode="r")
    f = make_mode_loss(mcfg)
    for mode in MODES if mcfg.use_memory else ("normal",):
        row[f"heldout_loss_{mode}"] = float(np.mean([float(f(params, b["inputs"], b["targets"], mode))
                                                     for b in ts.windows(val, n_windows)]))
    if mcfg.use_memory:
        u = pool_usage(pr, tok, facts, {"web_heldout": ts.windows(val, n_windows)}, n_facts_items=1000)
        row["temperature"] = u["temperature"]
        row["usage_web_all_layers"] = u["web_heldout_all_layers"]
        row["usage_fact_answers"] = u["fact_answers"]
    with open(log_path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    return row


# -------------------------------------------------------------------- main
def verify_arm(data, out, arm, facts, tok, n_targeted, n_windows):
    mcfg, params, meta = ts.load(out, arm)
    pr = Prober(mcfg, params)
    t0 = time.time()
    val = np.load(os.path.join(data, "val.npy"), mmap_mode="r")
    a_docs = np.load(os.path.join(data, "a_docs.npy"), mmap_mode="r")
    b_docs = np.load(os.path.join(data, "b_docs.npy"), mmap_mode="r")
    row = {"arm": arm, "flags": " ".join(ks.ARMS.get(arm, [])),
           "params_total": int(sum(x.size for x in jax.tree_util.tree_leaves(params))),
           "params_on_accelerator_without_pool_values": int(
               sum(x.size for x in jax.tree_util.tree_leaves(params))
               - (params["pool"]["values"].size if mcfg.use_memory else 0)),
           "train_seconds": meta.get("train_seconds"), "history": meta["history"]}
    row["recall"] = recall_table(pr, tok, facts)
    print(arm, "recall", json.dumps(row["recall"])[:1500], flush=True)
    splits = {"web_heldout": val, "bios_known_people": a_docs, "bios_unknown_people": b_docs}
    row["reliance"] = reliance(pr, tok, facts, splits, n_windows)
    print(arm, "reliance", json.dumps(row["reliance"]["loss"]), flush=True)
    if mcfg.use_memory:
        row["usage"] = pool_usage(pr, tok, facts, {
            "web_heldout": ts.windows(val, n_windows), "bios_known_people": ts.windows(a_docs, n_windows)})
        print(arm, "usage", json.dumps(row["usage"])[:1500], flush=True)
        row["targeted"] = targeted_ablation(pr, tok, facts, n_targeted)
    row["verify_seconds"] = time.time() - t0
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="*", default=None, help="default: every trained arm")
    ap.add_argument("--targeted", type=int, default=200, help="facts in the targeted ablation")
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--write_arms", nargs="*", default=[], help="arms for the write test")
    ap.add_argument("--write_steps", type=int, default=1500)
    ap.add_argument("--skip_verify", action="store_true")
    ap.add_argument("--monitor", default=None, help="save path of a running training: log one snapshot and exit")
    ap.add_argument("--monitor_log", default=None)
    ap.add_argument("--max_people", type=int, default=0,
                    help="probe only this many trained people (random, fixed seed); 0 = all")
    ap.add_argument("--suffix", default="", help="output files knowledge_verify<suffix>.json / knowledge_write<suffix>.json")
    a = ap.parse_args()
    if a.monitor:
        monitor(a.data, a.monitor, a.monitor_log or a.monitor + ".monitor.jsonl")
        return
    facts, tok = ks.load_facts(a.data)
    if a.max_people and len(facts["A"]) > a.max_people:
        keep = np.sort(np.random.default_rng(0).permutation(len(facts["A"]))[: a.max_people])
        facts = {**facts, "A": [facts["A"][i] for i in keep]}
    arms = a.arms or [x for x in ks.ARMS if os.path.exists(os.path.join(a.out, "ckpt", x + ".msgpack.json"))]
    if not a.skip_verify:
        rows = []
        for arm in arms:
            rows.append(verify_arm(a.data, a.out, arm, facts, tok, a.targeted, a.windows))
            ks.save(a.out, f"knowledge_verify{a.suffix}.json", rows)
    if a.write_arms:
        rows = [write_test(a.out, a.data, arm, a.write_steps, facts, tok) for arm in a.write_arms]
        ks.save(a.out, f"knowledge_write{a.suffix}.json", rows)


if __name__ == "__main__":
    main()
