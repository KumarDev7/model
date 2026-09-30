"""Look at what a trained model actually writes, not just the recall number.

Greedy generation (no teacher forcing) for
  * Q/A prompts ("Q: Where was X born? A:") for people trained with Q/A,
    people seen only in bios, and people never seen (set B),
  * bio prompts in a training wording and in wordings never trained,
  * held-out web text (continuations, to judge fluency),
with the pool as trained and with its reads shuffled. Each generated answer is
cut at the first period / newline and compared with the true value; wrong
answers are split into "another valid value of the same kind" (the model
knows the format but recalls the wrong fact) and "not a valid value".

    python -m experiments.inspect_outputs --data <fit dir> --out <results dir> --arm pool_4x
    python -m experiments.inspect_outputs --data <fit dir> --state <ckpt>/pool_4x.msgpack  # running training
"""

from __future__ import annotations

import argparse
import json
import os
import re

import jax
import jax.numpy as jnp
import numpy as np

from memory_pool_model.model import MemoryPoolLM

from . import facts_in_text as fit
from . import knowledge_study as ks
from . import text_study as ts


class Generator:
    def __init__(self, mcfg, params, length):
        self.mcfg, self.params, self.L = mcfg, params, length
        model = MemoryPoolLM(mcfg)

        def logits_fn(params, tokens, shuffled):  # shuffled: Python bool, one compiled function per mode
            return model.apply({"params": params}, tokens, shuffle_pool=shuffled)[0]

        self._logits = {m: jax.jit(lambda p, t, m=m: logits_fn(p, t, m)) for m in (False, True)}

    def greedy(self, prompts, n_new, shuffled=False, batch=128):
        """prompts: lists of token ids. Returns lists of n_new generated ids."""
        out = []
        for s in range(0, len(prompts), batch):
            chunk = prompts[s: s + batch]
            buf = np.zeros((batch, self.L), np.int32)
            lens = np.zeros(batch, np.int32)
            for i, p in enumerate(chunk):
                p = p[-(self.L - n_new):]
                buf[i, : len(p)] = p
                lens[i] = len(p)
            gen = np.zeros((batch, n_new), np.int32)
            for t in range(n_new):
                lg = np.asarray(self._logits[shuffled](self.params, jnp.asarray(buf)))
                nxt = lg[np.arange(batch), lens - 1].argmax(-1)
                gen[:, t] = nxt
                buf[np.arange(batch), lens] = nxt
                lens += 1
            out.extend(gen[: len(chunk)].tolist())
        return out


def first_answer(text):
    """The answer part of a generation: up to the first period, newline or new question."""
    return re.split(r"\.|\n|Q:", text, maxsplit=1)[0].strip()


def fact_items(people, templates, rel_filter=None):
    """(prompt text, true answer text, relation, person) for each fact x template."""
    items = []
    for p in people:
        for r in fit.RELS:
            if rel_filter and r not in rel_filter:
                continue
            for t in templates[r]:
                full = fit.render(t, p["name"], p[r])[:-1]  # drop the final period
                ans = f"{fit.article(p[r])} {p[r]}" if "{a} {v}" in t else p[r]
                prompt = full[: full.rindex(" " + ans)]
                items.append((prompt, ans, r, p))
    return items


def judge(items, gens, tok, values):
    rows, counts = [], {"exact": 0, "wrong_valid": 0, "invalid": 0}
    for (prompt, ans, r, p), g in zip(items, gens):
        text = tok.decode(g)
        got = first_answer(text)
        core = got.split(" ", 1)[1] if r == "job" and " " in got else got
        if got == ans:
            kind = "exact"
        elif core in values[r]:
            kind = "wrong_valid"
        else:
            kind = "invalid"
        counts[kind] += 1
        rows.append({"prompt": prompt, "truth": ans, "generated": text, "answer": got, "verdict": kind})
    n = max(len(items), 1)
    return {k: v / n for k, v in counts.items()}, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", help="results dir with ckpt/<arm>.msgpack (finished run)")
    ap.add_argument("--arm")
    ap.add_argument("--state", help="<save>.msgpack of a running training (uses <save>.state)")
    ap.add_argument("--people", type=int, default=200)
    ap.add_argument("--show", type=int, default=8)
    ap.add_argument("--gen_tokens", type=int, default=10)
    ap.add_argument("--web", type=int, default=6)
    ap.add_argument("--json", help="write summary + all samples here")
    a = ap.parse_args()
    facts, tok = ks.load_facts(a.data)
    if a.state:
        from .knowledge_verify import load_training_state

        mcfg, params, step = load_training_state(a.state)
        name = os.path.basename(a.state)
    else:
        mcfg, params, meta = ts.load(a.out, a.arm)
        step, name = meta.get("step", "final"), a.arm
    gen = Generator(mcfg, params, 64)
    rng = np.random.default_rng(1)
    nq = facts.get("qa_trained_people", 0)
    A = facts["A"]
    groups = {}
    if nq:
        groups["qa / trained with Q/A"] = (list(rng.permutation(A[:nq]))[: a.people], facts["qa_templates"])
        groups["qa / seen only in bios"] = (list(rng.permutation(A[nq:]))[: a.people], facts["qa_templates"])
        groups["qa / never seen (chance)"] = (facts["B"][: a.people], facts["qa_templates"])
    one_t = {r: facts["train_templates"][r][:1] for r in fit.RELS}
    groups["bio, training wording"] = (list(rng.permutation(A))[: a.people], one_t)
    groups["bio, unseen wording"] = (list(rng.permutation(A))[: a.people], {r: facts["test_templates"][r][:1] for r in fit.RELS})
    report = {"model": name, "step": step, "groups": {}}
    modes = (False, True) if mcfg.use_memory else (False,)
    print(f"=== {name} @ step {step}")
    for g, (people, templates) in groups.items():
        items = fact_items(people, templates)
        prompts = [tok.encode(pr).ids for pr, *_ in items]
        entry = {}
        for shuffled in modes:
            gens = gen.greedy(prompts, a.gen_tokens, shuffled=shuffled)
            summ, rows = judge(items, gens, tok, facts["values"])
            key = "pool_shuffled" if shuffled else "normal"
            entry[key] = summ
            if not shuffled:
                entry["samples"] = rows
            print(f"--- {g} [{key}]: exact {summ['exact']:.1%}, wrong-but-valid {summ['wrong_valid']:.1%}, "
                  f"invalid {summ['invalid']:.1%} (n={len(items)})")
            for row in (rows[: a.show] if not shuffled else rows[: max(2, a.show // 4)]):
                mark = {"exact": "OK ", "wrong_valid": "XX ", "invalid": "?? "}[row["verdict"]]
                print(f"   {mark}{row['prompt']!r:70.70s} -> {row['generated']!r:40.40s} (truth: {row['truth']})")
        report["groups"][g] = entry
    if a.web:
        val = np.load(os.path.join(a.data, "val.npy"), mmap_mode="r")
        eot = tok.token_to_id("<|endoftext|>")
        starts = np.where(np.asarray(val[:2_000_000]) == eot)[0][: a.web * 7: 7] + 1
        prompts = [list(map(int, val[s: s + 32])) for s in starts]
        gens = gen.greedy(prompts, 32)
        report["web"] = []
        print("--- held-out web text (32-token prompt -> 32 greedy tokens)")
        for p, g in zip(prompts, gens):
            pt, gt = tok.decode(p), tok.decode(g)
            report["web"].append({"prompt": pt, "continuation": gt})
            print(f"   PROMPT: {pt!r}\n   MODEL : {gt!r}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
