"""Summarise a training run from <save>.metrics.jsonl (and .samples.jsonl).

    python -m memory_pool_model.metrics_report runs/x/model.msgpack.metrics.jsonl
    python -m memory_pool_model.metrics_report ... --brief          # a few lines, for status checks
    python -m memory_pool_model.metrics_report ... --plot run.png   # curves (needs matplotlib)

Steps run twice (training resumed from an earlier checkpoint) keep their
last record.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np


def load(path: str):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:  # a line cut by a crash
                    pass
    train: Dict[int, dict] = {}
    for r in recs:
        if r.get("type") == "train":
            train[r["step"]] = r
    by_type: Dict[str, List[dict]] = {}
    for r in recs:
        by_type.setdefault(r.get("type", "?"), []).append(r)
    return [train[s] for s in sorted(train)], by_type


def _num(v):
    return float(v) if isinstance(v, (int, float)) else float("nan")


def series(train, key):
    return np.array([_num(r.get(key)) for r in train])


def _fmt_tokens(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


def last_samples(path: str):
    if not os.path.exists(path):
        return None
    last = None
    with open(path) as f:
        for line in f:
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                pass
    return last


def brief(train, by_type, samples) -> str:
    if not train:
        return "no training steps logged yet"
    r = train[-1]
    window = train[-100:]
    loss = np.nanmean(series(window, "loss"))
    steady = [x for x in window if not x.get("first")]
    tps = np.nanmedian(series(steady, "tokens_per_s")) if steady else float("nan")
    out = [f"step {r['step']}  loss(last {len(window)})={loss:.4f}  grad_norm={_num(r.get('grad_norm')):.3g}  "
           f"lr={_num(r.get('lr')):.3g}  {tps / 1e3:.0f}k tok/s"
           + (f"  mfu={np.nanmedian(series(steady, 'mfu')):.2f}" if steady and "mfu" in steady[-1] else "")]
    if "data" in r:
        out.append(f"data   {r['data']['shard']}  {_fmt_tokens(r['data']['tokens'])} tokens read, "
                   f"waited {r['data']['wait_s']:.0f}s for data")
    ev = by_type.get("eval", [])
    if ev:
        e = ev[-1]
        out.append("eval   step {}: ".format(e["step"]) + " ".join(
            f"{k}={e[k]:.4g}" for k in ("ce", "acc", "acc_nopool", "acc_shuffled_pool", "ce_shuffled_pool") if k in e))
    bad = sum(x.get("nonfinite", False) for x in train)
    spikes = [x["step"] for x in train if "grad_spike" in x]
    out.append(f"health non-finite steps: {bad}, grad spikes (>5x mean): {len(spikes)}"
               + (f" (last at {spikes[-1]})" if spikes else ""))
    if samples:
        s = samples["samples"][0]
        out.append(f"sample step {samples['step']}: {s['prompt']!r} -> {s['text'][:100]!r}")
    return "\n".join(out)


def report(train, by_type, samples) -> str:
    out = []
    starts, ends = by_type.get("start", []), by_type.get("end", [])
    if not train:
        return "no training steps logged"
    steps = np.array([r["step"] for r in train])
    out.append(f"== run: steps {steps[0]}-{steps[-1]} logged ({len(train)} records), "
               f"{len(starts)} session(s), {sum(s.get('resumed', False) for s in starts)} resumed; "
               f"last outcome: {ends[-1]['outcome'] if ends else 'still running or killed'}")
    if starts:
        s0 = starts[-1]
        out.append(f"   {s0.get('devices')} x {s0.get('device_kind')}, {s0.get('params', 0):,} params "
                   f"({s0.get('backbone_params', 0):,} backbone), {s0.get('flops_per_token', 0) / 1e9:.2f} GFLOP/token")
    out.append(f"   tokens trained: {_fmt_tokens(train[-1].get('tokens', 0))}")

    loss = series(train, "loss")
    out.append("== loss (mean over each tenth of the logged steps)")
    for chunk in np.array_split(np.arange(len(train)), min(10, len(train))):
        if len(chunk):
            out.append(f"   steps {steps[chunk[0]]:>7}-{steps[chunk[-1]]:<7} loss {np.nanmean(loss[chunk]):.4f}  "
                       f"grad_norm {np.nanmean(series(train, 'grad_norm')[chunk]):.3g}  "
                       f"logit_max {np.nanmax(series(train, 'logit_max')[chunk]) if 'logit_max' in train[0] else float('nan'):.1f}")
    ev = by_type.get("eval", [])
    if ev:
        out.append("== eval (held-out)")
        keys = [k for k in ("ce", "acc", "acc_nopool", "acc_shuffled_pool", "ce_shuffled_pool", "pool_coverage")
                if k in ev[-1]]
        out.append("   " + "step".rjust(8) + "".join(k.rjust(18) for k in keys))
        seen = {}
        for e in ev:
            seen[e["step"]] = e
        for st in sorted(seen):
            out.append("   " + str(st).rjust(8) + "".join(f"{seen[st][k]:18.4f}" for k in keys))

    steady = [r for r in train if not r.get("first")]
    if steady:
        st = series(steady, "step_s")
        out.append("== speed")
        out.append(f"   step {np.nanmedian(st) * 1e3:.1f} ms median, {np.nanpercentile(st, 95) * 1e3:.1f} ms p95; "
                   f"{np.nanmedian(series(steady, 'tokens_per_s')) / 1e3:.0f}k tokens/s"
                   + (f"; MFU {np.nanmedian(series(steady, 'mfu')):.3f}" if "mfu" in steady[-1] else ""))
        out.append(f"   host data time {np.nanmedian(series(steady, 'data_s')) * 1e3:.2f} ms/step median")
        pause = np.nansum(series(train, "pause_s"))
        busy = np.nansum(st)
        out.append(f"   training paused {pause:.0f}s in total (eval, checkpoint copies, sample copies) "
                   f"= {100 * pause / max(pause + busy, 1e-9):.2f}% of the time")
        if "data" in train[-1]:
            out.append(f"   waited for the data stream: {train[-1]['data']['wait_s']:.0f}s")
        hbm = series(train, "hbm_peak_gb")
        if np.isfinite(hbm).any():
            out.append(f"   device memory peak {np.nanmax(hbm):.2f} GB of {np.nanmax(series(train, 'hbm_limit_gb')):.2f} GB")
    ck = by_type.get("checkpoint", [])
    if ck:
        out.append(f"== checkpoints: {len(ck)}, written in {np.mean([c.get('write_s', 0) for c in ck]):.1f}s on average "
                   f"(background), training waited {np.mean([c.get('copy_s', 0) + c.get('waited_s', 0) for c in ck]):.2f}s "
                   f"on average for the copy")

    out.append("== stability")
    bad = [r["step"] for r in train if r.get("nonfinite")]
    out.append(f"   non-finite steps: {len(bad)}" + (f" (first at {bad[0]})" if bad else ""))
    spikes = [r for r in train if "grad_spike" in r]
    out.append(f"   grad spikes (>5x running mean): {len(spikes)}")
    for r in sorted(spikes, key=lambda r: -r["grad_spike"])[:5]:
        groups = sorted(((k, v) for k, v in r.items() if k.startswith("gn_") and isinstance(v, (int, float))),
                        key=lambda kv: -kv[1])[:3]
        out.append(f"     step {r['step']}: grad_norm {r['grad_norm']:.3g} ({r['grad_spike']:.1f}x); largest: "
                   + ", ".join(f"{k[3:]}={v:.3g}" for k, v in groups))
    gn = series(train, "grad_norm")
    if np.isfinite(gn).any():
        i = int(np.nanargmax(gn))
        out.append(f"   max grad_norm {gn[i]:.3g} at step {steps[i]}")
    if "update_norm" in train[-1] and "param_norm" in train[-1]:
        ratio = series(train, "update_norm") / series(train, "param_norm")
        out.append(f"   update/param norm ratio: median {np.nanmedian(ratio):.2e}, last {ratio[-1]:.2e}")
    pool_keys = [k for k in ("slot_active_ema", "subkey_spread", "slot_spread_ema", "top1_weight", "pick_agreement",
                             "temperature") if k in train[-1]]
    if pool_keys:
        out.append("== pool (first -> last)")
        out.append("   " + "  ".join(f"{k} {_num(train[0].get(k)):.3g}->{_num(train[-1].get(k)):.3g}" for k in pool_keys))
    smp = by_type.get("sample_started", [])
    if smp or samples:
        done = by_type.get("sample_done", [])
        out.append(f"== samples: {len(smp)} started, {len(by_type.get('sample_skipped', []))} skipped, "
                   f"{sum(d.get('returncode') == 0 for d in done)} finished")
        if samples:
            out.append(f"   last snapshot (step {samples['step']}, same text with the pool shuffled: "
                       f"{samples.get('same_with_pool_shuffled')} of {len(samples['samples'])}):")
            for s in samples["samples"][:4]:
                out.append(f"     {s['prompt']!r} -> {s['text'][:120]!r}")
    return "\n".join(out)


def plot(train, by_type, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.array([r["step"] for r in train])
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(steps, series(train, "loss"), lw=0.5, label="train loss")
    ev = by_type.get("eval", [])
    if ev:
        ax[0, 0].plot([e["step"] for e in ev], [e["ce"] for e in ev], "o-", label="held-out ce")
    ax[0, 0].legend()
    ax[0, 1].semilogy(steps, series(train, "grad_norm"), lw=0.5)
    ax[0, 1].set_title("grad_norm")
    ax[1, 0].plot(steps, series(train, "lr"))
    ax[1, 0].set_title("lr")
    ax[1, 1].plot(steps, series(train, "tokens_per_s") / 1e3, lw=0.5)
    ax[1, 1].set_title("k tokens/s")
    for a in ax.flat:
        a.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metrics")
    ap.add_argument("--samples", help="default: the .samples.jsonl next to the metrics file")
    ap.add_argument("--brief", action="store_true")
    ap.add_argument("--plot")
    a = ap.parse_args()
    train, by_type = load(a.metrics)
    samples = last_samples(a.samples or a.metrics.replace(".metrics.jsonl", ".samples.jsonl"))
    print(brief(train, by_type, samples) if a.brief else report(train, by_type, samples))
    if a.plot:
        plot(train, by_type, a.plot)
        print(f"plot: {a.plot}")


if __name__ == "__main__":
    main()
