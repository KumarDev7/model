"""Training loop with anti-collapse regularisation and dead-key revival.

Usage:
    python -m memory_pool_model.train --task facts --steps 3000
    python -m memory_pool_model.train --task facts --use_memory false   # baseline
    python -m memory_pool_model.train --task text --text_path input.txt
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization, struct

from .config import ModelConfig, TrainConfig
from .data import FactDataset, TextDataset
from .memory import key_diversity_loss, revive_dead_keys
from .model import MemoryPoolLM

QUERY_SAMPLE = 2048  # recent queries kept for reviving dead sub-keys


@struct.dataclass
class TrainState:
    step: jax.Array
    params: Any
    opt_state: Any
    subkey_usage: jax.Array  # [H, 2, n] EMA of sub-key selection frequency
    slot_usage: jax.Array  # [N] EMA of slot selection frequency


def _path_is(path, *names) -> bool:
    keys = [getattr(p, "key", None) for p in path]
    return keys[: len(names)] == list(names)


def make_optimizer(tcfg: TrainConfig) -> optax.GradientTransformation:
    schedule = optax.warmup_cosine_decay_schedule(
        0.0, tcfg.lr, tcfg.warmup_steps, max(tcfg.steps, tcfg.warmup_steps + 1), tcfg.lr * 0.1
    )

    def decay_mask(params):
        # Decay backbone matrices only; decaying the pool would erase knowledge
        # stored in slots that simply weren't fetched in this batch.
        return jax.tree_util.tree_map_with_path(
            lambda p, x: x.ndim >= 2 and not _path_is(p, "pool"), params
        )

    def scale_pool_values(mult: float) -> optax.GradientTransformation:
        def update(updates, state, params=None):
            updates = jax.tree_util.tree_map_with_path(
                lambda p, u: u * mult if _path_is(p, "pool", "values") else u, updates
            )
            return updates, state

        return optax.GradientTransformation(lambda _: optax.EmptyState(), update)

    return optax.chain(
        optax.clip_by_global_norm(tcfg.grad_clip),
        optax.adamw(schedule, weight_decay=tcfg.weight_decay, mask=decay_mask),
        scale_pool_values(tcfg.pool_lr_mult),
    )


def usage_stats(usage: jax.Array) -> Dict[str, jax.Array]:
    """Spread of a usage distribution: 1.0 = perfectly uniform, 1/N = collapsed."""
    p = usage / jnp.maximum(usage.sum(axis=-1, keepdims=True), 1e-9)
    entropy = -jnp.sum(jnp.where(p > 0, p * jnp.log(p), 0.0), axis=-1)
    n = usage.shape[-1]
    return {
        "spread": jnp.exp(entropy) / n,
        "active": jnp.mean(p > 0.1 / n, axis=-1),
    }


class Trainer:
    def __init__(self, mcfg: ModelConfig, tcfg: TrainConfig):
        self.mcfg, self.tcfg = mcfg, tcfg
        self.model = MemoryPoolLM(mcfg)
        self.optimizer = make_optimizer(tcfg)
        self.train_step = jax.jit(self._train_step)
        self.eval_step = jax.jit(self._eval_step)
        self.revive = jax.jit(self._revive)

    # ------------------------------------------------------------------ init
    def init(self, rng: jax.Array) -> TrainState:
        dummy = jnp.zeros((1, self.mcfg.max_len), jnp.int32)
        params = self.model.init({"params": rng}, dummy)["params"]
        m = self.mcfg
        return TrainState(
            step=jnp.zeros((), jnp.int32),
            params=params,
            opt_state=self.optimizer.init(params),
            subkey_usage=jnp.full((m.pool_heads, 2, m.n_sub_keys), 1.0 / m.n_sub_keys),
            slot_usage=jnp.full((m.pool_size,), 1.0 / m.pool_size),
        )

    # ------------------------------------------------------------------ loss
    def _loss(self, params, batch, rng, train: bool):
        r_route, r_drop = jax.random.split(rng)
        logits, aux = self.model.apply(
            {"params": params},
            batch["inputs"],
            train=train,
            rngs={"routing": r_route, "dropout": r_drop},
        )
        ce = optax.softmax_cross_entropy_with_integer_labels(logits, batch["targets"])
        mask = batch["mask"]
        denom = jnp.maximum(mask.sum(), 1.0)
        ce = (ce * mask).sum() / denom
        acc = ((logits.argmax(-1) == batch["targets"]) * mask).sum() / denom

        loss = ce
        metrics = {"ce": ce, "acc": acc}
        if self.mcfg.use_memory:
            div = key_diversity_loss(params["pool"]["sub_keys"])
            loss = loss + self.tcfg.balance_coef * aux["balance_loss"]
            loss = loss + self.tcfg.key_diversity_coef * div
            metrics.update(
                balance_loss=aux["balance_loss"],
                key_diversity=div,
                temperature=aux["temperature"],
            )
        metrics["loss"] = loss
        return loss, (metrics, aux)

    # ------------------------------------------------------------ train step
    def _train_step(self, state: TrainState, batch, rng):
        r_loss, r_sample = jax.random.split(rng)
        grad_fn = jax.value_and_grad(self._loss, has_aux=True)
        (_, (metrics, aux)), grads = grad_fn(state.params, batch, r_loss, True)
        updates, opt_state = self.optimizer.update(grads, state.opt_state, state.params)
        params = optax.apply_updates(state.params, updates)
        metrics["grad_norm"] = optax.tree.norm(grads)

        subkey_usage, slot_usage = state.subkey_usage, state.slot_usage
        queries = jnp.zeros((0,))
        if self.mcfg.use_memory:
            d = self.tcfg.usage_ema_decay
            sk = aux["subkey_counts"]
            sk = sk / sk.sum(axis=-1, keepdims=True)
            subkey_usage = d * subkey_usage + (1 - d) * sk
            slot_usage = d * slot_usage + (1 - d) * aux["slot_counts"] / aux["slot_counts"].sum()
            pick = jax.random.randint(r_sample, (QUERY_SAMPLE,), 0, aux["queries"].shape[0])
            queries = aux["queries"][pick]
            batch_stats = usage_stats(aux["slot_counts"])
            ema_stats = usage_stats(slot_usage)
            metrics.update(
                slot_spread_batch=batch_stats["spread"],
                slot_spread_ema=ema_stats["spread"],
                slot_active_ema=ema_stats["active"],
                subkey_spread=usage_stats(subkey_usage)["spread"].mean(),
            )

        new_state = state.replace(
            step=state.step + 1,
            params=params,
            opt_state=opt_state,
            subkey_usage=subkey_usage,
            slot_usage=slot_usage,
        )
        return new_state, metrics, queries

    # --------------------------------------------------------------- revive
    def _revive(self, state: TrainState, queries, rng):
        params = state.params
        new_keys, new_usage, dead = revive_dead_keys(
            params["pool"]["sub_keys"],
            state.subkey_usage,
            queries,
            rng,
            self.tcfg.revive_threshold,
        )
        params = {**params, "pool": {**params["pool"], "sub_keys": new_keys}}

        # Clear Adam moments of revived keys so stale momentum can't drag
        # them straight back to where they died.
        mask = jax.tree_util.tree_map_with_path(
            lambda p, x: dead[..., None] if _path_is(p, "pool", "sub_keys") else jnp.zeros((), bool),
            params,
        )
        opt_state = optax.tree_utils.tree_map_params(
            self.optimizer,
            lambda x, m: jnp.where(m, jnp.zeros_like(x), x),
            state.opt_state,
            mask,
            transform_non_params=lambda x: x,
        )
        new_state = state.replace(params=params, opt_state=opt_state, subkey_usage=new_usage)
        return new_state, dead.sum()

    # ----------------------------------------------------------------- eval
    def _eval_step(self, params, batch):
        _, (metrics, aux) = self._loss(params, batch, jax.random.PRNGKey(0), False)
        slot_counts = aux.get("slot_counts", jnp.zeros((1,)))
        return metrics, slot_counts, batch["mask"].sum()

    def evaluate(self, params, dataset, batch_size: int) -> Dict[str, float]:
        totals = {"ce": 0.0, "acc": 0.0}
        n = 0.0
        slot_hits = None
        for batch in dataset.eval_batches(batch_size):
            metrics, slot_counts, count = self.eval_step(params, batch)
            count = float(count)
            for k in totals:
                totals[k] += float(metrics[k]) * count
            n += count
            slot_hits = slot_counts if slot_hits is None else slot_hits + slot_counts
        out = {k: v / max(n, 1.0) for k, v in totals.items()}
        if self.mcfg.use_memory:
            out["pool_coverage"] = float(jnp.mean(slot_hits > 0))
            out["pool_spread"] = float(usage_stats(slot_hits)["spread"])
        return out


def _fmt(metrics: Dict[str, Any]) -> str:
    return " ".join(f"{k}={float(v):.4g}" for k, v in metrics.items())


def run(mcfg: ModelConfig, tcfg: TrainConfig, dataset, save_path: str | None = None):
    trainer = Trainer(mcfg, tcfg)
    rng = jax.random.PRNGKey(tcfg.seed)
    rng, init_rng = jax.random.split(rng)
    state = trainer.init(init_rng)
    np_rng = np.random.default_rng(tcfg.seed)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    n_pool = sum(x.size for x in jax.tree_util.tree_leaves(state.params.get("pool", {})))
    print(f"params: total={n_params:,} backbone={n_params - n_pool:,} pool={n_pool:,}")
    if mcfg.use_memory:
        print(f"pool: {mcfg.pool_size:,} slots x {mcfg.d_value} dims, "
              f"{mcfg.pool_heads} heads x top-{mcfg.top_k}")

    history = []
    t0 = time.time()
    for step in range(1, tcfg.steps + 1):
        rng, step_rng, revive_rng = jax.random.split(rng, 3)
        batch = dataset.sample(np_rng, tcfg.batch_size)
        state, metrics, queries = trainer.train_step(state, batch, step_rng)

        revived = 0
        if (
            mcfg.use_memory
            and tcfg.revive_every > 0
            and step % tcfg.revive_every == 0
            and step <= tcfg.revive_until * tcfg.steps
        ):
            state, revived = trainer.revive(state, queries, revive_rng)
            revived = int(revived)

        if step % tcfg.log_every == 0 or step == 1:
            msg = f"step {step:5d} | {_fmt(metrics)}"
            if revived:
                msg += f" | revived_subkeys={revived}"
            print(f"{msg} | {time.time() - t0:.0f}s", flush=True)

        if step % tcfg.eval_every == 0 or step == tcfg.steps:
            ev = trainer.evaluate(state.params, dataset, tcfg.batch_size)
            history.append({"step": step, **ev})
            print(f"  eval  {step:5d} | {_fmt(ev)}", flush=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(serialization.to_bytes(state.params))
        with open(save_path + ".json", "w") as f:
            json.dump({"model": dataclasses.asdict(mcfg), "train": dataclasses.asdict(tcfg)}, f, indent=2)
        print(f"saved params to {save_path}")
    return trainer, state, history


# ---------------------------------------------------------------------- CLI
def _add_dataclass_args(parser: argparse.ArgumentParser, cls) -> None:
    for field in dataclasses.fields(cls):
        default = field.default
        if isinstance(default, bool):
            parser.add_argument(f"--{field.name}", type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        elif isinstance(default, tuple):
            parser.add_argument(
                f"--{field.name}",
                type=lambda s: tuple(int(v) for v in s.split(",") if v),
                default=None,
            )
        else:
            parser.add_argument(f"--{field.name}", type=type(default), default=None)


def _from_args(cls, args, **overrides):
    kwargs = {f.name: getattr(args, f.name) for f in dataclasses.fields(cls) if getattr(args, f.name) is not None}
    kwargs = {**overrides, **kwargs}
    return cls(**kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["facts", "text"], default="facts")
    parser.add_argument("--text_path", type=str, default=None)
    parser.add_argument("--num_entities", type=int, default=4096)
    parser.add_argument("--num_relations", type=int, default=4)
    parser.add_argument("--save", type=str, default=None, help="path to save trained params")
    _add_dataclass_args(parser, ModelConfig)
    _add_dataclass_args(parser, TrainConfig)
    args = parser.parse_args()

    if args.task == "facts":
        dataset = FactDataset(num_entities=args.num_entities, num_relations=args.num_relations)
        mcfg = _from_args(ModelConfig, args, vocab_size=dataset.vocab_size, max_len=dataset.seq_len)
    else:
        if not args.text_path:
            parser.error("--text_path is required for --task text")
        seq_len = args.max_len or ModelConfig.max_len
        dataset = TextDataset(args.text_path, seq_len=seq_len)
        mcfg = _from_args(ModelConfig, args, vocab_size=256)
    tcfg = _from_args(TrainConfig, args)
    run(mcfg, tcfg, dataset, save_path=args.save)


if __name__ == "__main__":
    main()
