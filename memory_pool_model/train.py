"""Training loop with anti-collapse regularisation and dead-key revival.

Usage:
    python -m memory_pool_model.train --task facts --steps 3000
    python -m memory_pool_model.train --task facts --use_memory false   # baseline
    python -m memory_pool_model.train --task text --text_path input.txt
    python -m memory_pool_model.train --task tokens --train_tokens train.npy --eval_tokens val.npy --vocab_size 16384
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
from .data import FactDataset, TextDataset, TokenDataset
from .memory import key_diversity_loss, revive_dead_keys
from .model import MemoryPoolLM

QUERY_SAMPLE = 2048  # recent queries kept for reviving dead sub-keys
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8  # optax.adamw defaults, reused for the pool


@struct.dataclass
class TrainState:
    step: jax.Array
    params: Any  # full parameter tree (pool values included)
    opt_state: Any  # optax state for everything except the pool values (sparse mode)
    subkey_usage: jax.Array  # [H, 2, n] EMA of sub-key selection frequency
    slot_usage: jax.Array  # [N] EMA of slot selection frequency
    pool_m: jax.Array  # [N, D] lazy-Adam first moment of pool values ((0,) if dense)
    pool_v: jax.Array  # [N, D] lazy-Adam second moment


def phase_at(tcfg: TrainConfig, step: int) -> Dict[str, bool]:
    """Which training options are active at this step."""
    return {
        "route": tcfg.route_through_pool or 0 < tcfg.route_after_step < step,
        "nopool": (tcfg.nopool_kl_coef > 0 or tcfg.nopool_true_coef > 0)
        and step > tcfg.nopool_after_step,
        "freeze": 0 < tcfg.freeze_backbone_after_step < step,
    }


def noise_scale_at(tcfg: TrainConfig, step):
    """Routing-noise multiplier at `step` (1-based): 1 until
    noise_anneal_start * steps, then linear down to 0 at noise_anneal_end."""
    if tcfg.noise_anneal_start >= 1.0:
        return 1.0
    frac = step / tcfg.steps
    span = max(tcfg.noise_anneal_end - tcfg.noise_anneal_start, 1e-6)
    return jnp.clip((tcfg.noise_anneal_end - frac) / span, 0.0, 1.0)


def _is_pool_path(path) -> bool:
    top = getattr(path[0], "key", "")
    return top == "pool" or top.startswith(("router_", "mem_gate_", "mem_out_"))


def _path_is(path, *names) -> bool:
    keys = [getattr(p, "key", None) for p in path]
    return keys[: len(names)] == list(names)


def split_values(params):
    """(values, rest): the pool value table and every other parameter.
    values is None when the table lives off-device (host pool)."""
    pool = dict(params["pool"])
    values = pool.pop("values", None)
    return values, {**params, "pool": pool}


def merge_values(rest, values):
    if values is None:
        return rest
    return {**rest, "pool": {**rest["pool"], "values": values}}


def make_schedule(tcfg: TrainConfig):
    return optax.warmup_cosine_decay_schedule(
        0.0, tcfg.lr, tcfg.warmup_steps, max(tcfg.steps, tcfg.warmup_steps + 1), tcfg.lr * 0.1
    )


def make_optimizer(tcfg: TrainConfig, clip: bool = True) -> optax.GradientTransformation:
    """AdamW for the dense parameters. In sparse mode the pool values are not
    in the tree (they get lazy Adam in the train step) and clipping is done
    by hand over both parts, so clip=False."""

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

    parts = [optax.clip_by_global_norm(tcfg.grad_clip)] if clip else []
    parts += [
        optax.adamw(make_schedule(tcfg), weight_decay=tcfg.weight_decay, mask=decay_mask),
        scale_pool_values(tcfg.pool_lr_mult),
    ]
    return optax.chain(*parts)


def usage_stats(usage: jax.Array) -> Dict[str, jax.Array]:
    """Spread of a usage distribution: 1.0 = perfectly uniform, 1/N = collapsed."""
    p = usage / jnp.maximum(usage.sum(axis=-1, keepdims=True), 1e-9)
    entropy = -jnp.sum(jnp.where(p > 0, p * jnp.log(p), 0.0), axis=-1)
    n = usage.shape[-1]
    return {
        "spread": jnp.exp(entropy) / n,
        "active": jnp.mean(p > 0.1 / n, axis=-1),
    }


def fetched_row_grads(aux, probe_grads, layers, pool_size):
    """Gradient of the loss w.r.t. the pool rows fetched this step.

    For a read out = sum_f w_f * V[s_f], dL/dV[s] = sum over fetches of s of
    w_f * dL/dout. dL/dout comes from the zero probes added to each read.
    Returns (unique_slots [U], row_grads [U, D]); padding slots equal
    pool_size (out of range) and are dropped by the caller's scatters.
    """
    slots, contribs = [], []
    for layer, sl, w in zip(layers, aux["slots"], aux["weights"]):
        g = probe_grads[layer]  # [B, T, D]
        d = g.shape[-1]
        sl = sl.reshape(-1, sl.shape[-2] * sl.shape[-1])  # [BT, H*k]
        w = jax.lax.stop_gradient(w).reshape(sl.shape)
        contribs.append((w[..., None] * g.reshape(-1, 1, d)).reshape(-1, d))
        slots.append(sl.reshape(-1))
    slots, contribs = jnp.concatenate(slots), jnp.concatenate(contribs)
    return _merge_rows(slots, contribs, pool_size)


def _merge_rows(slots, contribs, pool_size):
    """Sum contributions per distinct slot. Padding slots (== pool_size)
    sort last and are dropped by the caller's scatters."""
    cap = min(slots.shape[0], pool_size)
    uniq, inv = jnp.unique(slots, size=cap, fill_value=pool_size, return_inverse=True)
    rows = jax.ops.segment_sum(contribs, inv.reshape(-1), num_segments=cap)
    return uniq, rows


def fetched_row_grads_sharded(aux, probe_grads, layers, pool_size, mesh):
    """Data-parallel version of fetched_row_grads.

    Each device first merges its own fetches (no communication), then the
    devices all-gather just their merged rows and merge once more, so every
    device ends with the same (uniq, rows). Letting XLA partition a global
    unique/segment_sum instead reshuffles every per-fetch contribution
    through an all-to-all (about 1 GB per step at batch 64 x 128, two memory
    layers), which failed with an NCCL error on 2x T4.
    """
    try:
        from jax import shard_map
    except ImportError:  # older JAX
        from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    grads = [probe_grads[layer] for layer in layers]

    def local(slots, weights, grads):
        uniq, rows = fetched_row_grads({"slots": slots, "weights": weights},
                                       dict(zip(layers, grads)), layers, pool_size)
        uniq = jax.lax.all_gather(uniq, "data", tiled=True)
        rows = jax.lax.all_gather(rows, "data", tiled=True)
        return _merge_rows(uniq, rows, pool_size)

    # every device merges the same all-gathered data, so the outputs are
    # replicated by construction; JAX can't infer that through unique()
    import inspect

    flag = "check_vma" if "check_vma" in inspect.signature(shard_map).parameters else "check_rep"
    return shard_map(local, mesh=mesh, in_specs=(P("data"), P("data"), P("data")),
                     out_specs=(P(), P()), **{flag: False})(list(aux["slots"]), list(aux["weights"]), grads)


class Trainer:
    def __init__(self, mcfg: ModelConfig, tcfg: TrainConfig, mesh=None, donate: bool = False):
        self.host = None
        if mcfg.use_memory and mcfg.pool_location == "host":
            from .host_pool import HostPool

            if not tcfg.sparse_pool_updates:
                raise ValueError("pool_location='host' requires sparse_pool_updates")
            if mesh is not None:
                raise NotImplementedError("host pool with data parallel is not supported yet")
            self.host = HostPool(mcfg.pool_size, mcfg.d_value, path=mcfg.pool_dir or None, seed=tcfg.seed,
                                 optimizer=tcfg.pool_optimizer)
            mcfg = dataclasses.replace(mcfg, host_pool=self.host.name)
        elif mcfg.pool_location != "device":
            raise ValueError(f"unknown pool_location {mcfg.pool_location!r}")
        self.mcfg, self.tcfg = mcfg, tcfg
        self.model = MemoryPoolLM(mcfg)
        self.sparse = mcfg.use_memory and tcfg.sparse_pool_updates
        self.schedule = make_schedule(tcfg)
        self.optimizer = make_optimizer(tcfg, clip=not self.sparse)
        self.mesh = mesh
        extra = {}
        if mesh is not None:
            from jax.sharding import NamedSharding, PartitionSpec

            self.replicated = NamedSharding(mesh, PartitionSpec())
            self.batch_sharding = NamedSharding(mesh, PartitionSpec("data"))
            extra = {"out_shardings": self.replicated}
        self._jit_step = jax.jit(
            self._train_step,
            static_argnames=("route", "nopool", "freeze"),
            donate_argnums=(0,) if donate else (),
            **extra,
        )
        self.eval_step = jax.jit(self._eval_step)
        self.revive = jax.jit(self._revive, **extra)

    def train_step(self, state, batch, rng, route=None, nopool=None, freeze=False):
        """One optimisation step. With a host pool, the fetched rows' Adam
        update runs on the host right after the device step."""
        lr = float(self.schedule(int(state.step))) * self.tcfg.pool_lr_mult if self.host else 0.0
        t0 = time.perf_counter()
        state, metrics, queries = self._jit_step(state, batch, rng, route=route, nopool=nopool, freeze=freeze)
        if self.host is not None:
            uniq, rows = metrics.pop("_pool_slots"), metrics.pop("_pool_grads")
            uniq, rows = jax.block_until_ready((uniq, rows))
            t1 = time.perf_counter()
            uniq, rows = np.asarray(uniq), np.asarray(rows)
            t2 = time.perf_counter()
            self.host.update(uniq, rows, int(state.step), lr)
            t3 = time.perf_counter()
            # seconds spent per part of the last step (profiling host pools)
            self.host_timing = {"device_step": t1 - t0, "to_host": t2 - t1, "host_adam": t3 - t2,
                                "gather": self.host.pop_gather_seconds()}
        return state, metrics, queries

    # ------------------------------------------------------------ placement
    def place_state(self, state):
        return jax.device_put(state, self.replicated) if self.mesh is not None else state

    def place_batch(self, batch):
        if self.mesh is None:
            return batch
        return jax.device_put(batch, self.batch_sharding)

    # ------------------------------------------------------------------ init
    def init(self, rng: jax.Array) -> TrainState:
        dummy = jnp.zeros((1, self.mcfg.max_len), jnp.int32)
        params = self.model.init({"params": rng}, dummy)["params"]
        m = self.mcfg
        if self.sparse:
            values, rest = split_values(params)
            opt_state = self.optimizer.init(rest)
            if values is None:  # host pool keeps its own optimizer state
                pool_m, pool_v = jnp.zeros((0,)), jnp.zeros((0,))
            elif self.tcfg.pool_optimizer == "rowwise_adagrad":
                pool_m, pool_v = jnp.zeros((0,)), jnp.zeros((values.shape[0],))
            else:
                pool_m, pool_v = jnp.zeros_like(values), jnp.zeros_like(values)
        else:
            opt_state = self.optimizer.init(params)
            # two distinct arrays: donating one buffer twice is an error
            pool_m, pool_v = jnp.zeros((0,)), jnp.zeros((0,))
        return TrainState(
            step=jnp.zeros((), jnp.int32),
            params=params,
            opt_state=opt_state,
            # explicit dtype: a weakly typed initial state would make the
            # second step recompile (the step's outputs are strongly typed)
            subkey_usage=jnp.full((m.pool_heads, 2, m.n_sub_keys), 1.0 / m.n_sub_keys, jnp.float32),
            slot_usage=jnp.full((m.pool_size,), 1.0 / m.pool_size, jnp.float32),
            pool_m=pool_m,
            pool_v=pool_v,
        )

    # ------------------------------------------------------------------ loss
    def _loss(self, params, batch, rng, train: bool, route: bool = False, nopool: bool = False,
              probes=None, noise_scale=1.0):
        r_route, r_drop = jax.random.split(rng)
        logits, aux = self.model.apply(
            {"params": params},
            batch["inputs"],
            train=train,
            route_through_pool=train and route,
            sparse_grad=probes is not None,
            probes=probes,
            noise_scale=noise_scale,
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
                top1_weight=aux["top1_weight"],
            )
        use_nopool = self.mcfg.use_memory and (not train or nopool)
        if use_nopool:
            # Same model with the pool switched off: how much can the
            # backbone answer by itself?
            logits_np, _ = self.model.apply(
                {"params": params}, batch["inputs"], train=train, pool_off=True,
                rngs={"dropout": r_drop},
            )
            metrics["acc_nopool"] = ((logits_np.argmax(-1) == batch["targets"]) * mask).sum() / denom
            if train:
                # KL(uniform || p_nopool) on scored tokens: 0 when the backbone
                # alone has no idea, large when it knows the answer.
                logp = jax.nn.log_softmax(logits_np, -1)
                kl = -jnp.log(logp.shape[-1]) - logp.mean(-1)
                kl = (kl * mask).sum() / denom
                loss = loss + self.tcfg.nopool_kl_coef * kl
                metrics["nopool_kl"] = kl
                # -log(1 - p_correct) with the pool off: only punishes the
                # backbone for putting mass on the right answer by itself.
                p_true = jnp.take_along_axis(
                    jax.nn.softmax(logits_np, -1), batch["targets"][..., None], -1)[..., 0]
                pen = -jnp.log(jnp.clip(1.0 - p_true, 1e-6, 1.0))
                pen = (pen * mask).sum() / denom
                loss = loss + self.tcfg.nopool_true_coef * pen
                metrics["nopool_true_pen"] = pen
        metrics["loss"] = loss
        return loss, (metrics, aux)

    # ------------------------------------------------------------ train step
    def _train_step(self, state: TrainState, batch, rng, route=None, nopool=None, freeze=False):
        """route / nopool / freeze are static: the trainer passes the phase
        for each step explicitly, so switching phases recompiles instead of
        silently keeping the first trace."""
        t = self.tcfg
        if route is None:
            route = t.route_through_pool
        if nopool is None:
            nopool = t.nopool_kl_coef > 0 or t.nopool_true_coef > 0
        r_loss, r_sample = jax.random.split(rng)
        pool_m, pool_v = state.pool_m, state.pool_v
        noise = noise_scale_at(t, state.step + 1)

        if self.sparse:
            values, rest = split_values(state.params)
            B, T = batch["inputs"].shape
            layers = sorted(self.mcfg.memory_layers)
            probes = {i: jnp.zeros((B, T, self.mcfg.d_value)) for i in layers}

            def loss_fn(rest, probes):
                return self._loss(merge_values(rest, values), batch, r_loss, True, route, nopool,
                                  probes=probes, noise_scale=noise)

            (_, (metrics, aux)), (g_rest, g_probe) = jax.value_and_grad(
                loss_fn, argnums=(0, 1), has_aux=True)(rest, probes)
            if self.mesh is not None:
                uniq, g_rows = fetched_row_grads_sharded(aux, g_probe, layers, self.mcfg.pool_size, self.mesh)
            else:
                uniq, g_rows = fetched_row_grads(aux, g_probe, layers, self.mcfg.pool_size)

            # clip by the global norm of both parts together
            gnorm = jnp.sqrt(optax.tree.norm(g_rest) ** 2 + jnp.sum(g_rows**2))
            scale = jnp.minimum(1.0, t.grad_clip / (gnorm + 1e-6))
            g_rest = jax.tree_util.tree_map(lambda g: g * scale, g_rest)
            g_rows = g_rows * scale

            updates, opt_state = self.optimizer.update(g_rest, state.opt_state, rest)
            if freeze:
                updates = jax.tree_util.tree_map_with_path(
                    lambda p, u: u if _is_pool_path(p) else jnp.zeros_like(u), updates)
            rest = optax.apply_updates(rest, updates)

            if values is None:
                # host pool: the trainer applies lazy Adam on the host
                metrics["_pool_slots"], metrics["_pool_grads"] = uniq, g_rows
            elif t.pool_optimizer == "rowwise_adagrad":
                lr = self.schedule(state.step) * t.pool_lr_mult
                acc = jnp.take(pool_v, uniq, mode="fill", fill_value=0) + jnp.mean(g_rows**2, axis=-1)
                values = values.at[uniq].add(-(lr / (jnp.sqrt(acc) + 1e-8))[:, None] * g_rows, mode="drop")
                pool_v = pool_v.at[uniq].set(acc, mode="drop")
            else:
                # lazy Adam on the fetched rows only
                n = (state.step + 1).astype(jnp.float32)
                lr = self.schedule(state.step) * t.pool_lr_mult
                m_rows = ADAM_B1 * jnp.take(pool_m, uniq, axis=0, mode="fill", fill_value=0) + (1 - ADAM_B1) * g_rows
                v_rows = ADAM_B2 * jnp.take(pool_v, uniq, axis=0, mode="fill", fill_value=0) + (1 - ADAM_B2) * g_rows**2
                step_rows = -lr * (m_rows / (1 - ADAM_B1**n)) / (jnp.sqrt(v_rows / (1 - ADAM_B2**n)) + ADAM_EPS)
                values = values.at[uniq].add(step_rows, mode="drop")
                pool_m = pool_m.at[uniq].set(m_rows, mode="drop")
                pool_v = pool_v.at[uniq].set(v_rows, mode="drop")
            params = merge_values(rest, values)
            metrics["grad_norm"] = gnorm
            metrics["rows_updated"] = jnp.sum(uniq < self.mcfg.pool_size)
        else:
            grad_fn = jax.value_and_grad(self._loss, has_aux=True)
            (_, (metrics, aux)), grads = grad_fn(state.params, batch, r_loss, True, route, nopool,
                                                 noise_scale=noise)
            updates, opt_state = self.optimizer.update(grads, state.opt_state, state.params)
            if freeze:
                # Stage 2: backbone frozen; only the pool path (pool, router,
                # read gate/projection) keeps learning.
                updates = jax.tree_util.tree_map_with_path(
                    lambda p, u: u if _is_pool_path(p) else jnp.zeros_like(u), updates)
            params = optax.apply_updates(state.params, updates)
            metrics["grad_norm"] = optax.tree.norm(grads)
        metrics["noise_scale"] = jnp.asarray(noise, jnp.float32)

        subkey_usage, slot_usage = state.subkey_usage, state.slot_usage
        queries = jnp.zeros((0,))
        if self.mcfg.use_memory:
            d = self.tcfg.usage_ema_decay
            sk = aux["subkey_counts"]
            sk = sk / sk.sum(axis=-1, keepdims=True)
            subkey_usage = d * subkey_usage + (1 - d) * sk
            slot_usage = d * slot_usage + (1 - d) * aux["slot_counts"] / aux["slot_counts"].sum()
            # Revival sample: the same random positions from every sequence
            # and layer. Indexing the (unsharded) time axis and stacking layers
            # on a new axis keeps it local to each device under data parallelism.
            B, T = batch["inputs"].shape
            qs = [q.reshape(B, T, *q.shape[1:]) for q in aux["queries"]]
            n_t = max(1, -(-QUERY_SAMPLE // (B * len(qs))))
            t_idx = jax.random.randint(r_sample, (n_t,), 0, T)
            queries = jnp.stack([q[:, t_idx] for q in qs], axis=1).reshape(-1, *qs[0].shape[2:])
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
            pool_m=pool_m,
            pool_v=pool_v,
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
        tree = split_values(params)[1] if self.sparse else params
        mask = jax.tree_util.tree_map_with_path(
            lambda p, x: dead[..., None] if _path_is(p, "pool", "sub_keys") else jnp.zeros((), bool),
            tree,
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
        if self.mcfg.use_memory:
            totals["acc_nopool"] = 0.0
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
            stats = usage_stats(slot_hits)
            # fetched at least once / fetched >= 10% of a fair share / evenness
            out["pool_coverage"] = float(jnp.mean(slot_hits > 0))
            out["pool_active"] = float(stats["active"])
            out["pool_spread"] = float(stats["spread"])
        return out


def _fmt(metrics: Dict[str, Any]) -> str:
    return " ".join(f"{k}={float(v):.4g}" for k, v in metrics.items())


# ------------------------------------------------------------- checkpoints
def save_checkpoint(path: str, state: TrainState, np_rng: np.random.Generator, history, host=None) -> None:
    """Resumable checkpoint: full train state + data RNG + eval history
    (+ a snapshot of a host pool). Written to temp files and renamed, so a
    crash never leaves a torn file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if host is not None:
        host.save(path + ".pool")
    with open(path + ".tmp", "wb") as f:
        f.write(serialization.to_bytes(jax.device_get(state)))
    with open(path + ".json.tmp", "w") as f:
        json.dump({"step": int(state.step), "np_rng": np_rng.bit_generator.state, "history": history}, f)
    os.replace(path + ".tmp", path)
    os.replace(path + ".json.tmp", path + ".json")


def load_checkpoint(path: str, template: TrainState, host=None):
    if host is not None:
        host.load(path + ".pool")
    with open(path, "rb") as f:
        state = serialization.from_bytes(template, f.read())
    with open(path + ".json") as f:
        meta = json.load(f)
    np_rng = np.random.default_rng()
    np_rng.bit_generator.state = meta["np_rng"]
    return state, np_rng, meta["history"]


def run(mcfg: ModelConfig, tcfg: TrainConfig, dataset, save_path: str | None = None,
        meta: Dict[str, Any] | None = None, resume: bool = False, stop_after: int | None = None):
    """Train. Checkpoints go to <save_path>.state; resume=True continues from
    it exactly (per-step RNGs are derived from the step number).
    stop_after simulates a preemption (used by tests)."""
    mesh = None
    if tcfg.data_parallel and jax.device_count() > 1:
        from jax.sharding import Mesh

        if tcfg.batch_size % jax.device_count():
            raise ValueError("batch_size must be divisible by the number of devices")
        mesh = Mesh(np.array(jax.devices()), ("data",))
    trainer = Trainer(mcfg, tcfg, mesh=mesh, donate=True)
    base_rng, init_rng = jax.random.split(jax.random.PRNGKey(tcfg.seed))
    state = trainer.init(init_rng)
    np_rng = np.random.default_rng(tcfg.seed)
    history = []
    ckpt = save_path + ".state" if save_path else None
    if resume and ckpt and os.path.exists(ckpt):
        state, np_rng, history = load_checkpoint(ckpt, state, trainer.host)
        print(f"resumed from {ckpt} at step {int(state.step)}")
    state = trainer.place_state(state)
    start = int(state.step) + 1

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
    n_pool = sum(x.size for x in jax.tree_util.tree_leaves(state.params.get("pool", {})))
    print(f"params: total={n_params:,} backbone={n_params - n_pool:,} pool={n_pool:,}")
    if trainer.host is not None:
        where = trainer.host.path or "host RAM"
        print(f"pool values on {where}: {trainer.host.nbytes() / 1e9:.2f} GB incl. Adam moments")
    if mcfg.use_memory:
        print(f"pool: {mcfg.pool_size:,} slots x {mcfg.d_value} dims, "
              f"{mcfg.pool_heads} heads x top-{mcfg.top_k}, "
              f"{'sparse' if trainer.sparse else 'dense'} pool updates")
    if mesh is not None:
        print(f"data parallel over {jax.device_count()} devices")

    step_key, revive_key = jax.random.split(base_rng)
    t0 = time.time()
    for step in range(start, tcfg.steps + 1):
        batch = trainer.place_batch(dataset.sample(np_rng, tcfg.batch_size))
        phase = phase_at(tcfg, step)
        state, metrics, queries = trainer.train_step(
            state, batch, jax.random.fold_in(step_key, step), **phase)

        revived = 0
        if (
            mcfg.use_memory
            and tcfg.revive_every > 0
            and step % tcfg.revive_every == 0
            and step <= tcfg.revive_until * tcfg.steps
        ):
            state, revived = trainer.revive(state, queries, jax.random.fold_in(revive_key, step))
            revived = int(revived)

        if step % tcfg.log_every == 0 or step == start:
            msg = f"step {step:5d} | {_fmt(metrics)}"
            if revived:
                msg += f" | revived_subkeys={revived}"
            print(f"{msg} | {time.time() - t0:.0f}s", flush=True)

        if step % tcfg.eval_every == 0 or step == tcfg.steps:
            ev = trainer.evaluate(state.params, dataset, tcfg.batch_size)
            history.append({"step": step, **ev})
            print(f"  eval  {step:5d} | {_fmt(ev)}", flush=True)

        if ckpt and ((tcfg.checkpoint_every and step % tcfg.checkpoint_every == 0)
                     or step == stop_after):
            save_checkpoint(ckpt, state, np_rng, history, trainer.host)
        if step == stop_after:
            print(f"stopping after step {step} (simulated preemption)")
            return trainer, state, history

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(serialization.to_bytes(jax.device_get(state.params)))
        with open(save_path + ".json", "w") as f:
            json.dump(
                {
                    "model": dataclasses.asdict(mcfg),
                    "train": dataclasses.asdict(tcfg),
                    "history": history,
                    "train_seconds": time.time() - t0,
                    **(meta or {}),
                },
                f,
                indent=2,
            )
        save_checkpoint(ckpt, state, np_rng, history, trainer.host)
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
    parser.add_argument("--task", choices=["facts", "text", "tokens"], default="facts")
    parser.add_argument("--train_tokens", type=str, default=None, help="--task tokens: training .npy")
    parser.add_argument("--eval_tokens", type=str, default=None, help="--task tokens: held-out .npy")
    parser.add_argument("--eval_windows", type=int, default=640, help="--task tokens: held-out windows per eval")
    parser.add_argument("--text_path", type=str, default=None)
    parser.add_argument("--num_entities", type=int, default=4096)
    parser.add_argument("--num_relations", type=int, default=4)
    parser.add_argument("--num_attributes", type=int, default=256)
    parser.add_argument("--name_alphabet", type=int, default=16)
    parser.add_argument("--name_len", type=int, default=3)
    parser.add_argument("--data_seed", type=int, default=0)
    parser.add_argument("--save", type=str, default=None, help="path to save trained params")
    parser.add_argument("--resume", action="store_true", help="continue from <save>.state if present")
    _add_dataclass_args(parser, ModelConfig)
    _add_dataclass_args(parser, TrainConfig)
    args = parser.parse_args()

    meta = {}
    if args.task == "facts":
        meta["dataset"] = dict(
            num_entities=args.num_entities,
            num_relations=args.num_relations,
            num_attributes=args.num_attributes,
            name_alphabet=args.name_alphabet,
            name_len=args.name_len,
            seed=args.data_seed,
        )
        dataset = FactDataset(**meta["dataset"])
        mcfg = _from_args(ModelConfig, args, vocab_size=dataset.vocab_size, max_len=dataset.seq_len)
    elif args.task == "tokens":
        if not (args.train_tokens and args.eval_tokens and args.vocab_size):
            parser.error("--task tokens needs --train_tokens, --eval_tokens and --vocab_size")
        seq_len = args.max_len or ModelConfig.max_len
        dataset = TokenDataset(args.train_tokens, args.eval_tokens, args.vocab_size, seq_len=seq_len,
                               eval_windows=args.eval_windows)
        meta["dataset"] = {"train_tokens": args.train_tokens, "eval_tokens": args.eval_tokens,
                           "train_size": int(len(dataset.train)), "eval_size": int(len(dataset.eval))}
        mcfg = _from_args(ModelConfig, args)
    else:
        if not args.text_path:
            parser.error("--text_path is required for --task text")
        seq_len = args.max_len or ModelConfig.max_len
        dataset = TextDataset(args.text_path, seq_len=seq_len)
        mcfg = _from_args(ModelConfig, args, vocab_size=256)
    tcfg = _from_args(TrainConfig, args)
    run(mcfg, tcfg, dataset, save_path=args.save, meta=meta, resume=args.resume)


if __name__ == "__main__":
    main()
