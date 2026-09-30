"""Trainable memory pool with a top-k product-key router.

The pool plays the role of a RAG vector database, except that nothing in it
is text or a frozen embedding: both the addressing keys and the stored value
vectors are parameters learned end-to-end with the backbone.

Lookup cost is sub-linear in the pool size thanks to product keys
(Lample et al., 2019): a query is split in two halves, each half is matched
against `n_sub_keys` sub-keys, and the top-k of the `k * k` combined
candidates is taken. This finds the *exact* top-k over all
`n_sub_keys ** 2` slots while only scoring `2 * n_sub_keys` keys.

Anti-collapse mechanisms implemented here:
  * cosine routing scores (no key can win just by growing its norm),
  * Gumbel noise on routing scores during training (exploration),
  * a Switch-style load-balancing loss over each sub-key codebook,
  * a key-diversity loss that spreads sub-keys over the unit sphere,
  * usage statistics consumed by `revive_dead_keys` (see train.py).
"""

from __future__ import annotations

import concurrent.futures
import functools
import os
import warnings
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn


def _l2_normalize(x: jax.Array, axis: int = -1, eps: float = 1e-6) -> jax.Array:
    return x * jax.lax.rsqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)


def _top_k_by_max(x: jax.Array, k: int) -> Tuple[jax.Array, jax.Array]:
    """Exact top-k as k rounds of argmax + mask-out (ties: lowest index
    first, like lax.top_k). On a T4 this is 9-15x faster than lax.top_k for
    the router's shapes (65k rows x 512, k=16: 10.7 vs 95 ms)."""
    ar = jnp.arange(x.shape[-1])
    vals, idx = [], []
    for _ in range(k):
        i = jnp.argmax(x, axis=-1)
        vals.append(jnp.take_along_axis(x, i[..., None], axis=-1)[..., 0])
        idx.append(i)
        x = jnp.where(ar == i[..., None], -jnp.inf, x)
    return jnp.stack(vals, -1), jnp.stack(idx, -1)


@functools.lru_cache(maxsize=None)
def _pallas_top_k_ok() -> bool:
    """Whether the Pallas top-k kernel compiles and runs here (TPU only; it
    needs a libtpu that matches jaxlib). MEMPOOL_PALLAS=0 turns it off."""
    if os.environ.get("MEMPOOL_PALLAS", "1") == "0" or jax.default_backend() != "tpu":
        return False

    def probe():
        from . import topk_pallas

        x = jnp.arange(8 * 128, dtype=jnp.float32).reshape(8, 128)
        _, i = topk_pallas.top_k(x, 2)
        return bool(jnp.all(i == jnp.array([127, 126])))

    try:
        # usually called while a step is being traced; JAX's trace state is
        # per thread, so a fresh thread runs the probe eagerly
        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            ok = ex.submit(probe).result()
    except Exception as e:  # noqa: BLE001 - any compile/runtime failure means "don't use it"
        warnings.warn(f"Pallas top-k unavailable, using lax.top_k: {type(e).__name__}: {str(e)[:200]}")
        return False
    if not ok:
        warnings.warn("Pallas top-k gave a wrong result, using lax.top_k")
    return ok


@functools.lru_cache(maxsize=None)
def _fused_router_ok() -> bool:
    """Whether to use the fused Pallas router (router_pallas.py): opt-in with
    MEMPOOL_FUSED_ROUTER=1 on TPU, and only if the kernel compiles (probed
    once). It is exact but not yet faster: v5e, d512 x 6 pool, top-8, a
    training step takes 54.1 ms with it and 53.3 ms with the XLA router."""
    if os.environ.get("MEMPOOL_FUSED_ROUTER", "0") != "1" or not _pallas_top_k_ok():
        return False

    def probe():
        from . import router_pallas

        q = jnp.ones((2, 8, 8), jnp.float32) / jnp.sqrt(8.0)
        keys = jnp.eye(128, 8, dtype=jnp.float32)[None].repeat(2, 0)
        idx = router_pallas.fused_router(q, keys, jnp.float32(1.0), jnp.int32(0), jnp.float32(0.0), 2, True, False)[0]
        return bool(jnp.all(idx[..., 0] < 8))

    try:
        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            ok = ex.submit(probe).result()
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"fused Pallas router unavailable, using the XLA router: {type(e).__name__}: {str(e)[:200]}")
        return False
    return ok


def _top_k(x: jax.Array, k: int) -> Tuple[jax.Array, jax.Array]:
    """Top-k over the last axis, always on a 2-D view (lax.top_k on N-D
    inputs was 7x slower on TPU). TPU: a Pallas kernel (topk_pallas.py) when
    the width is lane-aligned; XLA's top_k is a full sort there (v5e, 65k
    rows x 512, k=16: 1.2 ms vs 7.5 ms, and 23.5 ms with argmax rounds).
    GPU: repeated argmax, where XLA's top_k is slow for small k. CPU:
    lax.top_k. Values carry no gradient here; read differentiable values
    with take_last."""
    flat = jax.lax.stop_gradient(x.reshape(-1, x.shape[-1]))
    backend = jax.default_backend()
    if backend == "tpu" and _pallas_top_k_ok():
        from . import topk_pallas

        if topk_pallas.supported(flat.shape[-1], k):
            v, i = topk_pallas.top_k(flat, k)
            return v.reshape(*x.shape[:-1], k), i.reshape(*x.shape[:-1], k)
    if backend == "gpu" and k <= 32:
        v, i = _top_k_by_max(flat, k)
    else:
        v, i = jax.lax.top_k(flat, k)
    return v.reshape(*x.shape[:-1], k), i.reshape(*x.shape[:-1], k)


def _onehot_take(x: jax.Array, idx: jax.Array) -> jax.Array:
    return jnp.sum(jnp.where(idx[..., :, None] == jnp.arange(x.shape[-1]), x[..., None, :], 0), axis=-1)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _onehot_take_diff(x: jax.Array, idx: jax.Array, width: int) -> jax.Array:
    return _onehot_take(x, idx)


def _onehot_take_fwd(x, idx, width):
    return _onehot_take(x, idx), idx


def _onehot_take_bwd(width, idx, g):
    dx = jnp.sum(jnp.where(idx[..., :, None] == jnp.arange(width), g[..., :, None], 0), axis=-2)
    return dx.astype(g.dtype), None


_onehot_take_diff.defvjp(_onehot_take_fwd, _onehot_take_bwd)


def take_last(x: jax.Array, idx: jax.Array) -> jax.Array:
    """take_along_axis(x, idx, axis=-1) for a small number of indices per row.

    On TPU a gather along a short minor axis and its scatter-add gradient
    are very slow (v5e, x [8192, 4, 2, 512], 16 indices per row: 16.5 ms
    forward, 10.8 ms backward); a one-hot compare-select-reduce, which XLA
    fuses without materialising the one-hot, is 0.7 ms and 1.0 ms. Other
    backends use the gather."""
    if jax.default_backend() != "tpu":
        return jnp.take_along_axis(x, idx, axis=-1)
    if not jnp.issubdtype(x.dtype, jnp.floating):
        return _onehot_take(x, idx)
    return _onehot_take_diff(x, idx, x.shape[-1])


def _count_ids(ids: jax.Array, n: int) -> jax.Array:
    """ids [M, H, k] in [0, n) -> per-head counts [H, n] (one-hot sum, which
    XLA fuses into a reduction; no scatter)."""
    return jnp.sum(ids[..., None] == jnp.arange(n), axis=(0, 2), dtype=jnp.float32)


def unit_sphere_init(key, shape, dtype=jnp.float32):
    return _l2_normalize(jax.random.normal(key, shape, dtype))


class MemoryPool(nn.Module):
    """A shared pool of `n_sub_keys ** 2` trainable knowledge vectors.

    Call with router queries of shape [..., heads, d_key]; returns the
    fetched knowledge of shape [..., d_value] and a dict of routing stats.
    """

    n_sub_keys: int
    heads: int
    d_key: int
    d_value: int
    top_k: int
    routing_noise: float = 0.1  # see ModelConfig.routing_noise
    init_temperature: float = 10.0
    # Lower bound of the routing temperature. With a low bound the model can
    # flatten the mixing weights until the Gumbel noise decides every pick.
    min_temperature: float = 10.0
    # Let the balance loss change the temperature. When on, the loss can be
    # lowered by flattening the router softmax instead of spreading usage.
    balance_temperature_grad: bool = False
    # Upper bound of the routing temperature.
    max_temperature: float = 100.0
    # Scale each query's scores by its own length (|q| / sqrt(d_key/2)):
    # keys stay unit-norm, so the router can make individual reads sharp
    # without a global temperature. Off = pure cosine routing.
    query_scale: bool = False
    # Name of a registered host_pool.HostPool: the value table then lives in
    # host RAM / on SSD and fetched rows are copied in (no "values" param).
    host_pool: str = ""
    # Initial values: normal(0, value_init_scale / sqrt(d_value)); 0 = zeros.
    value_init_scale: float = 1.0
    # Balance-loss usage from noise-free picks (costs a second top-k per layer).
    balance_on_clean_picks: bool = True
    # Name of a registered sharded_pool mesh: the value table is sharded by
    # rows over its data axis and training reads go through an all-to-all.
    pool_mesh: str = ""

    def setup(self):
        assert self.d_key % 2 == 0, "d_key must be even (split into two halves)"
        assert self.top_k <= self.n_sub_keys
        half = self.d_key // 2
        # [heads, 2 halves, n_sub_keys, d_key / 2]
        self.sub_keys = self.param(
            "sub_keys", unit_sphere_init, (self.heads, 2, self.n_sub_keys, half)
        )
        # The knowledge itself: one vector per slot, shared by all heads
        # and by every backbone layer that reads from the pool.
        if not self.host_pool:
            self.values = self.param(
                "values",
                nn.initializers.normal(stddev=self.value_init_scale * self.d_value**-0.5),
                (self.n_sub_keys**2, self.d_value),
            )
        self.log_temperature = self.param(
            "log_temperature",
            lambda _: jnp.log(jnp.asarray(self.init_temperature, jnp.float32)),
        )

    @property
    def pool_size(self) -> int:
        return self.n_sub_keys**2

    def __call__(
        self, queries: jax.Array, *, train: bool = False, sparse_grad: bool = False,
        noise_scale: jax.Array | float = 1.0, shuffle_reads: bool = False,
    ) -> Tuple[jax.Array, Dict[str, Any]]:
        """sparse_grad: don't differentiate through the value table. The
        trainer then builds gradients for just the fetched rows (see
        train.py), so gradient/optimizer work scales with rows used, not
        with pool size.
        noise_scale: multiplies routing_noise (the trainer anneals it to 0 at
        the end of training so the rows read in training match inference).
        shuffle_reads: ablation. Routing is unchanged, but every read fetches
        another slot's vector (a fixed shift with no fixed points), so the
        model gets well-formed but wrong knowledge."""
        lead_shape = queries.shape[:-2]
        H, n, k = self.heads, self.n_sub_keys, self.top_k
        half = self.d_key // 2

        q = queries.reshape(-1, H, 2, half)  # [M, H, 2, half]
        q_len = jnp.sqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6) / jnp.sqrt(half)
        q = _l2_normalize(q)
        keys = _l2_normalize(self.sub_keys)
        # Clamp with a straight-through gradient: a plain clip has zero
        # gradient outside the range, so a temperature that hit the bound
        # could never move back.
        log_t = self.log_temperature
        log_t = log_t + jax.lax.stop_gradient(
            jnp.clip(log_t, jnp.log(self.min_temperature), jnp.log(self.max_temperature)) - log_t)
        temperature = jnp.exp(log_t)

        # Cosine similarity of each query half with every sub-key. Heads and
        # halves are one axis of G = 2H codebooks: [M, G, n] tiles cleanly on
        # TPU, while [M, H, 2, n] (a size-2 dim next to the minor one) made
        # XLA relayout every routing tensor (v5e: 1.6 ms for the top-k
        # reshape alone, plus slower elementwise passes).
        M = q.shape[0]
        G = 2 * H
        noisy = train and self.routing_noise > 0
        fused = None
        if (not self.query_scale and not self.balance_temperature_grad and _fused_router_ok()):
            from . import router_pallas

            if router_pallas.supported(n, k, M):
                # Stage 1 in one Pallas kernel: the [M, G, n] scores never
                # leave VMEM (see router_pallas.py).
                # the kernel draws the Gumbel noise itself from this seed
                seed = (jax.random.bits(self.make_rng("routing"), (), jnp.uint32).astype(jnp.int32)
                        if noisy else jnp.zeros((), jnp.int32))
                scale = jnp.asarray(self.routing_noise * noise_scale if noisy else 0.0, jnp.float32)
                idx_t, sub_t, sel_t, member_sum, p_sum, agree = router_pallas.fused_router(
                    q.reshape(M, G, half).transpose(1, 0, 2), keys.reshape(G, n, half), temperature, seed,
                    scale, k, self.balance_on_clean_picks, noisy)
                sub_idx_g, sub_scores_g, sel_vals_g = (x.transpose(1, 0, 2) for x in (idx_t, sub_t, sel_t))
                fused = (member_sum, p_sum, agree)
        if fused is None:
            cos = jnp.einsum("mgd,gnd->mgn", q.reshape(M, G, half), keys.reshape(G, n, half))
            scores = cos * temperature
            if self.query_scale:
                scores = scores * q_len.reshape(M, G, 1)

            # Noise only changes *which* slots are selected, never their weights.
            if noisy:
                gumbel = jax.random.gumbel(self.make_rng("routing"), scores.shape)
                select_scores = scores + self.routing_noise * noise_scale * gumbel
            else:
                select_scores = scores

            # Stage 1: top-k sub-keys for each half.
            sel_vals_g, sub_idx_g = _top_k(select_scores, k)  # [M, G, k]
            sub_scores_g = take_last(scores, sub_idx_g)  # differentiable
        sub_idx = sub_idx_g.reshape(M, H, 2, k)
        sub_scores = sub_scores_g.reshape(M, H, 2, k)
        sub_select = sel_vals_g.reshape(M, H, 2, k)  # = select_scores at sub_idx (selection only)

        # Stage 2: exact top-k over the k*k cartesian candidates.
        cand = sub_scores[..., 0, :, None] + sub_scores[..., 1, None, :]
        cand_select = sub_select[..., 0, :, None] + sub_select[..., 1, None, :]
        cand = cand.reshape(*cand.shape[:-2], k * k)  # [M, H, k*k]
        cand_select = cand_select.reshape(cand.shape)
        _, flat_idx = _top_k(cand_select, k)  # [M, H, k]
        slot_scores = take_last(cand, flat_idx)
        idx_a = take_last(sub_idx[..., 0, :], flat_idx // k)
        idx_b = take_last(sub_idx[..., 1, :], flat_idx % k)
        slots = idx_a * n + idx_b  # [M, H, k]

        # Fetch and mix knowledge vectors; heads are summed.
        weights = jax.nn.softmax(slot_scores, axis=-1)
        read_slots = (slots + self.pool_size // 2 + 1) % self.pool_size if shuffle_reads else slots
        shard_dropped = jnp.zeros((), jnp.float32)
        if self.host_pool:
            from . import host_pool as hp

            fetched = hp.fetch(self.host_pool, read_slots, self.d_value)  # [M, H, k, d_value]
        elif self.pool_mesh and train:
            from . import sharded_pool

            values = jax.lax.stop_gradient(self.values) if sparse_grad else self.values
            fetched, shard_dropped = sharded_pool.fetch(self.pool_mesh, values, read_slots)
        else:
            values = jax.lax.stop_gradient(self.values) if sparse_grad else self.values
            fetched = jnp.take(values, read_slots, axis=0)  # [M, H, k, d_value]
            if jax.default_backend() == "gpu":
                # Keep the gathered rows in row-major order. Otherwise XLA can
                # pick a layout with d_value outermost to suit the consumer, and
                # the gather then reads every row with scattered accesses (113 ms
                # instead of ~8 ms per layer on a T4 at batch 32 x 256). On TPU
                # the barrier costs a relayout copy of all fetched rows and stops
                # XLA fusing the gather into the mix (v5e: 6.6 ms per step).
                fetched = jax.lax.optimization_barrier(fetched.reshape(-1)).reshape(fetched.shape)
        out = jnp.einsum("mhk,mhkd->md", weights, fetched)
        out = out.reshape(*lead_shape, self.d_value)

        # ---- routing statistics / anti-collapse terms ----
        # Scatters are slow on TPU (~7 ms per 1M updates on v5e), so there is
        # one per layer: per-head slot counts, from which the sub-key counts
        # and the pool-wide slot counts are sums. Top-k memberships are
        # counted densely by comparing scores with the k-th largest.
        if train and jax.default_backend() == "tpu":
            # Training on TPU: no scatter at all. Sub-key counts by a fused
            # one-hot reduction (exact); per-slot counts are left out (None)
            # and the trainer takes the touched rows from the row gradients
            # (v5e: 3.5 ms per layer saved).
            subkey_counts = jnp.stack([_count_ids(idx_a, n), _count_ids(idx_b, n)], axis=1)  # [H, 2, n]
            slot_counts = None
        else:
            head_slots = (slots + jnp.arange(H)[None, :, None] * self.pool_size).reshape(-1)
            head_counts = jnp.zeros((H * self.pool_size,), jnp.float32).at[head_slots].add(1.0).reshape(H, n, n)
            # Sub-keys actually used by the final selection (for usage tracking).
            subkey_counts = jnp.stack([head_counts.sum(2), head_counts.sum(1)], axis=1)  # [H, 2, n]
            slot_counts = head_counts.sum(0).reshape(-1)

        # Load balancing is measured on the *clean* router so the noise can't
        # hide a collapse. f: fraction of top-k picks per sub-key (no grad).
        # balance_on_clean_picks=False reuses the training picks instead and
        # skips a second sub-key top-k per layer (small noise barely changes
        # them); pick_agreement is then not measured (NaN).
        if fused is not None:
            member_sum, p_sum, agree = fused
            f = jax.lax.stop_gradient(member_sum / (M * k)).reshape(H, 2, n)
            P = (p_sum / M).reshape(H, 2, n)
            if not noisy:
                pick_agreement = jnp.ones((), jnp.float32)
            elif not self.balance_on_clean_picks:
                pick_agreement = jnp.full((), jnp.nan, jnp.float32)
            else:
                pick_agreement = agree / (M * G * k)
        else:
            clean = select_scores is scores or not self.balance_on_clean_picks
            if clean:
                member = select_scores >= sel_vals_g[..., -1:]  # [M, G, n]
            else:
                clean_vals = _top_k(scores, k)[0]  # [M, G, k]
                member = scores >= clean_vals[..., -1:]
            # exact up to ties at the k-th score (a zero-probability event for
            # float scores), in which case a tied sub-key is counted too
            f = jax.lax.stop_gradient(jnp.sum(member, axis=0, dtype=jnp.float32) / (M * k)).reshape(H, 2, n)
            # Share of the noisy sub-key picks that the clean router also makes.
            # Near 0 means the noise, not the router, decides what training reads
            # (Ultra-FineWeb, 512 sub-keys, routing_noise 1.0: 0.4% of the slots
            # read in training were the slots inference reads).
            if select_scores is scores:
                pick_agreement = jnp.ones((), jnp.float32)
            elif clean:
                pick_agreement = jnp.full((), jnp.nan, jnp.float32)
            else:
                pick_agreement = jnp.mean(sub_scores_g >= clean_vals[..., -1:])
            # P: mean router probability for each sub-key (differentiable).
            p_scores = scores if self.balance_temperature_grad else cos * jax.lax.stop_gradient(temperature)
            P = jax.nn.softmax(p_scores, axis=-1).mean(axis=0).reshape(H, 2, n)
        # Equals 1 when routing is perfectly uniform; grows with collapse.
        balance_loss = n * jnp.mean(jnp.sum(f * P, axis=-1))

        aux = {
            "balance_loss": balance_loss,
            "subkey_counts": jax.lax.stop_gradient(subkey_counts),  # [H, 2, n]
            "slot_counts": None if slot_counts is None else jax.lax.stop_gradient(slot_counts),  # [N]
            "queries": jax.lax.stop_gradient(q),  # [M, H, 2, half] (unit norm)
            "slots": slots.reshape(*lead_shape, H, k),
            "weights": weights.reshape(*lead_shape, H, k),
            "temperature": temperature,
            "top1_weight": jax.lax.stop_gradient(weights.max(-1).mean()),
            "pick_agreement": jax.lax.stop_gradient(pick_agreement.astype(jnp.float32)),
            # reads dropped by full buckets of the sharded pool (0 otherwise)
            "shard_dropped": jax.lax.stop_gradient(shard_dropped),
        }
        return out, aux


def key_diversity_loss(sub_keys: jax.Array) -> jax.Array:
    """Mean squared off-diagonal cosine similarity inside each sub-key codebook.

    Minimised when the sub-keys form a tight frame, i.e. are spread evenly
    over the sphere instead of clumping in one region of query space.
    """
    keys = _l2_normalize(sub_keys)  # [H, 2, n, d]
    gram = jnp.einsum("hcnd,hcmd->hcnm", keys, keys)
    n = keys.shape[2]
    off_diag = gram * (1.0 - jnp.eye(n, dtype=gram.dtype))
    return jnp.sum(off_diag**2) / (keys.shape[0] * keys.shape[1] * n * (n - 1))


def revive_dead_keys(
    sub_keys: jax.Array,
    subkey_usage: jax.Array,
    queries: jax.Array,
    rng: jax.Array,
    threshold: float,
    noise: float = 0.05,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Re-initialise rarely-used sub-keys onto recently observed queries.

    Like codebook restarts in VQ-VAE: a sub-key the router has stopped
    selecting receives no gradient and would stay dead forever. Moving it to
    where real queries live brings the pool slots behind it back into use.

    Args:
      sub_keys: [H, 2, n, d] pool sub-keys.
      subkey_usage: [H, 2, n] EMA usage fractions (each [h, c] row sums to 1,
        and still does after revival).
      queries: [M, H, 2, d] unit-norm query halves from recent batches.
      threshold: a key is dead if usage < threshold * (1 / n).

    Returns:
      (new_sub_keys, new_subkey_usage, dead_mask)
    """
    H, C, n, d = sub_keys.shape
    dead = subkey_usage < threshold / n
    k_pick, k_noise = jax.random.split(rng)
    pick = jax.random.randint(k_pick, (H, C, n), 0, queries.shape[0])
    h_idx = jnp.arange(H)[:, None, None]
    c_idx = jnp.arange(C)[None, :, None]
    replacement = queries[pick, h_idx, c_idx]  # [H, C, n, d]
    replacement = replacement + noise * jax.random.normal(k_noise, replacement.shape)
    replacement = _l2_normalize(replacement)
    new_keys = jnp.where(dead[..., None], replacement, sub_keys)
    # Give revived keys a grace period at uniform usage.
    new_usage = jnp.where(dead, 1.0 / n, subkey_usage)
    # keep each codebook's usage a distribution (the revived keys added mass)
    new_usage = new_usage / new_usage.sum(axis=-1, keepdims=True)
    return new_keys, new_usage, dead
