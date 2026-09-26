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

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn


def _l2_normalize(x: jax.Array, axis: int = -1, eps: float = 1e-6) -> jax.Array:
    return x * jax.lax.rsqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)


def _top_k(x: jax.Array, k: int) -> Tuple[jax.Array, jax.Array]:
    """lax.top_k over the last axis, run on a 2-D view (much faster on CPU)."""
    v, i = jax.lax.top_k(x.reshape(-1, x.shape[-1]), k)
    return v.reshape(*x.shape[:-1], k), i.reshape(*x.shape[:-1], k)


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
    routing_noise: float = 1.0
    init_temperature: float = 10.0
    # Lower bound of the routing temperature. With a low bound the model can
    # flatten the mixing weights until the Gumbel noise decides every pick.
    min_temperature: float = 1.0
    # Let the balance loss change the temperature. When on, the loss can be
    # lowered by flattening the router softmax instead of spreading usage.
    balance_temperature_grad: bool = True
    # Name of a registered host_pool.HostPool: the value table then lives in
    # host RAM / on SSD and fetched rows are copied in (no "values" param).
    host_pool: str = ""

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
                nn.initializers.normal(stddev=self.d_value**-0.5),
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
        noise_scale: jax.Array | float = 1.0,
    ) -> Tuple[jax.Array, Dict[str, Any]]:
        """sparse_grad: don't differentiate through the value table. The
        trainer then builds gradients for just the fetched rows (see
        train.py), so gradient/optimizer work scales with rows used, not
        with pool size.
        noise_scale: multiplies routing_noise (the trainer anneals it to 0 at
        the end of training so the rows read in training match inference)."""
        lead_shape = queries.shape[:-2]
        H, n, k = self.heads, self.n_sub_keys, self.top_k
        half = self.d_key // 2

        q = queries.reshape(-1, H, 2, half)  # [M, H, 2, half]
        q = _l2_normalize(q)
        keys = _l2_normalize(self.sub_keys)
        # Clamp with a straight-through gradient: a plain clip has zero
        # gradient outside the range, so a temperature that hit the bound
        # could never move back.
        log_t = self.log_temperature
        log_t = log_t + jax.lax.stop_gradient(
            jnp.clip(log_t, jnp.log(self.min_temperature), jnp.log(100.0)) - log_t)
        temperature = jnp.exp(log_t)

        # Cosine similarity of each query half with every sub-key.
        cos = jnp.einsum("mhcd,hcnd->mhcn", q, keys)
        scores = cos * temperature

        # Noise only changes *which* slots are selected, never their weights.
        if train and self.routing_noise > 0:
            gumbel = jax.random.gumbel(self.make_rng("routing"), scores.shape)
            select_scores = scores + self.routing_noise * noise_scale * gumbel
        else:
            select_scores = scores

        # Stage 1: top-k sub-keys for each half.
        _, sub_idx = _top_k(select_scores, k)  # [M, H, 2, k]
        sub_scores = jnp.take_along_axis(scores, sub_idx, axis=-1)
        sub_select = jnp.take_along_axis(select_scores, sub_idx, axis=-1)

        # Stage 2: exact top-k over the k*k cartesian candidates.
        cand = sub_scores[..., 0, :, None] + sub_scores[..., 1, None, :]
        cand_select = sub_select[..., 0, :, None] + sub_select[..., 1, None, :]
        cand = cand.reshape(*cand.shape[:-2], k * k)  # [M, H, k*k]
        cand_select = cand_select.reshape(cand.shape)
        _, flat_idx = _top_k(cand_select, k)  # [M, H, k]
        slot_scores = jnp.take_along_axis(cand, flat_idx, axis=-1)
        idx_a = jnp.take_along_axis(sub_idx[..., 0, :], flat_idx // k, axis=-1)
        idx_b = jnp.take_along_axis(sub_idx[..., 1, :], flat_idx % k, axis=-1)
        slots = idx_a * n + idx_b  # [M, H, k]

        # Fetch and mix knowledge vectors; heads are summed.
        weights = jax.nn.softmax(slot_scores, axis=-1)
        if self.host_pool:
            from . import host_pool as hp

            fetched = hp.fetch(self.host_pool, slots, self.d_value)  # [M, H, k, d_value]
        else:
            values = jax.lax.stop_gradient(self.values) if sparse_grad else self.values
            fetched = jnp.take(values, slots, axis=0)  # [M, H, k, d_value]
        out = jnp.einsum("mhk,mhkd->md", weights, fetched)
        out = out.reshape(*lead_shape, self.d_value)

        # ---- routing statistics / anti-collapse terms ----
        M = q.shape[0]
        codebook = (jnp.arange(H)[:, None, None] * 2 + jnp.arange(2)[None, :, None]) * n

        def count_subkeys(ids):  # ids: [M, H, 2, k] -> counts [H, 2, n]
            flat = (ids + codebook).reshape(-1)
            return jnp.zeros((H * 2 * n,), jnp.float32).at[flat].add(1.0).reshape(H, 2, n)

        # Sub-keys actually used by the final selection (for usage tracking).
        subkey_counts = count_subkeys(jnp.stack([idx_a, idx_b], axis=2))
        # Load balancing is measured on the *clean* router so the noise can't
        # hide a collapse. f: fraction of top-k picks per sub-key (no grad).
        clean_idx = sub_idx if select_scores is scores else _top_k(scores, k)[1]
        f = jax.lax.stop_gradient(count_subkeys(clean_idx) / (M * k))  # [H, 2, n]
        # P: mean router probability for each sub-key (differentiable).
        p_scores = scores if self.balance_temperature_grad else cos * jax.lax.stop_gradient(temperature)
        P = jax.nn.softmax(p_scores, axis=-1).mean(axis=0)  # [H, 2, n]
        # Equals 1 when routing is perfectly uniform; grows with collapse.
        balance_loss = n * jnp.mean(jnp.sum(f * P, axis=-1))

        slot_counts = jnp.zeros((self.pool_size,), jnp.float32).at[slots.reshape(-1)].add(1.0)

        aux = {
            "balance_loss": balance_loss,
            "subkey_counts": jax.lax.stop_gradient(subkey_counts),  # [H, 2, n]
            "slot_counts": jax.lax.stop_gradient(slot_counts),  # [N]
            "queries": jax.lax.stop_gradient(q),  # [M, H, 2, half] (unit norm)
            "slots": slots.reshape(*lead_shape, H, k),
            "weights": weights.reshape(*lead_shape, H, k),
            "temperature": temperature,
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
      subkey_usage: [H, 2, n] EMA usage fractions (each [h, c] row sums to 1).
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
    return new_keys, new_usage, dead
