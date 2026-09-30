"""Fused stage-1 router (sub-key scoring + top-k + balance statistics) for TPU.

The XLA router materialises the [tokens, 2*heads, n_sub_keys] score tensor
in HBM and makes ~8 memory-bound passes over it per layer (scale, noise,
top-k, one-hot gather of clean scores, membership, softmax statistics) plus
their gradients. This kernel keeps a block of rows in VMEM, computes the
scores on the MXU and writes only what the rest of the router needs:

  sub_idx     [G, M, k]  top-k sub-keys of the noisy scores
  sub_scores  [G, M, k]  clean scores at those sub-keys (differentiable)
  sel_vals    [G, M, k]  noisy scores at those sub-keys (selection only)
  member      [G, n]     how many rows have each sub-key in their top-k
                         (clean top-k when `clean` is set, else the noisy one)
  p_sum       [G, n]     sum over rows of softmax(scores) (differentiable)
  agree       []         noisy picks that are also in the clean top-k

Status: exact selection (same picks as XLA; clean scores within float
rounding, gradients within 1e-6), with the Gumbel noise generated in the
kernel from a counter hash (router_noise() gives the same noise as an array).
Per layer, forward + backward at 8,192 tokens x 8 codebooks x 512 sub-keys
on v5e: 6.7 ms at top-16 and 3.7 ms at top-8. A full training step is still
not faster than with the XLA router and the Pallas top-k (d512 x 6 pool:
54.1 vs 53.3 ms at top-8, 84.9 vs 78.3 ms at top-16), so it is opt-in
(MEMPOOL_FUSED_ROUTER=1). The forward is bound by cross-lane reductions (two
per selection round: max, then the lowest index at the max).

The backward kernel recomputes the scores in VMEM. Gradients flow to q, keys
and the temperature through sub_scores, and to q and keys (not the
temperature, like the XLA path) through p_sum.
"""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

TM = int(os.environ.get("MEMPOOL_ROUTER_TM", "128"))  # rows (tokens) per block
VMEM_LIMIT = 96 * 2**20  # v5e has 128 MiB of VMEM; the default scoped limit is 16 MiB


def supported(n: int, k: int, m: int) -> bool:
    return n % 128 == 0 and n <= 2048 and k <= 32 and m % 8 == 0


_GOLD = 0x9E3779B9


def _hash_u32(x):
    """lowbias32 integer hash (uint32 -> uint32)."""
    x = x ^ (x >> 16)
    x = x * jnp.uint32(0x7FEB352D)
    x = x ^ (x >> 15)
    x = x * jnp.uint32(0x846CA68B)
    return x ^ (x >> 16)


def _gumbel_from(seed_u32, g, row, col, n):
    """Gumbel(0, 1) noise for (codebook g, row, col) from a counter hash.
    row / col: int32 arrays that broadcast; seed_u32: uint32 scalar."""
    ctr = (row.astype(jnp.uint32) * jnp.uint32(n) + col.astype(jnp.uint32))
    x = _hash_u32(ctr ^ (seed_u32 + jnp.uint32((g + 1) * _GOLD & 0xFFFFFFFF)))
    x = _hash_u32(x + seed_u32)
    # 23 bits: every (x + 0.5) is exact in float32, so u < 1 (with 24 bits the
    # top value rounds to 1.0 and the noise becomes infinite)
    u = ((x >> 9).astype(jnp.int32).astype(jnp.float32) + 0.5) * (1.0 / (1 << 23))
    return -jnp.log(-jnp.log(u))


def router_noise(seed, G, M, n):
    """The kernel's noise as a [G, M, n] array (for tests and reference)."""
    s = jnp.asarray(seed).astype(jnp.uint32).reshape(())
    row = jnp.arange(M, dtype=jnp.int32)[:, None]
    col = jnp.arange(n, dtype=jnp.int32)[None, :]
    return jnp.stack([_gumbel_from(s, g, row, col, n) for g in range(G)])


def _topk_rounds(x, k, col, n):
    """k rounds of (max, lowest index at the max, mask out): [TM, n] -> vals, idx [TM, k]."""
    out_col = jax.lax.broadcasted_iota(jnp.int32, (x.shape[0], k), 1)
    vals = jnp.zeros((x.shape[0], k), jnp.float32)
    idxs = jnp.zeros((x.shape[0], k), jnp.int32)
    for j in range(k):
        m = jnp.max(x, axis=1, keepdims=True)
        i = jnp.min(jnp.where(x == m, col, n), axis=1, keepdims=True)
        vals = jnp.where(out_col == j, m, vals)
        idxs = jnp.where(out_col == j, i, idxs)
        x = jnp.where(col == i, -jnp.inf, x)
    return vals, idxs


def _kth_largest(x, k):
    """k-th largest per row: k rounds of max, masking every entry equal to the
    max (no index search; exact unless values tie exactly)."""
    for _ in range(k - 1):
        m = jnp.max(x, axis=1, keepdims=True)
        x = jnp.where(x == m, -jnp.inf, x)
    return jnp.max(x, axis=1, keepdims=True)


def _fwd_kernel(t_ref, seed_ref, scale_ref, q_ref, k_ref, idx_ref, sub_ref, sel_ref, mem_ref, p_ref, agr_ref,
                *, k, clean, noisy, m_valid):
    g_count = q_ref.shape[0]
    tm = q_ref.shape[1]
    n = k_ref.shape[1]
    t = t_ref[0]
    seed = seed_ref[0].astype(jnp.uint32)
    scale = scale_ref[0]
    col = jax.lax.broadcasted_iota(jnp.int32, (tm, n), 1)
    row = jax.lax.broadcasted_iota(jnp.int32, (tm, 1), 0) + pl.program_id(0) * tm
    valid = (row < m_valid).astype(jnp.float32)  # padding rows add nothing to the statistics
    agree = jnp.zeros((1, 1), jnp.float32)
    for g in range(g_count):
        cos = jax.lax.dot_general(q_ref[g], k_ref[g], (((1,), (1,)), ((), ())),
                                  preferred_element_type=jnp.float32)  # [tm, n]
        scores = cos * t
        if noisy:
            sel = scores + scale * _gumbel_from(seed, g, row, col, n)
        else:
            sel = scores
        sel_vals, idx = _topk_rounds(sel, k, col, n)
        # clean score at each pick: the noisy value minus the recomputed noise
        # (elementwise, no reduction over the n sub-keys)
        sub = sel_vals - scale * _gumbel_from(seed, g, row, idx, n) if noisy else sel_vals
        if clean and noisy:
            kth = _kth_largest(scores, k)
            member = (scores >= kth).astype(jnp.float32)
            agree = agree + jnp.sum((sub >= kth).astype(jnp.float32) * valid, keepdims=True)
        else:
            member = (sel >= sel_vals[:, k - 1:k]).astype(jnp.float32)
        mem_ref[0, g:g + 1, :] = jnp.sum(member * valid, axis=0, keepdims=True)
        e = jnp.exp(scores - jnp.max(scores, axis=1, keepdims=True))
        s_ = e / jnp.sum(e, axis=1, keepdims=True)
        p_ref[0, g:g + 1, :] = jnp.sum(s_ * valid, axis=0, keepdims=True)
        idx_ref[g] = idx
        sub_ref[g] = sub
        sel_ref[g] = sel_vals
    agr_ref[0] = agree


def _bwd_kernel(t_ref, q_ref, k_ref, idx_ref, gsub_ref, gp_ref, dq_ref, dk_ref, dt_ref, *, k, m_valid):
    g_count = q_ref.shape[0]
    tm = q_ref.shape[1]
    n = k_ref.shape[1]
    t = t_ref[0]
    col = jax.lax.broadcasted_iota(jnp.int32, (tm, n), 1)
    row = jax.lax.broadcasted_iota(jnp.int32, (tm, 1), 0) + pl.program_id(0) * tm
    valid = (row < m_valid).astype(jnp.float32)
    dt = jnp.zeros((1, 1), jnp.float32)
    for g in range(g_count):
        qg, kg = q_ref[g], k_ref[g]
        cos = jax.lax.dot_general(qg, kg, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)
        scores = cos * t
        # d sub_scores -> d scores at the picked sub-keys
        dsel = jnp.zeros((tm, n), jnp.float32)
        idx, gs = idx_ref[g], gsub_ref[g]
        for j in range(k):
            dsel = dsel + jnp.where(col == idx[:, j:j + 1], gs[:, j:j + 1], 0.0)
        # d p_sum -> d scores through the softmax (temperature not differentiated)
        e = jnp.exp(scores - jnp.max(scores, axis=1, keepdims=True))
        s = e / jnp.sum(e, axis=1, keepdims=True)
        gp = gp_ref[g:g + 1, :]  # [1, n]
        dp = s * (gp - jnp.sum(s * gp, axis=1, keepdims=True)) * valid
        dt = dt + jnp.sum(cos * dsel, keepdims=True)
        dcos = t * (dsel + dp)
        dq_ref[g] = jax.lax.dot_general(dcos, kg, (((1,), (0,)), ((), ())), preferred_element_type=jnp.float32)
        dk_ref[0, g] = jax.lax.dot_general(dcos, qg, (((0,), (0,)), ((), ())), preferred_element_type=jnp.float32)
    dt_ref[0] = dt


def _pad_rows(x, m_pad, axis):
    pad = m_pad - x.shape[axis]
    if pad == 0:
        return x
    widths = [(0, 0)] * x.ndim
    widths[axis] = (0, pad)
    return jnp.pad(x, widths)


def _forward(q, keys, t, seed, scale, k, clean, noisy):
    """q [G, M, d], keys [G, n, d], t [] temperature, seed [] int32, scale []
    noise scale (used when noisy)."""
    G, M, d = q.shape
    n = keys.shape[1]
    tm = min(TM, -(-M // 8) * 8)
    m_pad = -(-M // tm) * tm
    nb = m_pad // tm
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    gk = lambda i: (0, i, 0)
    out = pl.pallas_call(
        functools.partial(_fwd_kernel, k=k, clean=clean, noisy=noisy, m_valid=M),
        grid=(nb,),
        in_specs=[smem, smem, smem,
                  pl.BlockSpec(block_shape=(G, tm, d), index_map=gk),
                  pl.BlockSpec(block_shape=(G, n, d), index_map=lambda i: (0, 0, 0))],
        out_specs=[pl.BlockSpec(block_shape=(G, tm, k), index_map=gk),
                   pl.BlockSpec(block_shape=(G, tm, k), index_map=gk),
                   pl.BlockSpec(block_shape=(G, tm, k), index_map=gk),
                   pl.BlockSpec(block_shape=(1, G, n), index_map=lambda i: (i, 0, 0)),
                   pl.BlockSpec(block_shape=(1, G, n), index_map=lambda i: (i, 0, 0)),
                   pl.BlockSpec(block_shape=(1, 1, 1), index_map=lambda i: (i, 0, 0))],
        out_shape=[jax.ShapeDtypeStruct((G, m_pad, k), jnp.int32),
                   jax.ShapeDtypeStruct((G, m_pad, k), jnp.float32),
                   jax.ShapeDtypeStruct((G, m_pad, k), jnp.float32),
                   jax.ShapeDtypeStruct((nb, G, n), jnp.float32),
                   jax.ShapeDtypeStruct((nb, G, n), jnp.float32),
                   jax.ShapeDtypeStruct((nb, 1, 1), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",), vmem_limit_bytes=VMEM_LIMIT),
    )(jnp.reshape(t, (1,)).astype(jnp.float32), jnp.reshape(seed, (1,)).astype(jnp.int32),
      jnp.reshape(scale, (1,)).astype(jnp.float32), _pad_rows(q, m_pad, 1), keys)
    idx, sub, sel, mem, p, agr = out
    return (idx[:, :M], sub[:, :M], sel[:, :M], mem.sum(0), p.sum(0), agr.sum())


def _backward(q, keys, t, idx, g_sub, g_p, k):
    G, M, d = q.shape
    n = keys.shape[1]
    tm = min(TM, -(-M // 8) * 8)
    m_pad = -(-M // tm) * tm
    nb = m_pad // tm
    gk = lambda i: (0, i, 0)
    dq, dk, dt = pl.pallas_call(
        functools.partial(_bwd_kernel, k=k, m_valid=M),
        grid=(nb,),
        in_specs=[pl.BlockSpec(memory_space=pltpu.SMEM),
                  pl.BlockSpec(block_shape=(G, tm, d), index_map=gk),
                  pl.BlockSpec(block_shape=(G, n, d), index_map=lambda i: (0, 0, 0)),
                  pl.BlockSpec(block_shape=(G, tm, k), index_map=gk),
                  pl.BlockSpec(block_shape=(G, tm, k), index_map=gk),
                  pl.BlockSpec(block_shape=(G, n), index_map=lambda i: (0, 0))],
        out_specs=[pl.BlockSpec(block_shape=(G, tm, d), index_map=gk),
                   pl.BlockSpec(block_shape=(1, G, n, d), index_map=lambda i: (i, 0, 0, 0)),
                   pl.BlockSpec(block_shape=(1, 1, 1), index_map=lambda i: (i, 0, 0))],
        out_shape=[jax.ShapeDtypeStruct((G, m_pad, d), jnp.float32),
                   jax.ShapeDtypeStruct((nb, G, n, d), jnp.float32),
                   jax.ShapeDtypeStruct((nb, 1, 1), jnp.float32)],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",), vmem_limit_bytes=VMEM_LIMIT),
    )(jnp.reshape(t, (1,)).astype(jnp.float32), _pad_rows(q, m_pad, 1), keys,
      _pad_rows(idx, m_pad, 1), _pad_rows(g_sub, m_pad, 1), g_p)
    return dq[:, :M], dk.sum(0), dt.sum()


@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7))
def fused_router(q, keys, t, seed, scale, k, clean, noisy):
    """See the module docstring. q [G, M, d] and keys [G, n, d] unit-norm, t
    the (clamped) temperature; with noisy, selection uses scores + scale *
    Gumbel noise generated in the kernel from `seed` (router_noise() gives the
    same noise as an array)."""
    return _forward(q, keys, t, seed, scale, k, clean, noisy)


def _vjp_fwd(q, keys, t, seed, scale, k, clean, noisy):
    out = _forward(q, keys, t, seed, scale, k, clean, noisy)
    return out, (q, keys, t, out[0])


def _vjp_bwd(k, clean, noisy, res, cot):
    q, keys, t, idx = res
    _, g_sub, _, _, g_p, _ = cot
    dq, dk, dt = _backward(q, keys, t, idx, jnp.asarray(g_sub, jnp.float32), jnp.asarray(g_p, jnp.float32), k)
    return dq, dk, dt.astype(jnp.asarray(t).dtype), None, None


fused_router.defvjp(_vjp_fwd, _vjp_bwd)
