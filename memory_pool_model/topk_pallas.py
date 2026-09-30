"""Exact top-k over the last axis as a Pallas TPU kernel.

XLA lowers lax.top_k on TPU to a full sort of every row: 7.5 ms for the
router's 65,536 rows x 512 sub-keys at k=16 (TPU v5e), four times per
training step. This kernel keeps a block of rows in VMEM and runs k rounds
of (row max, lowest index at the max, mask it out), which is exact and
breaks ties towards the lower index like lax.top_k.

Indices only carry no gradient; callers that need differentiable values
read them from the scores at the returned indices.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

ROWS_PER_BLOCK = 256


def supported(width: int, k: int) -> bool:
    """Shapes the kernel handles: lane-aligned rows, small k."""
    return width % 128 == 0 and width <= 4096 and k <= 64


def _kernel(x_ref, v_ref, i_ref, *, k: int):
    x = x_ref[...].astype(jnp.float32)  # [TR, W]
    width = x.shape[1]
    col = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    out_col = jax.lax.broadcasted_iota(jnp.int32, (x.shape[0], k), 1)
    vals = jnp.zeros((x.shape[0], k), jnp.float32)
    idxs = jnp.zeros((x.shape[0], k), jnp.int32)
    for j in range(k):
        m = jnp.max(x, axis=1, keepdims=True)
        i = jnp.min(jnp.where(x == m, col, width), axis=1, keepdims=True)
        vals = jnp.where(out_col == j, m, vals)
        idxs = jnp.where(out_col == j, i, idxs)
        x = jnp.where(col == i, -jnp.inf, x)
    v_ref[...] = vals.astype(v_ref.dtype)
    i_ref[...] = idxs


@functools.partial(jax.jit, static_argnames=("k", "interpret"))
def top_k(x: jax.Array, k: int, interpret: bool = False):
    """(values, indices) of the k largest entries of each row of 2-D x.
    Rows must not contain NaN (a NaN row returns index `width`)."""
    x = jax.lax.stop_gradient(x)
    rows, width = x.shape
    tr = min(ROWS_PER_BLOCK, max(8, -(-rows // 8) * 8))
    pad = (-rows) % tr
    if pad:
        x = jnp.pad(x, ((0, pad), (0, 0)))
    n = x.shape[0]

    def rows_block(cols):
        return pl.BlockSpec(block_shape=(tr, cols), index_map=lambda r: (r, 0))

    vals, idx = pl.pallas_call(
        functools.partial(_kernel, k=k),
        grid=(n // tr,),
        in_specs=[rows_block(width)],
        out_specs=[rows_block(k), rows_block(k)],
        out_shape=[jax.ShapeDtypeStruct((n, k), x.dtype), jax.ShapeDtypeStruct((n, k), jnp.int32)],
        interpret=interpret,
    )(x)
    return vals[:rows], idx[:rows]
