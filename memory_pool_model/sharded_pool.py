"""Pool value table sharded by rows across the data-parallel devices.

With a replicated pool every device holds the whole table and, every step,
the devices exchange the gradients of every row read (or all-reduce the whole
table): data parallel scaled 1.24-1.35x on 2 GPUs. Here device d owns rows
[d * N/P, (d + 1) * N/P) of the values and of their Adam moments, and only
the rows actually read move between devices:

  forward   each device buckets its reads by owner, an all-to-all sends the
            row ids to the owners, the owners gather their rows and a second
            all-to-all sends the vectors back;
  backward  each read's gradient contribution goes the same way to its owner,
            which scatter-adds into its own rows (lazy Adam then runs on the
            local rows only).

Buckets have a fixed size, `capacity` x the average number of reads per
(device, owner) pair. Reads beyond it are dropped (read as zero, no gradient)
and counted, as in mixture-of-experts token dropping; with enough capacity
nothing is dropped and training matches a single device (tested).
"""

from __future__ import annotations

import functools
import inspect
import math
import uuid
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

AXIS = "data"
_REGISTRY: Dict[str, Tuple[object, float]] = {}


def register(mesh, capacity: float) -> str:
    """Make a mesh available to the model by name (module attributes must be
    hashable); returns the name to put in ModelConfig.pool_mesh."""
    name = f"mesh-{uuid.uuid4().hex[:8]}"
    _REGISTRY[name] = (mesh, capacity)
    return name


def get(name: str):
    return _REGISTRY[name]


def _shard_map(f, mesh, in_specs, out_specs):
    try:
        from jax import shard_map
    except ImportError:  # older JAX
        from jax.experimental.shard_map import shard_map
    flag = "check_vma" if "check_vma" in inspect.signature(shard_map).parameters else "check_rep"
    return shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, **{flag: False})


def bucket_capacity(n_requests: int, n_dev: int, capacity: float) -> int:
    return max(8, int(math.ceil(capacity * n_requests / n_dev / 8.0)) * 8)


def _route(slots: jax.Array, rows_per: int, n_dev: int, cap: int):
    """Bucket local read requests by owner.

    slots [R] global row ids. Returns keep [R] (fits in its bucket), flat [R]
    (index into the flattened [n_dev * cap] buckets; n_dev * cap when
    dropped) and send [n_dev, cap] (local row id at the owner, -1 empty)."""
    owner = slots // rows_per
    local = slots % rows_per
    onehot = (owner[:, None] == jnp.arange(n_dev)).astype(jnp.int32)  # [R, P]
    pos = jnp.sum((jnp.cumsum(onehot, axis=0) - 1) * onehot, axis=1)  # rank within its owner's bucket
    keep = pos < cap
    flat = jnp.where(keep, owner * cap + pos, n_dev * cap)
    send = jnp.full((n_dev * cap,), -1, jnp.int32).at[flat].set(local.astype(jnp.int32), mode="drop")
    return keep, flat, send.reshape(n_dev, cap)


def _fetch_local(slots, values, *, n_dev, capacity):
    """Per device: slots [m, H, k] (global ids), values [N/P, D] (own rows)."""
    shape = slots.shape
    r = slots.reshape(-1)
    rows_per, d = values.shape
    cap = bucket_capacity(r.shape[0], n_dev, capacity)
    keep, flat, send = _route(r, rows_per, n_dev, cap)
    recv = jax.lax.all_to_all(send, AXIS, 0, 0, tiled=True)  # [n_dev, cap]: requests from each device
    rows = jnp.take(values, jnp.clip(recv, 0), axis=0)
    rows = jnp.where((recv >= 0)[..., None], rows, jnp.zeros((), values.dtype))  # [n_dev, cap, D]
    back = jax.lax.all_to_all(rows, AXIS, 0, 0, tiled=True)  # [n_dev, cap, D]: answers from each owner
    out = jnp.take(back.reshape(n_dev * cap, d), jnp.minimum(flat, n_dev * cap - 1), axis=0)
    out = jnp.where(keep[:, None], out, jnp.zeros((), values.dtype))
    dropped = jax.lax.psum(jnp.sum(~keep).astype(jnp.float32), AXIS)
    return out.reshape(*shape, d), dropped


def fetch(mesh_name: str, values: jax.Array, slots: jax.Array):
    """Rows `slots` [M, H, k] of the row-sharded table `values` [N, D], with
    M sharded over the data axis. Returns ([M, H, k, D], reads dropped)."""
    mesh, capacity = get(mesh_name)
    n_dev = mesh.shape[AXIS]
    f = functools.partial(_fetch_local, n_dev=n_dev, capacity=capacity)
    return _shard_map(f, mesh, in_specs=(P(AXIS), P(AXIS, None)), out_specs=(P(AXIS), P()))(slots, values)


def _row_grads_local(slots_list, weights_list, grads_list, *, rows_per, n_dev, capacity):
    """Per device: gradient of the loss w.r.t. its own rows, [N/P, D]."""
    d = grads_list[0].shape[-1]
    g_local = jnp.zeros((rows_per, d), jnp.float32)
    for sl, w, g in zip(slots_list, weights_list, grads_list):
        g = g.reshape(-1, 1, d)  # [bt, 1, D]
        sl = sl.reshape(g.shape[0], -1)  # [bt, H*k]
        w = jax.lax.stop_gradient(w).reshape(sl.shape)
        contrib = (w[..., None] * g).reshape(-1, d)  # one row per read
        r = sl.reshape(-1)
        cap = bucket_capacity(r.shape[0], n_dev, capacity)
        _, flat, send = _route(r, rows_per, n_dev, cap)
        buf = jnp.zeros((n_dev * cap, d), jnp.float32).at[flat].set(contrib, mode="drop")
        recv = jax.lax.all_to_all(buf.reshape(n_dev, cap, d), AXIS, 0, 0, tiled=True)
        recv_ids = jax.lax.all_to_all(send, AXIS, 0, 0, tiled=True)
        ids = jnp.where(recv_ids >= 0, recv_ids, rows_per).reshape(-1)
        g_local = g_local.at[ids].add(recv.reshape(-1, d), mode="drop")
    return g_local


def row_grads(mesh_name: str, pool_size: int, slots: List[jax.Array], weights: List[jax.Array],
              grads: List[jax.Array]) -> jax.Array:
    """Row-sharded [N, D] gradient of the pool values from each memory
    layer's reads (slots/weights [B, T, H, k]) and the gradient reaching each
    read (grads [B, T, D]); batch sharded over the data axis."""
    mesh, capacity = get(mesh_name)
    n_dev = mesh.shape[AXIS]
    f = functools.partial(_row_grads_local, rows_per=pool_size // n_dev, n_dev=n_dev, capacity=capacity)
    spec = [P(AXIS)] * len(slots)
    return _shard_map(f, mesh, in_specs=(spec, spec, spec), out_specs=P(AXIS, None))(
        list(slots), list(weights), list(grads))
