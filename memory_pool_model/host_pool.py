"""Pool value table kept off the accelerator: in host RAM or on SSD.

The backbone, router and product keys stay on the GPU. Only the rows the
router picks are copied in, inside the jitted step, through a host callback.
At inference that is pool_heads * top_k rows per token per memory layer
(64 by default), which is what lets the knowledge live on disk.

Training uses the same read path plus a lazy-Adam update of the fetched
rows on the host, so the pool (and its optimizer state) can be larger than
device memory.
"""

from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

_REGISTRY: Dict[str, "HostPool"] = {}

ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
ADAGRAD_EPS = 1e-8
_INIT_CHUNK = 1 << 18  # rows initialised per chunk (bounded temporary memory)
_THREADS = max(1, min(8, os.cpu_count() or 1))
_EXEC = ThreadPoolExecutor(_THREADS)
_MIN_CHUNK = 4096  # rows per thread task


def _chunks(n: int):
    step = max(_MIN_CHUNK, -(-n // _THREADS))
    return [(s, min(s + step, n)) for s in range(0, n, step)]


def _parallel(fn, n: int) -> None:
    """Run fn(start, end) over row chunks on a thread pool (numpy releases
    the GIL for these copies, so gathers/scatters use several cores)."""
    parts = _chunks(n)
    if len(parts) == 1:
        fn(*parts[0])
    else:
        list(_EXEC.map(lambda se: fn(*se), parts))


def get(name: str) -> "HostPool":
    return _REGISTRY[name]


class HostPool:
    """values [n_slots, dim] (+ optimizer state when trainable) in RAM or memmap.

    path=None keeps arrays in RAM; otherwise they are .npy memmaps under
    `path` (created if missing, reopened if present).

    optimizer="adam" keeps two [n_slots, dim] moments (3x the table);
    "rowwise_adagrad" keeps one float per row (about 1x), the usual choice
    for very large embedding tables.
    """

    def __init__(self, n_slots: int, dim: int, path: Optional[str] = None,
                 trainable: bool = True, dtype=np.float32, seed: int = 0, name: Optional[str] = None,
                 optimizer: str = "adam"):
        if optimizer not in ("adam", "rowwise_adagrad"):
            raise ValueError(f"unknown pool optimizer {optimizer!r}")
        try:  # rows cross via a host callback, which needs JAX's CPU backend
            jax.devices("cpu")
        except RuntimeError as e:
            raise RuntimeError(
                "HostPool needs JAX's CPU backend for its host callback; include it, "
                "e.g. JAX_PLATFORMS=cuda,cpu") from e
        self.n_slots, self.dim, self.path = n_slots, dim, path
        self._gather_s = 0.0
        self.trainable = trainable
        self.name = name or f"pool-{uuid.uuid4().hex[:8]}"
        self.optimizer = optimizer
        self.values = self._array("values", dtype, init=True, seed=seed)
        adam = trainable and optimizer == "adam"
        self.m = self._array("m", np.float32) if adam else None
        self.v = self._array("v", np.float32) if adam else None
        self.acc = (self._array("acc", np.float32, shape=(n_slots,))
                    if trainable and optimizer == "rowwise_adagrad" else None)
        _REGISTRY[self.name] = self

    # ------------------------------------------------------------- storage
    def _array(self, key: str, dtype, init: bool = False, seed: int = 0, shape=None):
        shape = shape or (self.n_slots, self.dim)
        if self.path is None:
            arr = np.zeros(shape, dtype)
            fresh = True
        else:
            os.makedirs(self.path, exist_ok=True)
            f = os.path.join(self.path, f"{key}.npy")
            fresh = not os.path.exists(f)
            arr = np.lib.format.open_memmap(f, mode="w+" if fresh else "r+", dtype=dtype, shape=shape)
            if not fresh and (arr.shape != shape):
                raise ValueError(f"{f} has shape {arr.shape}, expected {shape}")
        if init and fresh:
            # same distribution as the on-device pool: normal(0, dim**-0.5)
            rng = np.random.default_rng(seed)
            for s in range(0, self.n_slots, _INIT_CHUNK):
                e = min(s + _INIT_CHUNK, self.n_slots)
                arr[s:e] = (rng.standard_normal((e - s, self.dim)) * self.dim**-0.5).astype(dtype)
        return arr

    def _state(self):
        return {k: getattr(self, k) for k in ("values", "m", "v", "acc") if getattr(self, k) is not None}

    def nbytes(self) -> int:
        return sum(a.nbytes for a in self._state().values())

    # ---------------------------------------------------------------- reads
    def gather(self, idx: np.ndarray) -> np.ndarray:
        t = time.perf_counter()
        idx = np.clip(np.asarray(idx).astype(np.int64), 0, self.n_slots - 1)
        out = np.empty((idx.shape[0], self.dim), np.float32)

        def part(s, e):
            out[s:e] = np.take(self.values, idx[s:e], axis=0)

        _parallel(part, idx.shape[0])
        self._gather_s += time.perf_counter() - t
        return out

    def pop_gather_seconds(self) -> float:
        s, self._gather_s = self._gather_s, 0.0
        return s

    # --------------------------------------------------------------- update
    def update(self, uniq: np.ndarray, grads: np.ndarray, step: int, lr: float) -> None:
        """Optimizer step on the rows in `uniq` (entries >= n_slots are padding)."""
        uniq = np.asarray(uniq)
        keep = uniq < self.n_slots
        rows, g = uniq[keep].astype(np.int64), np.asarray(grads, np.float32)[keep]
        if rows.size == 0:
            return
        order = np.argsort(rows)  # sorted indices -> sequential-ish disk access
        rows, g = rows[order], g[order]
        vals = self.values

        if self.optimizer == "rowwise_adagrad":
            acc = self.acc[rows] + np.mean(g * g, axis=1)
            self.acc[rows] = acc
            scale = (lr / (np.sqrt(acc) + ADAGRAD_EPS)).astype(np.float32)

            def part(s, e):
                r = rows[s:e]
                vals[r] = (np.take(vals, r, axis=0).astype(np.float32) - scale[s:e, None] * g[s:e]).astype(vals.dtype)
        else:
            c1, c2 = 1 - ADAM_B1**step, 1 - ADAM_B2**step
            m_all, v_all = self.m, self.v

            def part(s, e):
                r, gg = rows[s:e], g[s:e]
                m = ADAM_B1 * np.take(m_all, r, axis=0) + (1 - ADAM_B1) * gg
                v = ADAM_B2 * np.take(v_all, r, axis=0) + (1 - ADAM_B2) * gg * gg
                m_all[r], v_all[r] = m, v
                upd = lr * (m / c1) / (np.sqrt(v / c2) + ADAM_EPS)
                vals[r] = (np.take(vals, r, axis=0).astype(np.float32) - upd).astype(vals.dtype)

        _parallel(part, rows.shape[0])

    adam_update = update  # backwards-compatible name

    # ----------------------------------------------------------- snapshots
    def flush(self) -> None:
        for a in self._state().values():
            if isinstance(a, np.memmap):
                a.flush()

    def save(self, dest: str) -> None:
        """Copy the current contents to `dest` (a resumable snapshot)."""
        os.makedirs(dest, exist_ok=True)
        self.flush()
        for key, a in self._state().items():
            tmp = os.path.join(dest, f"{key}.tmp.npy")
            np.save(tmp, a)
            os.replace(tmp, os.path.join(dest, f"{key}.npy"))

    def load(self, src: str) -> None:
        for key, a in self._state().items():
            f = os.path.join(src, f"{key}.npy")
            if os.path.exists(f):
                a[:] = np.load(f, mmap_mode="r")

    def evict_page_cache(self) -> None:
        """Drop these files from the OS page cache, so the next reads come
        from the SSD itself (cold-read benchmarks). No-op in RAM mode."""
        if self.path is None or not hasattr(os, "posix_fadvise"):
            return
        self.flush()
        for key in self._state():
            f = os.path.join(self.path, f"{key}.npy")
            if os.path.exists(f):
                fd = os.open(f, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)


def fetch(name: str, slots: jax.Array, dim: int) -> jax.Array:
    """Rows of the named host pool for `slots` (any shape), inside jit.

    Duplicates are removed on the device first, so each distinct row crosses
    the host/device boundary once per call.
    """
    flat = slots.reshape(-1)
    uniq, inv = jnp.unique(flat, size=flat.shape[0], fill_value=0, return_inverse=True)
    rows = jax.pure_callback(
        lambda idx: get(name).gather(idx),
        jax.ShapeDtypeStruct((flat.shape[0], dim), jnp.float32),
        uniq,
        vmap_method="sequential",
    )
    return rows[inv.reshape(-1)].reshape(slots.shape + (dim,))
