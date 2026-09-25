"""Token-by-token generation with a KV cache.

The backbone, router and product keys run on the accelerator. With
pool_location="host" the value table stays in host RAM or on SSD and each
new token copies in only the rows its router picks (pool_heads * top_k per
memory layer).
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np

from .config import ModelConfig
from .model import MemoryPoolLM


class Generator:
    def __init__(self, mcfg: ModelConfig, params, batch_size: int = 1):
        self.mcfg, self.params, self.batch_size = mcfg, params, batch_size
        self.model = MemoryPoolLM(mcfg, decode=True)
        dummy = jnp.zeros((batch_size, mcfg.max_len), jnp.int32)
        self._empty_cache = jax.jit(lambda: self.model.init(jax.random.PRNGKey(0), dummy)["cache"])
        self._step = jax.jit(self._step_fn)

    def _step_fn(self, params, cache, tokens, pos):
        (logits, aux), mut = self.model.apply(
            {"params": params, "cache": cache}, tokens[:, None], positions=pos[None], mutable=["cache"])
        return logits[:, 0], mut["cache"]

    def generate(self, prompt: np.ndarray, n_new: int, temperature: float = 0.0,
                 seed: int = 0) -> Dict[str, object]:
        """prompt: [B, P] int tokens. Greedy when temperature == 0.
        Returns tokens, all step logits, and per-token wall times (seconds)."""
        prompt = np.asarray(prompt, np.int32)
        B, P = prompt.shape
        if P + n_new > self.mcfg.max_len:
            raise ValueError("prompt + n_new exceeds max_len")
        cache = self._empty_cache()
        rng = jax.random.PRNGKey(seed)
        out, logits_all, times = [], [], []
        tok = jnp.asarray(prompt[:, 0])
        for pos in range(P + n_new - 1):
            t0 = time.perf_counter()
            logits, cache = self._step(self.params, cache, tok, jnp.int32(pos))
            if pos + 1 < P:
                nxt = jnp.asarray(prompt[:, pos + 1])
            elif temperature > 0:
                rng, sub = jax.random.split(rng)
                nxt = jax.random.categorical(sub, logits / temperature)
            else:
                nxt = jnp.argmax(logits, -1)
            nxt = jax.block_until_ready(nxt)
            times.append(time.perf_counter() - t0)
            logits_all.append(np.asarray(logits))
            if pos + 1 >= P:
                out.append(np.asarray(nxt))
            tok = nxt.astype(jnp.int32)
        return {"tokens": np.stack(out, 1) if out else np.zeros((B, 0), np.int32),
                "logits": np.stack(logits_all, 1), "step_seconds": np.array(times)}
