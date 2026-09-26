"""Small transformer backbone that reads knowledge from a shared memory pool."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

from .config import ModelConfig
from .memory import MemoryPool


class MemoryPoolLM(nn.Module):
    """Causal LM: small backbone + router + one trainable memory pool.

    In each layer listed in `cfg.memory_layers`, a per-layer router projects
    the hidden state into queries, the pool returns the top-k knowledge
    vectors, and a gated projection writes them back into the residual
    stream. All memory layers share the same pool.
    """

    cfg: ModelConfig
    # Autoregressive decoding with a KV cache (one token per call).
    decode: bool = False

    @nn.compact
    def __call__(
        self,
        tokens: jax.Array,
        *,
        train: bool = False,
        pool_off: bool = False,
        route_through_pool: bool = False,
        sparse_grad: bool = False,
        probes: Dict[int, jax.Array] | None = None,
        positions: jax.Array | None = None,
        noise_scale: jax.Array | float = 1.0,
    ) -> Tuple[jax.Array, Dict[str, Any]]:
        """
        pool_off: skip the memory read (the backbone answers alone).
        route_through_pool: in memory layers, block gradients through the
          residual/FFN path, so the loss can reach earlier backbone weights
          only through the router -> pool read. The backbone can still learn
          how to query the pool, but not store answers itself.
        sparse_grad / probes: used by the trainer's sparse pool update. The
          value table is read without gradient, and a zero probe is added to
          each layer's pool read so d(loss)/d(read) can be recovered.
        noise_scale: multiplier on the routing noise (annealed by the trainer).
        """
        cfg = self.cfg
        B, T = tokens.shape
        embed = nn.Embed(cfg.vocab_size, cfg.d_model, name="embed")
        pos = self.param(
            "pos_embed", nn.initializers.normal(0.02), (cfg.max_len, cfg.d_model)
        )
        if positions is None:
            x = embed(tokens) + pos[None, :T]
        else:
            x = embed(tokens) + pos[positions][None]
        # in decode mode the attention cache applies the causal mask itself
        causal = None if self.decode else nn.make_causal_mask(tokens)

        pool = None
        if cfg.use_memory:
            pool = MemoryPool(
                n_sub_keys=cfg.n_sub_keys,
                heads=cfg.pool_heads,
                d_key=cfg.d_key,
                d_value=cfg.d_value,
                top_k=cfg.top_k,
                routing_noise=cfg.routing_noise,
                init_temperature=cfg.init_temperature,
                host_pool=cfg.host_pool,
                name="pool",
            )

        layer_aux: List[Dict[str, Any]] = []
        for i in range(cfg.n_layers):
            h = nn.LayerNorm(name=f"ln_attn_{i}")(x)
            h = nn.MultiHeadDotProductAttention(
                num_heads=cfg.n_heads,
                dropout_rate=cfg.dropout,
                deterministic=not train,
                decode=self.decode,
                name=f"attn_{i}",
            )(h, h, mask=causal)
            x = x + h

            h = nn.LayerNorm(name=f"ln_ffn_{i}")(x)
            is_mem_layer = pool is not None and i in cfg.memory_layers
            y = jnp.zeros_like(x)
            if not is_mem_layer or cfg.memory_ffn:
                y = nn.Dense(cfg.d_model * cfg.ffn_mult, name=f"ffn_in_{i}")(h)
                y = nn.Dense(cfg.d_model, name=f"ffn_out_{i}")(nn.gelu(y))

            if is_mem_layer:
                if route_through_pool:
                    x, y = jax.lax.stop_gradient(x), jax.lax.stop_gradient(y)
                if not pool_off:
                    # Router: hidden state -> one query per pool head.
                    q = nn.Dense(cfg.pool_heads * cfg.d_key, name=f"router_{i}")(h)
                    q = q.reshape(B, T, cfg.pool_heads, cfg.d_key)
                    mem, aux = pool(q, train=train, sparse_grad=sparse_grad, noise_scale=noise_scale)
                    if probes is not None:
                        mem = mem + probes[i]
                    gate = nn.silu(nn.Dense(cfg.d_value, name=f"mem_gate_{i}")(h))
                    y = y + nn.Dense(cfg.d_model, name=f"mem_out_{i}")(mem * gate)
                    layer_aux.append(aux)

            y = nn.Dropout(cfg.dropout, deterministic=not train)(y)
            x = x + y

        x = nn.LayerNorm(name="ln_final")(x)
        logits = embed.attend(x)
        return logits, merge_layer_aux(layer_aux)


def merge_layer_aux(layer_aux: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not layer_aux:
        return {}
    return {
        "balance_loss": jnp.mean(jnp.stack([a["balance_loss"] for a in layer_aux])),
        "subkey_counts": sum(a["subkey_counts"] for a in layer_aux),
        "slot_counts": sum(a["slot_counts"] for a in layer_aux),
        # kept per layer: concatenating along the (batch-sharded) leading
        # axis forces an all-to-all under data parallelism
        "queries": [a["queries"] for a in layer_aux],
        "slots": [a["slots"] for a in layer_aux],
        "weights": [a["weights"] for a in layer_aux],
        "temperature": layer_aux[0]["temperature"],
    }
