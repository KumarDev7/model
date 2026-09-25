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

    @nn.compact
    def __call__(
        self, tokens: jax.Array, *, train: bool = False
    ) -> Tuple[jax.Array, Dict[str, Any]]:
        cfg = self.cfg
        B, T = tokens.shape
        embed = nn.Embed(cfg.vocab_size, cfg.d_model, name="embed")
        pos = self.param(
            "pos_embed", nn.initializers.normal(0.02), (cfg.max_len, cfg.d_model)
        )
        x = embed(tokens) + pos[None, :T]
        causal = nn.make_causal_mask(tokens)

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
                name="pool",
            )

        layer_aux: List[Dict[str, Any]] = []
        for i in range(cfg.n_layers):
            h = nn.LayerNorm(name=f"ln_attn_{i}")(x)
            h = nn.MultiHeadDotProductAttention(
                num_heads=cfg.n_heads,
                dropout_rate=cfg.dropout,
                deterministic=not train,
                name=f"attn_{i}",
            )(h, h, mask=causal)
            x = x + h

            h = nn.LayerNorm(name=f"ln_ffn_{i}")(x)
            y = nn.Dense(cfg.d_model * cfg.ffn_mult, name=f"ffn_in_{i}")(h)
            y = nn.Dense(cfg.d_model, name=f"ffn_out_{i}")(nn.gelu(y))

            if pool is not None and i in cfg.memory_layers:
                # Router: hidden state -> one query per pool head.
                q = nn.Dense(cfg.pool_heads * cfg.d_key, name=f"router_{i}")(h)
                q = q.reshape(B, T, cfg.pool_heads, cfg.d_key)
                mem, aux = pool(q, train=train)
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
        "queries": jnp.concatenate([a["queries"] for a in layer_aux], axis=0),
        "slots": [a["slots"] for a in layer_aux],
        "weights": [a["weights"] for a in layer_aux],
        "temperature": layer_aux[0]["temperature"],
    }
