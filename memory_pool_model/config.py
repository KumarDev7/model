"""Configuration dataclasses for the memory-pool language model."""

from __future__ import annotations

import dataclasses
from typing import Tuple


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    # ---- backbone (kept deliberately small: it learns *how* to answer) ----
    vocab_size: int = 256
    max_len: int = 128
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    ffn_mult: int = 1
    dropout: float = 0.0

    # ---- memory pool (holds the *knowledge*) ----
    use_memory: bool = True
    # Backbone layers that read from the (single, shared) pool.
    memory_layers: Tuple[int, ...] = (1,)
    # Keep the feed-forward block inside memory layers. False = the pool
    # read replaces it (memory-layer style), removing a place to store facts.
    memory_ffn: bool = True
    # Product-key pool: the pool has n_sub_keys**2 trainable value slots.
    n_sub_keys: int = 64
    # Independent router heads; each head fetches `top_k` slots.
    pool_heads: int = 4
    # Query/key dimension per head (split in two halves for product keys).
    d_key: int = 64
    # Dimension of each stored knowledge vector.
    d_value: int = 128
    # Number of slots fetched per head per token.
    top_k: int = 16
    # Scale of Gumbel noise added to routing scores while training
    # (exploration; prevents the router from locking onto a few slots early).
    routing_noise: float = 1.0
    # Initial inverse temperature for cosine routing scores (learnable).
    init_temperature: float = 10.0

    @property
    def pool_size(self) -> int:
        return self.n_sub_keys**2


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    steps: int = 3000
    batch_size: int = 64
    lr: float = 3e-3
    warmup_steps: int = 200
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    # Pool values are updated sparsely (only fetched rows get gradient), so
    # they get a larger learning rate than the backbone.
    pool_lr_mult: float = 3.0

    # ---- anti-collapse ----
    # Switch-style load-balancing loss on the router (sub-key) distribution.
    balance_coef: float = 0.01
    # Pushes sub-keys apart so they tile the query space.
    key_diversity_coef: float = 0.01
    # EMA decay for tracked sub-key / slot usage.
    usage_ema_decay: float = 0.99
    # Every `revive_every` steps, sub-keys whose EMA usage is below
    # `revive_threshold * uniform_usage` are re-initialised onto live queries.
    revive_every: int = 100
    revive_threshold: float = 0.1
    # Stop reviving after this fraction of training so the pool can settle.
    revive_until: float = 0.8

    # ---- make the pool, not the backbone, carry the knowledge ----
    # Block answer-loss gradients through the residual/FFN path of memory
    # layers; earlier backbone weights learn only via the router -> pool read.
    route_through_pool: bool = False
    # Extra pass with the pool switched off: push that prediction toward
    # uniform on scored tokens, so the backbone can't answer on its own.
    nopool_kl_coef: float = 0.0

    seed: int = 0
    log_every: int = 100
    eval_every: int = 500
