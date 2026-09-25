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
    # Keep the feed-forward block inside memory layers. False (default) = the
    # pool read replaces it (memory-layer style), removing a place to store
    # facts. Measured: backbone-alone accuracy 56% -> 23% on the fact task.
    memory_ffn: bool = False
    # Where the value table lives: "device" (GPU/TPU memory) or "host"
    # (host RAM, or memory-mapped files on SSD when pool_dir is set).
    pool_location: str = "device"
    pool_dir: str = ""
    # Filled in by the Trainer: registry name of the HostPool (host mode).
    host_pool: str = ""
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
    # Same extra pass, penalising -log(1 - p_correct): the backbone is only
    # punished for knowing the right answer by itself.
    # Default 1.0: with memory_ffn=False this moved the fact task to 97.9%
    # accuracy with the pool and 1.8% without it (knowledge in the pool).
    nopool_true_coef: float = 1.0
    # Warm starts: switch the options above on only after this many steps
    # (0 = from the start, or never for the *_after_step switches).
    nopool_after_step: int = 0
    route_after_step: int = 0
    # Two-stage training: after this step the backbone is frozen and only
    # the pool path (pool, router, read gate/projection) keeps learning.
    freeze_backbone_after_step: int = 0

    # ---- scale ----
    # Update only the pool rows fetched this step (lazy Adam). Gradient and
    # optimizer work then scale with rows used, not with pool size.
    sparse_pool_updates: bool = True
    # Optimizer for the pool values: "rowwise_adagrad" (default; 1 float of
    # state per row, ~1x the table, standard for very large embedding
    # tables) or "adam" (2 moments per value, 3x). On the fact task
    # rowwise_adagrad reached 99.8% (0.01% without the pool) vs Adam's
    # 97.9% (1.8%), with more even pool usage. Needs sparse_pool_updates.
    pool_optimizer: str = "rowwise_adagrad"
    # Split each batch across all local devices (data parallel).
    data_parallel: bool = False
    # Save a resumable checkpoint every N steps (0 = only at the end).
    checkpoint_every: int = 0

    seed: int = 0
    log_every: int = 100
    eval_every: int = 500
