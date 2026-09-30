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
    # Filled in by the Trainer: registry name of the mesh of a row-sharded
    # pool (TrainConfig.pool_sharding).
    pool_mesh: str = ""
    # Filled in by the Trainer under data parallelism: registry name of the
    # mesh, so Pallas kernels (which XLA cannot partition) run per device.
    dp_mesh: str = ""
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
    # It must stay small next to the gaps between top routing scores, which
    # shrink as n_sub_keys grows. At 1.0 on Ultra-FineWeb (512 sub-keys) only
    # 0.4% of the slots read in training were the ones inference reads: the
    # pool learned nothing and the backbone ignored it. At 0.1 about 90% match
    # and the pool is used; the fact task reached 100% (99.99% at 1.0).
    # Watch pick_agreement in the training log.
    routing_noise: float = 0.1
    # Initial inverse temperature for cosine routing scores (learnable).
    init_temperature: float = 10.0
    # Lower bound of the learnable temperature, and whether the balance loss
    # may change it (see MemoryPool). With the old values (1.0, True) the
    # temperature fell to 1.0 in 7 of 9 fact-task runs: near-uniform mixing,
    # routing concentrated on 27-52% of the pool, 75.8-99.5% accuracy. With
    # these defaults 6 of 6 runs reached 99.92-99.99% with 99.95-100% of the
    # pool active (experiments/memorization_ablation.py).
    min_temperature: float = 10.0
    balance_temperature_grad: bool = False
    max_temperature: float = 100.0
    # Scale routing scores by each query's length (per-token sharpness).
    router_query_scale: bool = False
    # Initial pool values are normal(0, value_init_scale / sqrt(d_value));
    # 0 starts the pool at zero. It did not change whether the backbone used
    # the pool on Ultra-FineWeb (routing noise was the cause, see above).
    value_init_scale: float = 1.0
    # Measure balance-loss usage on noise-free picks (a second sub-key top-k
    # per memory layer). False reuses the training picks: cheaper, and with
    # small routing noise nearly the same; pick_agreement is then NaN.
    balance_on_clean_picks: bool = True

    # Dtype of the backbone's matmuls ("float32", "bfloat16", "float16").
    # Parameters, residual stream, layer norms, routing and the pool stay
    # float32. float16 (for GPUs without bf16, e.g. T4) trains with dynamic
    # loss scaling. T4, d512 x 6 dense: the float32 step is bound by matmuls.
    compute_dtype: str = "float32"

    @property
    def pool_size(self) -> int:
        return self.n_sub_keys**2


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # The pool learns slower than a dense backbone (it must settle where each
    # fact lives, and a slot only learns when read). Fact task, 2 seeds:
    # 3,000 steps left 257-357 facts wrong, 6,000 left 1-8.
    steps: int = 6000
    batch_size: int = 64
    lr: float = 3e-3
    warmup_steps: int = 200
    weight_decay: float = 0.01
    # Weight-decay the pool read path (router, read gate, read projection).
    # Turning it off did not change pool use on Ultra-FineWeb.
    decay_pool_path: bool = True
    grad_clip: float = 1.0
    # Adam beta2 for the dense parameters (the pool's lazy Adam keeps
    # 0.999). 0.95 is the usual choice for long language-model runs: with
    # 0.999 one of three d512 x 6 pool runs (bfloat16, 120k steps) had its
    # gradient norm grow from 0.3 to 1e4 after 90k steps and lost 0.34 nats.
    adam_b2: float = 0.999
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
    # Anneal the routing noise linearly from full strength at this fraction
    # of training to zero at `noise_anneal_end`, so the last steps train on
    # the same (clean) rows inference reads. 1.0 = keep the noise on.
    noise_anneal_start: float = 1.0
    noise_anneal_end: float = 1.0

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
    # accuracy with the pool and 1.8% without it. That low no-pool number is
    # what the penalty trains for; eval acc_shuffled_pool is the independent
    # check. The CLI turns it off for --task text/tokens (every token scored).
    nopool_true_coef: float = 1.0
    # Warm starts: switch the options above on only after this many steps
    # (0 = from the start, or never for the *_after_step switches).
    nopool_after_step: int = 0
    route_after_step: int = 0
    # Two-stage training: after this step the backbone is frozen and only
    # the pool path (pool, router, read gate/projection) keeps learning.
    freeze_backbone_after_step: int = 0
    # Update only the pool's value vectors (everything else frozen): adding
    # knowledge to a trained model by writing to the pool alone.
    pool_values_only: bool = False

    # ---- scale ----
    # Update only the pool rows fetched this step (lazy Adam). Gradient and
    # optimizer work then scale with rows used, not with pool size.
    sparse_pool_updates: bool = True
    # Optimizer for the pool values: "adam" (default; lazy Adam on the rows
    # read each step, 2 moments per value, so state is 2x the table) or
    # "rowwise_adagrad" (1 float per row, ~1x the table in total, for pools
    # too big to hold 3x). Row-wise Adagrad's step shrinks as a row gets
    # read, so often-read rows nearly stop learning: on TinyStories lazy Adam
    # made the pool's gain 4x larger (4,000 steps, loss 1.776 vs 1.835; dense
    # backbone 1.857). Fact task, 2 seeds: 100% with either (Adagrad 99.99%).
    # (With the old recipe Adagrad had looked better: 99.8% vs 97.9%.)
    pool_optimizer: str = "adam"
    # How the sparse update gathers the fetched rows' gradients: "dense"
    # scatter-adds them into a zero [pool_size, D] table and updates the
    # touched rows with a masked pass over the table (no sort; faster on
    # TPU, but its work grows with pool size; data parallel all-reduces the
    # table); "unique" sorts the fetched slots and works on those rows only
    # (host pools, very large pools). "auto": dense for a device pool on TPU
    # (one v5e: 19 + 2 ms vs 32 ms per step), unique elsewhere.
    pool_row_grads: str = "auto"
    # With data_parallel: shard the pool values (and their Adam moments) by
    # rows across the devices instead of replicating them; only the rows read
    # move between devices (sharded_pool.py). Buckets hold capacity x the
    # average reads per device pair; overflowing reads are dropped and logged
    # (shard_dropped).
    pool_sharding: bool = False
    pool_shard_capacity: float = 2.0
    # Split each batch across all local devices (data parallel).
    data_parallel: bool = False
    # Save a resumable checkpoint every N steps (0 = only at the end).
    checkpoint_every: int = 0

    seed: int = 0
    log_every: int = 100
    eval_every: int = 500
