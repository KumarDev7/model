# Memory-Pool LM (JAX / Flax)

A language model that splits **"how to answer"** from **"what it knows"**:

* a **small transformer backbone** learns reasoning/format (*how* to answer),
* a large **trainable memory pool** of vectors stores the knowledge,
* a **router** fetches the top-k most relevant pool vectors for every token
  in real time during the forward pass.

It is similar to RAG, with one difference: RAG stores embeddings of real
documents in a vector DB and retrieves text. Here the pool holds
**no text**. Both the pool's addressing keys and its value vectors are
parameters, trained end-to-end with the backbone. The pool ends up holding
the model's *generalised, compressed* knowledge.

```
tokens ─► embed ─► [attn ─► FFN(small) ─┬─► + ] ─► ... ─► logits
                                         │
                  hidden ─► router ─► query ─► top-k over pool ─► Σ w·value ─► gate ─► proj
                                                   │
                                  ┌────────────────┴────────────────┐
                                  │  MEMORY POOL  (n_sub_keys² slots)│
                                  │  sub_keys [H,2,n,d/2]  (address) │
                                  │  values   [N, d_value] (knowledge)│
                                  └─────────────────────────────────┘
```

## The goal: thinking on the GPU, knowledge on the SSD

The backbone that does the reasoning is small and stays on the GPU. The
knowledge lives in the pool, which can be far larger than GPU memory and
sit on an SSD. For each token the router picks `pool_heads * top_k` rows
(64 by default) per memory layer, and only those rows are read:

```
token → backbone (GPU) → router → product keys (GPU) → 64 slot ids
                                                        │  read 64 rows (fp16, 256 dims = 32 KB)
                                                        ▼
                                          pool values on SSD (memmap)
```

Three things make this work:

1. **Knowledge must really be in the pool.** With the defaults
   (`memory_ffn=False`, `nopool_true_coef=1.0`) the fact task reaches 97.9%
   accuracy, and only 1.8% with the pool removed. The old recipe gave 99.7% / 56.1%:
   the backbone had memorised half the facts itself.
   See `experiments/results/reliance*/`.
2. **The pool can live off the GPU.** `pool_location="host"` keeps the value
   table in RAM, or on SSD with `pool_dir`. Training fetches only the rows a
   step uses and updates them on the host with lazy Adam. Inference reads
   64 rows per token per memory layer.
3. **Updates touch only fetched rows** (`sparse_pool_updates`, default on).
   Gradients are built for just those rows, and they are tested equal to
   dense gradients.

## Validated on 2x NVIDIA T4

All numbers from `experiments/results/final/gpu_validation.json` and
`experiments/results/reliance2/`. Test suite: 27/27 on CPU and on GPU.

| Check | Result |
|---|---|
| Knowledge in the pool (fact task, 16,384 facts) | **99.95-99.99%** accuracy (1-8 of 16,384 facts wrong; 2 runs with the defaults, 4 more with noise fade-out or 6,000 steps on top), **0%** with the pool removed, 99.9-100% of the pool active, every slot reached (defaults: no memory-layer FFN, no-pool penalty, row-wise Adagrad, routing-temperature fix; `experiments/results/memorization2/`, `memorization3/`). Before the temperature fix: 99.8% in the best run, 75.8% in the worst |
| Sparse pool gradients | equal to dense gradients (test); only fetched rows change |
| Pool bigger than GPU memory, training | 4.2M rows in host RAM: 2.25 s/step, 2.5 GB GPU, 4.3 GB RAM (on-GPU version runs out of memory) |
| Same accuracy with the pool off the GPU | 98.3% host pool vs 97.9% device pool (same recipe, Adam) |
| Data parallel, 2 GPUs | 22.0k vs 13.7k tokens/s (1.6x), 1.7 vs 3.1 GB per GPU, same loss |
| Preemption | 2-GPU text run killed at step 1,750, resumed from step 1,500, finished; bit-exact resume in tests |
| Inference, pool on SSD | 67M rows (34 GB, larger than the 31 GB RAM): **5.4 ms/token** cold, 4.4 ms warm; pool on GPU: 2.3 ms |
| End to end | trained Shakespeare model (1.9M backbone + 67M pool) generates **identical text** with its pool read from SSD (8.1 ms/token) |

Known limits: host-pool training is ~8x slower per step than an on-GPU
pool (host row updates and transfers); host pool + data parallel is not
supported yet; no mixed precision (T4 lacks bf16); the Shakespeare model
overfits the 1 MB corpus after step 2,000.

## Real text: Ultra-FineWeb (2x T4)

`experiments/prepare_ultrafineweb.py` streams Ultra-FineWeb English and
trains a 16k byte-level BPE; `experiments/text_study.py` trains five models
on the same 30M training tokens (6,000 steps x 8,192 tokens, 1.6 passes)
and evaluates held-out documents from a different file, the training text,
and Tiny Shakespeare (another domain). Same backbone everywhere: d_model 256,
4 layers, FFN x4. Pool: 262,144 vectors x 256 (67M parameters), read in
layers 1 and 3, 128 vectors per token. Results: `experiments/results/text/`.

| Model | Params | Held-out ppl | Shakespeare ppl | Train - held-out loss | Held-out ppl, pool removed |
|---|---|---|---|---|---|
| dense (backbone only) | 7.4M | **78.5** | 157.7 | +0.09 | - |
| pool, FFN kept (`memory_ffn`) | 75.3M | 78.6 | **153.5** | +0.09 | ~10^16 |
| pool (defaults: pool replaces FFN in 2 layers) | 74.3M | 88.8 | 174.9 | +0.07 | ~10^15 |
| pool, no no-pool penalty | 74.3M | 88.3 | 174.2 | +0.08 | 88.5 |
| pool, old temperature settings | 74.3M | 88.9 | 175.2 | +0.07 | ~10^14 |

What this shows:

* **No collapse on real text.** On held-out text 92-100% of vectors are read
  and 76-95% get a fair share, every sub-key is used, and pool use kept
  spreading through training (active 70% -> 94% from step 1,000 to 6,000).
  The first memory layer is less even than the second (spread 0.40 vs 0.56-0.61).
  Shakespeare concentrates on fewer vectors (spread 0.22-0.30) but 99.9% of
  its reads land on vectors held-out text also uses.
* **No overfitting.** Held-out loss is *below* training loss for every model
  (the held-out file is slightly easier), and the pool models' gap equals
  the dense model's: 67M pool parameters did not memorise the 30M training
  tokens at 1.6 passes.
* **The pool does not help language modelling at this scale.** With the FFN
  kept, it matches the dense model on held-out text and is slightly better
  on Shakespeare; replacing the FFN with the pool costs 13% perplexity.
  Without the no-pool penalty the backbone ignores the pool (removing it
  changes nothing); with the penalty everything is routed through it but
  the result is no better. Rare tokens show no gain either.
* **Routing is not selective on text.** The top vector's weight is 0.07-0.08,
  about 1/16: each head averages its 16 vectors almost evenly. The
  temperature stays at its floor (10) with the fix; with the old settings
  it drifted down to 5.6 over training (on the fact task it rose to 20-30,
  with top weight ~0.6). Making reads selective on text (sharper scores,
  fewer vectors per head) and far more training tokens are the next steps
  before the pool can hold knowledge the backbone lacks.

## Layout

| file | what |
|---|---|
| `memory_pool_model/memory.py` | `MemoryPool` (product-key top-k router + trainable values), `key_diversity_loss`, `revive_dead_keys` |
| `memory_pool_model/model.py` | `MemoryPoolLM`: small causal transformer; chosen layers read from the one shared pool |
| `memory_pool_model/train.py` | optimizer, train/eval steps, sparse pool updates, data parallel, checkpoints/resume, CLI |
| `memory_pool_model/host_pool.py` | the pool kept off the GPU: host RAM or memory-mapped files on SSD |
| `memory_pool_model/generate.py` | token-by-token generation with a KV cache |
| `memory_pool_model/data.py` | synthetic knowledge-base task and byte-level text task |
| `memory_pool_model/config.py` | `ModelConfig`, `TrainConfig` (every field is a CLI flag) |
| `tests/` | correctness tests (exact top-k, collapse, revival, sparse = dense gradients, data parallel, exact resume, host pool, decoding) |
| `experiments/` | knowledge tests, pool-reliance study, scaling and SSD-inference benchmarks |

## How retrieval works

The pool uses **product keys** (Lample et al., 2019) so lookup stays cheap
for very large pools. Each router query is split into two halves; each half
is scored against `n_sub_keys` sub-keys, the top-k of each half is taken, and
the best k of the `k × k` combinations are the fetched slots. This is the
**exact** top-k over all `n_sub_keys²` slots while scoring only
`2 · n_sub_keys` keys (verified against brute force in the tests). With
`n_sub_keys=1024` the pool has ~1M slots.

Several router heads each fetch `top_k` slots; their softmax-weighted values
are summed, gated by the hidden state, and added to the residual stream.
Several backbone layers can read from the **same** pool, each with its own
router.

## Keeping the pool from collapsing

A trainable top-k memory tends to collapse: a few slots win early, only they
get gradients, they get better, and they win even more. The rest of the pool
is never used. This model uses six mechanisms against that:

1. **Cosine routing.** Queries and keys are L2-normalised and multiplied by a
   learned temperature, so no key can win just by growing its norm.
2. **Gumbel routing noise** (`routing_noise`) while training. Noise changes
   only *which* slots are picked (exploration), never the mixing weights. It
   is off at inference.
3. **Load-balancing loss** (`balance_coef`). This is a Switch-Transformer-style
   `n · Σ f_i · P_i` over every sub-key codebook, where `f` is the fraction of
   hard top-k picks and `P` is the mean router probability. It is computed on
   the *clean* (noise-free) router so noise can't hide a collapse. It equals
   1.0 for uniform routing and grows as routing concentrates.
4. **Key-diversity loss** (`key_diversity_coef`). This is the mean squared
   off-diagonal cosine similarity between sub-keys, which pushes the keys to
   tile the query space evenly.
5. **Dead-key revival** (`revive_every`, `revive_threshold`). This works like
   VQ-VAE codebook restarts. EMA usage is tracked for every sub-key. Keys used
   less than `threshold × uniform` are moved onto recently seen real queries,
   and their Adam moments are cleared. Revival stops after `revive_until` of
   training so the pool can settle.

6. **Routing-temperature guard** (`min_temperature`, `balance_temperature_grad`).
   The balance loss could be lowered by flattening the router softmax
   (lowering the learned temperature) instead of spreading usage, and a plain
   clip had zero gradient at its lower bound. In 7 of 9 runs with the old
   settings the temperature fell to 1.0 and stayed there: the 16 fetched
   vectors per head were mixed almost equally (top weight 0.07), the Gumbel
   noise decided every training pick, so usage *measured during training*
   looked even while inference routing concentrated on 27-52% of the pool.
   Those runs reached 75.8-99.5%. Now the temperature has a floor of 10 with
   a straight-through clamp, and the balance loss can't change it: 6 of 6
   runs reached 99.92-99.99% with 99.95-100% of the pool active
   (`experiments/memorization_ablation.py`, rounds 1-2). Fading the routing
   noise out at the end (`noise_anneal_start`) or training 6,000 steps on
   top of the fix gave 2-4 facts wrong, within the 1-13 spread between
   seeds, so both stay off by default (round 3). Results (per-run summaries
   and accuracy by step) are in `experiments/results/memorization*/`; round 1
   ran before the fix existed, so it used the old settings. Watch
   `temperature` in the training log: it should rise (to ~20-30), not fall.

Pool values also get a higher learning rate (`pool_lr_mult`), because each
slot only gets gradient when it is fetched. They are excluded from weight
decay so knowledge in rarely used slots is not erased.

### Monitoring

These metrics are logged during training:

* `slot_spread_*` is `exp(entropy(usage)) / N`: **1.0 = perfectly uniform
  use of the pool, 1/N = total collapse**. `_batch` is the current step and
  `_ema` is the running average.
* `slot_active_ema` is the fraction of slots with non-negligible use.
* `subkey_spread` is the same measure per sub-key codebook.
* `balance_loss` is ≈1.0 when balanced.
* `temperature` is the learned routing sharpness. It should rise during
  training; a temperature stuck at its floor means the mixing is uniform and
  training-time usage numbers are dominated by the routing noise.
* eval `pool_coverage` is the fraction of slots fetched at least once over the
  eval set with *no* noise, and eval `pool_spread` is the spread of that usage.

## Usage

```bash
pip install -r requirements.txt

# Synthetic knowledge base: 4096 entities x 4 relations = 16k facts.
# Entities are multi-token names, so facts can't hide in token embeddings.
python -m memory_pool_model.train --task facts --steps 3000

# Same backbone without the pool (baseline)
python -m memory_pool_model.train --task facts --steps 3000 --use_memory false

# Any text file, byte-level
python -m memory_pool_model.train --task text --text_path input.txt \
    --max_len 128 --n_sub_keys 256 --memory_layers 1,3 --n_layers 4

# Bigger pool (65,536 slots), save params
python -m memory_pool_model.train --n_sub_keys 256 --save checkpoints/pool.msgpack

python -m pytest tests -q
```

All `ModelConfig` / `TrainConfig` fields are CLI flags (`--top_k`,
`--pool_heads`, `--balance_coef`, `--routing_noise`, `--revive_every 0` to
disable revival, ...).

### Scaling

```bash
# pool in host RAM (or on SSD with --pool_dir), backbone on the GPU
python -m memory_pool_model.train --pool_location host --pool_dir /mnt/ssd/pool ...

# all local GPUs, data parallel; resumable checkpoints every 500 steps
python -m memory_pool_model.train --data_parallel true --checkpoint_every 500 --save ckpt/run --resume ...

# benchmarks
python -m experiments.scale_bench --n_sub 256 1024 2048            # step time / memory vs pool size
python -m experiments.ssd_inference --n_sub 2048 --where gpu ram ssd_warm ssd_cold
```

For bit-exact GPU runs set `XLA_FLAGS=--xla_gpu_deterministic_ops=true`;
otherwise GPU scatter-adds make runs differ in the last float bits.

### Generating tokens

```python
from memory_pool_model.generate import Generator
gen = Generator(mcfg, params)                    # works with device or host pools
out = gen.generate(prompt_tokens, n_new=64)      # greedy; temperature=... to sample
```

### Reading the retrieved slots

The model returns which slots it fetched, like RAG returns retrieved
documents:

```python
logits, aux = model.apply({"params": params}, tokens)
aux["slots"][layer]    # [B, T, heads, top_k] pool indices fetched per token
aux["weights"][layer]  # [B, T, heads, top_k] their mixing weights
```
