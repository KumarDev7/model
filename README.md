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

## Layout

| file | what |
|---|---|
| `memory_pool_model/memory.py` | `MemoryPool` (product-key top-k router + trainable values), `key_diversity_loss`, `revive_dead_keys` |
| `memory_pool_model/model.py` | `MemoryPoolLM`: small causal transformer; chosen layers read from the one shared pool |
| `memory_pool_model/train.py` | optimizer, train/eval steps, usage tracking, dead-key revival, CLI |
| `memory_pool_model/data.py` | synthetic knowledge-base task and byte-level text task |
| `memory_pool_model/config.py` | `ModelConfig`, `TrainConfig` (every field is a CLI flag) |
| `tests/` | correctness tests (exact top-k, collapse detection, revival, training) |

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
is never used. This model uses five mechanisms against that:

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

### Reading the retrieved slots

The model returns which slots it fetched, like RAG returns retrieved
documents:

```python
logits, aux = model.apply({"params": params}, tokens)
aux["slots"][layer]    # [B, T, heads, top_k] pool indices fetched per token
aux["weights"][layer]  # [B, T, heads, top_k] their mixing weights
```
