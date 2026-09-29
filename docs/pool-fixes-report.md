# Memory pool on real text: what was broken and what fixed it

Date: 2026-09-29. Hardware: 2x Tesla T4 (Kaggle), one TPU v5e-1 (Colab).
Raw numbers: `experiments/results/text_pool_usage/`, `experiments/results/tinystories/`.

## Summary

* **Why the Ultra-FineWeb model did not write English.** Data, not
  architecture. It saw 30M tokens of diverse web text. Trained on
  TinyStories (553M tokens of simple stories), the same 4.3M-parameter
  backbone writes fluent, coherent stories after 7.5 minutes on a TPU v5e.
* **Why the pool was ignored on text: routing noise.** With
  `routing_noise=1.0` only 0.4% of the slots read in training were the slots
  inference reads. Each vector was trained by unrelated contexts and stayed
  near its random start; the backbone learned to ignore it. Default is now 0.1.
* **Why the pool learned slowly: row-wise Adagrad.** Its step shrinks as a
  row is read, so often-read rows nearly stop learning. Lazy Adam (Adam on
  the rows read each step, still sparse) made the pool's gain about 4x
  larger. Default is now `pool_optimizer="adam"`.
* **Why the pool was 27x slower than dense on TPU: `lax.top_k`.** Now the
  argmax top-k is used on every accelerator (2.5x faster step on TPU).
* **Where it stands.** The pool clearly matters (shuffling its reads costs
  1.0-1.4 in loss, and greedy decoding falls into loops without it). A d384
  backbone with the pool roughly ties a dense d512 at equal wall-clock on a T4
  (1.643 vs 1.626, about 4% faster) with 9.6M instead of 14.8M parameters on the
  GPU. It does not yet beat widening the backbone at equal compute.

## 1. Why the text was not fluent

| Model | Data | Output |
|---|---|---|
| d256, 4 layers + pool (7.9M backbone) | Ultra-FineWeb, 30M tokens, 16k BPE | greedy loops ("is a muscle that is a muscle..."), sampled text grammatical but meaningless; held-out ppl 82.8-84.5 |
| d256, 4 layers, dense (4.3M) | TinyStories, 553M tokens, 4k BPE, 30,000 steps | fluent stories; validation ppl 4.45 (`tinystories/generations_dense.json`) |

TinyStories models of 1-33M parameters are fluent because the data is
narrow and simple (Eldan & Li, 2023). Open web text needs far more
parameters and tokens.

## 2. Routing noise (fixed: default 0.1)

Routing scores are cosines times a temperature of 10. With 512 sub-keys per
half the top scores sit close together, and Gumbel noise of scale 1.0
decided almost every pick.

Clean picks kept under training noise (Ultra-FineWeb model, `text_pool_usage/measurements.json`):

| routing_noise | 1.0 | 0.3 | 0.1 | 0.03 | 0.01 |
|---|---|---|---|---|---|
| training picks = inference picks | 0.4% | 23% | 70% | 90% | 97% |

Ultra-FineWeb screens, 3,000 steps (loss / loss with pool reads shuffled):

| Arm | loss | shuffled | gap |
|---|---|---|---|
| ref (noise 1.0) | 5.017 | 5.018 | 0.001 |
| pool values start at zero | 5.017 | 5.017 | 0.000 |
| 10x pool learning rate | 5.018 | 5.018 | 0.000 |
| no weight decay on the read path | 5.018 | 5.018 | 0.000 |
| all three | 5.017 | 5.017 | 0.000 |
| noise 0.1 | 5.010 | 5.067 | 0.057 |
| noise 0.03 | 5.008 | 5.068 | 0.060 |
| noise 0 | 5.009 | 5.070 | 0.061 |

Full 6,000-step Ultra-FineWeb run (FFN kept, top-4, query-scaled routing):
noise 0.1 held-out loss 4.416 (ppl 82.8) vs noise 1.0 4.437 (84.5) and dense
4.429 (83.8). Stored vectors that moved >10% from init: 72% vs 0.2%.

Fact task, 6,000 steps: 100% at noise 0.1, 99.99% at 1.0 (its fewer sub-keys
and higher temperature kept 83% of picks intact even at 1.0).

`pick_agreement` in the training log now shows this directly.

## 3. Pool optimizer (fixed: default lazy Adam)

TinyStories, 4,000 steps, d256 backbone, T4:

| Pool optimizer | loss | vs dense d256 (1.857) | shuffled-reads loss |
|---|---|---|---|
| row-wise Adagrad (old default) | 1.835 | -0.022 | 2.083 |
| dense Adam (all rows every step) | 1.773 | -0.084 | 2.852 |
| **lazy Adam (rows read only)** | **1.776** | **-0.081** | 2.819 |

Lazy Adam matches dense Adam and keeps sparse updates, which the host/SSD
pool needs. Its state is 2 moments per value (2x the table) against ~1/256 of
the table for row-wise Adagrad, which stays available for pools too large to
hold 3x. Fact task, 2 seeds: 100% / 100% with lazy Adam, 99.99% / 100% with
Adagrad; shuffled-read accuracy 0.7% vs 1.2-1.3% (lazy Adam relies on the pool
more).

## 4. TPU speed (fixed: argmax top-k on TPU)

One pool layer on TPU v5e, 16,384 tokens, 4 heads, 512 sub-keys, top-4:

| | ms |
|---|---|
| `lax.top_k`, N-D / 2-D | 255.7 / 37.2 |
| k rounds of argmax | 11.5 |
| pool forward, before / after | 74.7 / 41.7 |
| pool forward+backward, before / after | 93.7 / 62.5 |

Training step: 0.33 s before, 0.19 s after; a dense d256 step is 0.012 s.
The remaining cost is mostly the sub-key top-k (run twice per layer: noisy
picks and clean picks for the balance loss) and the scatters. New option
`balance_on_clean_picks=False` skips the second top-k (T4 pool layer
fwd+bwd 47.7 -> 41.8 ms; `pick_agreement` is then not measured).

## 5. Compute-fair comparison (TinyStories, 4,000 steps, one run per T4)

| Model | GPU params | pool | time | loss |
|---|---|---|---|---|
| dense d256 | 4.3M | - | 866 s | 1.857 |
| pool d256, lazy Adam, top-8 | 4.8M | 67M | 1,536 s | 1.754 |
| dense d384 | 8.8M | - | 1,508 s | 1.701 |
| pool d256, 4 memory layers | ~5.3M (est.) | 67M | 2,065 s | 1.707 |
| **pool d384, lazy Adam, top-8** | 9.6M | 67M | 2,185 s | **1.643** |
| dense d512 | 14.8M | - | ~2,280 s (measured speed; run not finished on T4) | 1.626 (TPU) |

* At equal time, pool d256 (1.754) loses to dense d384 (1.701).
* Pool d384 is about even with dense d512: about 4% faster, 0.017 higher loss,
  a third fewer parameters on the GPU; its 67M pool could live on SSD.
* Reading the pool in all 4 layers matches d384 in loss but is 37% slower.

## 6. What the pool does to the stories

TPU v5e, pool model (d256, FFN kept, top-4, Adagrad as then default)
stopped at step 16,000 of a 30,000-step schedule: validation loss 1.553 vs
dense 1.577 at the same step; shuffled reads 1.913; accuracy 60.4%, 54.3%
without the pool (`tinystories/ablation_pool.json`).

* Pool read normally: coherent stories.
* Reads shuffled or pool removed: greedy decoding falls into loops ("He is a
  good dog. He is a good dog...", "She mixed them in the bowl. She mixed
  them in the bowl...", "He said his dog, his dog, hiss"); sampled text
  derails more ("She put the sugar in the hat").

## 7. Other changes

* `acc_shuffled_pool` / `ce_shuffled_pool` eval metrics; `Generator(...,
  shuffle_pool=..., pool_off=...)` for text ablations.
* `value_init_scale`, `decay_pool_path` options (tested; no effect on text).
* `experiments/prepare_tinystories.py`.
* KV-cache decoding test pins f32 matmuls (TPU bf16 passes made cached and
  full paths differ by up to 0.03 on logits).
* Experiments whose saved results used the old defaults pin them
  (`--routing_noise 1.0`, `--pool_optimizer rowwise_adagrad`).

## 8. Limits and what is unverified

* Single seed for every text comparison; differences of ~0.01-0.02 are
  within noise.
* The full test suite passed (42/42, T4) before the last two changes (lazy
  Adam default, `balance_on_clean_picks`). The run with them was cut off when
  the GPU instance closed; only the targeted tests ran.
* Dense d512 on T4 did not finish (speed measured, loss from the TPU run).
  The TPU runs for "pool replaces the FFN" and a TPU cross-check of dense
  Adam were lost when the Colab session ended.
* TinyStories holds little factual knowledge, so it undersells a component
  built to store knowledge.

## 9. Next steps

1. Re-run the test suite with the new defaults.
2. A Pallas kernel fusing sub-key scoring and top-k (and the noisy and clean
   picks) so the score tensor never goes to HBM; the sub-key top-k is about
   half of a pool layer's forward time on TPU.
3. Judge the pool on a knowledge task (`experiments/knowledge_study.py`)
   against compute-matched dense models, with 2-3 seeds.
4. Longer TinyStories runs of pool d384 vs dense d512 to see whether the tie
   holds.

## Reproduce

```bash
python -m experiments.prepare_tinystories --out ts/tok
# backbone flags used everywhere:
B="--task tokens --train_tokens ts/tok/train.npy --eval_tokens ts/tok/val.npy --vocab_size 4096
   --steps 4000 --d_model 256 --n_layers 4 --n_heads 8 --ffn_mult 4 --max_len 256
   --batch_size 64 --lr 1e-3 --warmup_steps 500 --eval_every 1000"
python -m memory_pool_model.train $B --use_memory false                       # dense
python -m memory_pool_model.train $B --memory_layers 1,3 --n_sub_keys 512 \
  --d_key 128 --d_value 256 --memory_ffn true --router_query_scale true --top_k 8  # pool (new defaults)
```
