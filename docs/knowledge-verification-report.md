# Does the pool store the knowledge, and does the backbone use it?

Date: 2026-09-29. Hardware: TPU v5e-1 (Colab), JAX 0.7.2, libtpu 0.0.23.
Raw results: `experiments/results/knowledge/` (`knowledge_verify_<arm>.json`,
`knowledge_write_main.json`, training logs). Code:
`experiments/knowledge_study.py` (arms), `experiments/knowledge_verify.py`
(tests), `experiments/facts_in_text.py` (data).

## Summary

**Update (2026-09-30, TPU v5e-8).** With Q/A mixed training the pool model
answers questions about people it only read about in bios: 99.4-99.5% at
20,000 people (d512 x 6, two seeds; dense with the same backbone 93.9%),
98.9% for a 368M-parameter model (d768 x 12 + 1M-vector pool) trained on 3B
tokens and 50,000 people on all 8 chips, and 0-0.2% with the pool reads
shuffled. Checked on generated text, not only probe scores. Found and fixed
on the way: multi-chip TPU training crashed (Pallas kernel under data
parallelism), and bfloat16 attention softmax made long runs blow up late
(now float32). Details in the last three sections.

**Yes: the pool stores the knowledge, and the backbone reads it.** Every
test that nothing in training optimises for agrees:

* **Shuffled or removed pool -> the facts are gone.** In all 15 verified
  pool models (both scales), recall with the pool's reads shuffled (routing
  unchanged) or with the pool removed is 0.0-0.4%, from 5-95% with the
  pool; greedy answers become non-words.
* **Each fact lives in its own few vectors.** Zeroing only the vectors one
  fact reads (0.01-0.07% of the pool, top-4 per head) leaves 0-17% of it,
  while as many random vectors leave it intact (>= 97.8%), and most other
  facts survive.
* **A frozen backbone reads new knowledge from the pool.** Updating only
  the pool's vectors (backbone, router, keys bit-identical) teaches the
  model 1,000 new people (3% -> 99.6% recall for the d512 model) with
  held-out perplexity 30.8 -> 31.4; a full fine-tune forgets 88% of the old
  facts and loses 26% perplexity.
* **The backbone does not ignore the pool.** Shuffling the reads costs 0.6-
  1.8 nats of held-out loss on web text; even plain language suffers
  without it (the backbone is co-dependent on the pool, not a standalone
  model with an add-on).
* **No collapse.** 90-99.7% of the vectors are read on held-out text, every
  sub-key is used, and evenness keeps rising through 120k steps. No run
  diverged.
* **At scale the pool carries knowledge the backbone cannot.** 80,000
  facts: a d256 backbone (8.5M parameters on the chip) with a 1M-vector
  pool recalls 80.8%, more than a dense model 3.2x its size (69-74%); a
  d512 backbone recalls 94.5% with the pool vs 69-74% without, matching a
  dense model with 2.4x its parameters (93.8%), with better held-out loss
  than its dense twin (3.21 vs 3.31).

**What is not solved:**

* **Cost.** Measured alone on one v5e chip, a d512 x 6 step takes 8.3 ms
  dense and 78 ms with the pool at 16 reads per head (53 ms at 8, which
  keeps the recall), after a 4.5x speed-up of the pool step in this work.
  The dense model with the same backbone reaches similar recall with 3x the
  steps, so at equal wall-clock dense still wins.
* **Generalisation to new wordings.** No model recalls facts through
  wordings it never saw in training (0.4-11%, pool no better than dense),
  also with 10 training wordings per relation.
* **Pool size must match the number of facts.** With 262k vectors a d256
  pool model recalls only 6-15% of 80,000 facts (1M vectors: 80.8%), and
  run-to-run variance at 2,000 people is large (29-63%).

So the architecture works as designed (knowledge in the pool, reasoning in
the backbone, editable by writing to the pool), and is stable to train; it
is not yet cheaper than a wider dense model at equal compute.


## Results at 6,000 steps (2,000 people, 4 wordings)

Recall of trained facts with the training wordings (teacher-forced exact
answer). "Shuffled": routing unchanged, every read returns another slot's
vector. "Never seen": people not in training (chance level for a model that
guesses common values). Runs marked (C) are from the Colab v5e-1 (JAX 0.7),
the rest from the Kaggle v5e-8 (JAX 0.11); same data and code.

| model | params on accelerator | recall | shuffled | removed | new wording | never seen | held-out loss | loss, pool shuffled |
|---|---|---|---|---|---|---|---|---|
| dense (2 seeds) | 7.4M | 8.6-10.9% | - | - | 1.1-2.3% | 3.2-3.4% | 4.53-4.55 | - |
| dense_2x | 13.5M | 31.3% | - | - | 3.4% | 2.8% | 4.30 | - |
| dense_4x (C) | 27.4M | 76.4% | - | - | 11.2% | 2.8% | 4.13 | - |
| **pool** (3 runs) | 8.2M + 67M pool | **29.1-62.9%** | 0.0-0.1% | 0.0% | 1.4-1.9% | 2.4-3.3% | 4.43-4.46 | 5.62-5.64 |
| pool, no FFN in memory layers (2 seeds) | 7.2M + 67M | 38.4-56.8% | 0.0% | 0.0% | 2.1-2.7% | 2.5-3.3% | 4.48-4.49 | 5.73-5.75 |
| pool + no-pool penalty | 8.2M + 67M | 53.1% | 0.0% | 0.0% | 2.6% | 2.8% | 4.45 | 21.95 |
| pool, 4 reads per head | 8.2M + 67M | 13.3% | 0.0% | 0.0% | 0.6% | 4.0% | 4.47 | 5.07 |

Greedy generation, "<name> was born in": pool 15-32% correct, 0% with the
pool shuffled or removed (it then writes non-words such as "Kasasas"); dense
5-15%, dense_4x 70%.

Targeted deletion (pool, v5e-8 run; 200 facts that were recalled with all 4
wordings). The vectors a fact reads are 0.012% of the pool:

| vectors zeroed per head per read | vectors zeroed | fact still recalled | if another person's vectors (same relation) are zeroed | if as many random vectors are zeroed | 24 other facts: before -> after |
|---|---|---|---|---|---|
| top 1 | 31 | 29.2% | 67.6% | 100.0% | 54.0% -> 49.5% |
| top 4 | 119 | 3.4% | 29.9% | 99.6% | 53.2% -> 42.2% |
| all 16 | 452 | 0.8% | 8.5% | 99.6% | 54.1% -> 39.4% |

The same pattern holds in every pool run (all 7 pool runs: 1.6-4.2% left after
zeroing the top-4 vectors, >= 99% with random vectors). Another person's
vectors for the same relation overlap a fact's own by 17-33% (vectors that
encode "this is a birthplace" rather than one person), which is why zeroing
them also costs something.

Pool usage under inference routing (pool, web held-out text, both layers):
97.5% of vectors read, 86.5% get at least 10% of a fair share, evenness
0.47, heaviest vector 71x a fair share, every sub-key used; routing
temperature rose from 10 to 15 and the top vector of each read carries 0.18-
0.20 of the weight (0.06 = uniform). Answers to fact prompts concentrate on
7-20% of the vectors.

Write test (pool, v5e-8 run; 1,000 new people, 1,500 steps on their bios
mixed with fresh web text):

| model / update | old people (A) | new people (B) | B, pool shuffled | B, new wording | held-out ppl |
|---|---|---|---|---|---|
| pool before | 66.8% | 3.0% | 0.0% | 0.9% | 97.3 |
| pool: **pool vectors only** (backbone, router, keys bit-identical) | 20.4% | **91.8%** | 0.0% | 2.5% | 97.5 |
| pool: pool vectors only + replay of A's bios | **78.2%** | 85.6% | 0.0% | 1.9% | 98.0 |
| pool: full fine-tune | 6.1% | 98.3% | 0.1% | 2.5% | 101.9 |
| dense before / full fine-tune | 10.7% -> 4.5% | 3.5% -> 74.0% | - | 0.9% -> 3.5% | 108.5 -> 107.2 |

(A here is a 125-person sample, which the pool model recalled better than its
2,000-person average.)

## Results at scale: 20,000 people, 120,000 steps (TPU v5e-8, one run per chip)

Data: 380M tokens of Ultra-FineWeb + 60M tokens of bios (14%): 20,000
people, 80,000 facts, ~46 bios per person written with 10 wordings per
relation (the 2 test wordings stay held out). 120,000 steps x 8,192 tokens
(1B tokens, 2.2 passes). Probes use a fixed sample of 2,000 people.

| model | params on accelerator | recall | shuffled / removed | new wording | never seen | greedy "born in" | held-out loss | train time (1 v5e chip) |
|---|---|---|---|---|---|---|---|---|
| dense | 7.4M | 3.4% | - | 0.9% | 2.9% | 0% | 3.679 | 16 min |
| dense_2x | 13.5M | 5.9% | - | 1.5% | 2.9% | 0% | 3.495 | 19 min |
| pool (2 seeds) | 8.2M + 67M pool | 6.5-15.4% | 0.0% / 0.0% | 0.5-0.9% | 3.0-3.9% | 2.5-7.5% | 3.433-3.443 | 162 min |
| pool, no FFN in memory layers | 7.2M + 67M | 5.4% | 0.0% / 0.0% | 0.1% | 3.3% | 2.5% | 3.489 | 176 min |
| pool, 4 reads per head | 8.2M + 67M | 4.8% | 0.0% / 0.0% | 0.1% | 3.4% | 2.5% | 3.477 | 81 min |
| **pool_1m** (d256 + 1M-vector pool) | 8.5M + 269M | **80.8%** | 0.0% / 0.0% | 0.7% | 2.8% | 72.5% | 3.386 | 228 min |
| dense_4x (d512, 6 layers; 2 seeds) | 27.4M | 69.3-73.6% | - | 0.7-1.2% | 2.8-3.4% | 47.5-52.5% | 3.308-3.313 | 32 min |
| **pool_4x** (same backbone + pool) | 28.7M + 67M | **94.5%** | 0.2% / 0.4% | 2.1% | 3.1% | **92.5%** | **3.212** | 177 min |
| dense_8x (d768, 8 layers) | 69.5M | 93.8% | - | 0.4% | 2.9% | 90.0% | 3.141 | 72 min |

Robustness and cost checks (same data and schedule unless noted):

| model | params on accelerator | recall | shuffled / removed | greedy "born in" | held-out loss | train time |
|---|---|---|---|---|---|---|
| pool_4x, second seed | 28.7M + 67M | 95.0% | 0.0% / 0.1% | 85.0% | 3.208 | 193 min |
| pool_4x, 8 reads per head (half the pool traffic) | 28.7M + 67M | 93.9% | 0.1% / 0.4% | 95.0% | 3.219 | 128 min |
| pool_4x, no FFN in the memory layers | 24.5M + 67M | **95.9%** | 0.0% / 0.0% | 97.5% | 3.252 | 190 min |
| dense_4x, 360k steps (3x the steps; ~pool_4x's wall-clock) | 27.4M | 92.4% | - | 95.0% | 3.209 | 95 min |
| dense_8x, second seed | 69.5M | 92.4% | - | 90.0% | 3.146 | 72 min |
| dense_8x, 80k steps | 69.5M | 85.2% | - | 85.0% | 3.201 | 48 min |

| pool_4x, third seed, 80k steps | 28.7M + 67M | 85.1% | 0.0% / 0.1% | 77.5% | 3.260 | 129 min |
| **pool_8x** (d768 x 8 + pool), 80k steps | 71.3M + 67M | **94.1%** | 0.5% / 0.6% | 90.0% | **3.156** | 155 min |

At 80k steps the pool adds knowledge on top of the largest backbone too:
94.1% vs 85.2% recall and 3.156 vs 3.201 held-out loss for the same d768 x 8
model with and without it, and the third pool_4x seed (85.1%) matches the
2.4x larger dense model at the same step (85.2%).

The pool result is robust across seeds and variants (93.9-95.9%), and
halving the reads per head costs almost nothing. It is a gain in *sample*
efficiency: the dense model with the same backbone needs about 3x the steps
(360k) to reach similar recall and loss. Converting that into a wall-clock
win depends on the pool's step cost (next section).

Recall over training (monitor snapshots from the resumable checkpoints,
300 people): the dense model learns the facts late; the pool learns them
much earlier on the same backbone.

| step | dense_4x (seed 1) | pool_4x |
|---|---|---|
| 20k | 3.2% | - |
| 40k | 4.1% | 8.6% |
| 60k | 8.1% | 60.7% |
| 80k | 28.7% | - |
| 100k | 57.4% | - |
| 120k | 69.6% | 94.5% (2,000 people) |

**Stability.** None of the 13 long runs diverged: no NaN, the held-out loss
fell at every one of the 24 evaluations of every run, and after warmup the
gradient norm stayed at or below 0.55 (clip 1.0; median 0.24-0.37 for the
pool runs). Pool use kept spreading: for pool_4x the share of vectors read
on held-out text is 99.7% at 120k steps with 96.6% getting a fair share and
evenness rising from 0.51 (5k) to 0.66; every sub-key is used; the routing
temperature rose from 10 to 16 and the top vector carries ~0.2-0.3 of each
read (0.06 = uniform).

**Targeted deletion at scale** (pool_4x, 200 facts): zeroing the vectors a
fact reads (0.016-0.25% of the pool) leaves 43% / 17% / 9% (top 1 / 4 / 16
per head) against 100% / 99.8% / 99.6% for as many random vectors; other
facts drop from 88-90% to 82% / 75% / 69%. At this scale facts share more
vectors (34% overlap with another person's vectors for the same relation)
and are spread over more of them than at 2,000 people.

**Write test at scale** (pool, d256; 1,000 new people, 3,000 steps):

| update | old people | new people | new people, pool shuffled | held-out ppl |
|---|---|---|---|---|
| before | 15.0% | 3.2% | 0.0% | 38.0 |
| pool vectors only (rest bit-identical) | 5.8% | **97.7%** | 0.0% | **39.6** |
| pool vectors only + replay | 19.2% | 96.4% | 0.0% | 39.6 |
| full fine-tune | 3.2% | 99.8% | 0.1% | 47.4 |

Same test on pool_4x (d512 backbone; 1,000 new people, 3,000 steps):

| update | old people | new people | new people, pool shuffled | held-out ppl |
|---|---|---|---|---|
| before | 95.9% | 3.1% | 0.1% | 30.8 |
| pool vectors only (rest bit-identical) | **74.1%** | **99.6%** | 0.1% | **31.4** |
| pool vectors only + replay | **92.9%** | 99.1% | 0.1% | 31.4 |
| full fine-tune | 11.4% | 100.0% | 0.3% | 38.7 |

A frozen backbone reads knowledge written into the pool after training, and
the model keeps most of what it knew; a full fine-tune learns the new people
as well but forgets 88% of the old facts and loses 26% perplexity on web
text.





## Method

**Data (facts in text).** 2,000 invented people (names built from random
syllables, so no prior knowledge helps), 4 facts each (birthplace, job, field
of study, favourite food; 40 possible values per relation) = 8,000 facts.
Each person gets ~39 short bios written with 4 training wordings per
relation, in random order, mixed into 30M tokens of Ultra-FineWeb (14% of
the 35M training tokens are bios). 1,000 more people (set B) are never in
training: they are the chance-level control and the material for the
write test. Recall is teacher-forced exact match of the whole answer after
a prompt such as "Rasnarfen Rovel was born in".

**Models.** Same backbone everywhere: d_model 256, 4 layers, 8 heads, FFN
x4, 16k BPE, 256 tokens, batch 32, 6,000 steps (49M tokens, 1.4 passes),
lr 1e-3. Pool: 262,144 vectors x 256 (67M parameters) read in layers 1 and
3, 4 heads x top-16, current defaults (routing noise 0.1, lazy Adam, no
no-pool penalty).

**Tests** (nothing in training optimises for any of them):

1. *Shuffled reads*: routing unchanged, every read returns another slot's
   vector. If the answers come from what the pool stores, recall collapses.
2. *Pool removed*: the memory read is skipped.
3. *Targeted deletion*: for each of 200 facts recalled with all 4 wordings,
   zero only the vectors read while its answer is predicted (top-1, top-4
   or all 16 per head, both layers, all 4 wordings). Controls: the same
   number of random vectors; the vectors of another person for the same
   relation. Collateral: 24 random other facts before and after.
4. *Usage under inference routing* (no noise): share of vectors read,
   share with at least 10% of a fair share, evenness exp(H)/N, heaviest
   vector vs a fair share, sub-keys used, top-1 mixing weight, temperature.
5. *Write test*: set B written into a trained model by updating only the
   pool's value vectors (backbone, router, keys, gates bit-identical). If
   the frozen backbone then answers about B, it knows how to read new
   knowledge from the pool.

## Speed: pool training step 4.5x faster on TPU

Pool model above, one step, TPU v5e-1:

| version | ms / step |
|---|---|
| before (argmax top-k, 3 usage scatters per layer, sort-based row gradients, `take_along_axis`) | 329 |
| `lax.top_k`; one usage scatter per layer; dense scatter-add row gradients + masked lazy Adam | 252 |
| routing-score gathers as fused one-hot reductions (forward and backward) | 117 |
| Pallas top-k kernel | 92 |
| no layout barrier on the fetched rows on TPU; no usage-count scatter in TPU training (JAX 0.11, v5e-8 chip) | 72 |
| routing tensors laid out as [tokens, 2 x heads, sub-keys] | **70** |

Profiled with `jax.profiler` (per-op device time). The matmuls were never
the cost: each `take_along_axis` over the [8192, 4, 2, 512] score tensor
took 16.5 ms on TPU (8 such gathers per step) and its scatter gradient
10.8 ms; XLA's `top_k` is a full sort (7.5 ms per call, 4 per step). The
Pallas kernel (`memory_pool_model/topk_pallas.py`) keeps 256 rows in VMEM
and runs k rounds of max / lowest index / mask (1.2 ms, bit-identical to
`lax.top_k` including ties; 11x faster at width 1024). Left: the pool rows'
gradient scatter-add and row gathers.

Later rounds (Kaggle v5e-8, JAX 0.11): on TPU the layout barrier on the
fetched rows is dropped (it forced a relayout copy of all 537 MB per layer),
the usage-count scatter is skipped in training steps (exact sub-key counts by
a fused one-hot sum, touched rows from the gradients), and routing tensors
are laid out as [tokens, 2 x heads, sub-keys]. d512 x 6 pool model:

| | 16 reads per head | 8 reads per head | dense, same backbone |
|---|---|---|---|
| ms per step | 96 -> **78** | 56 -> **53** | 8.3 |

Where the 53 ms go (8 reads per head): pool-row gradient scatter-add 9.1,
row gathers 6.4, masked lazy Adam 3.9 (one DMA per 1 KB row: a Pallas
gather-and-mix kernel that DMAs rows straight into VMEM was exact but slower,
8.9 vs 7.2 ms per layer), router ~8 (memory-bound passes over the
[tokens, 8, 512] score tensor: cosine einsum 3.0, top-k 2.4, one-hot takes
2.6, temperature / noise / softmax statistics), backbone ~15. A fused Pallas router kernel (`memory_pool_model/router_pallas.py`) does
all of stage 1 in VMEM: scores on the MXU, Gumbel noise generated in the
kernel from a counter hash, noisy top-k, the clean score at each pick (the
noisy value minus the recomputed noise, no reduction), the clean k-th value,
membership counts and the softmax sums for the balance loss, with a
hand-written backward that recomputes the scores. It is exact (same picks as
XLA, clean scores within float rounding, gradients within 1e-6). Forward +
backward per layer went from 12.9 to 6.7 ms (top-16) and from 6.3 to 3.7 ms
(top-8) with the in-kernel noise, but a full training step is still not
faster than with the XLA router plus the Pallas top-k (54.1 vs 53.3 ms at
top-8, 84.9 vs 78.3 ms at top-16), so it is opt-in (`MEMPOOL_FUSED_ROUTER=1`).
The forward is bound by cross-lane reductions: each selection round needs a
max and then the lowest index at the max. Fusing the two (e.g. packing the
index into the low mantissa bits, at the cost of exactness among near-ties)
is the next thing to try.

## Bugs and problems found and fixed

| # | problem | effect | fix |
|---|---|---|---|
| 1 | `knowledge_study.py` arms pinned the pre-fix settings (routing noise 1.0, row-wise Adagrad, no-pool penalty on every token) | the study would have re-run the regime in which the pool is ignored on text (0.4% of training reads matched inference reads) | arms use the current defaults; the penalty is its own arm |
| 2 | `knowledge_study.finetune` gave the donating train step the caller's buffers (`jnp.asarray` does not copy) | the trained model was deleted after the first fine-tune: the write test and the `update` command crashed with "Array has been deleted" | copy the parameters first |
| 3 | argmax-rounds top-k on TPU | with JAX 0.7 3x slower than `lax.top_k` (and both far slower than needed) | Pallas top-k on TPU, `lax.top_k` fallback, argmax kept for GPU |
| 4 | `take_along_axis` on the routing scores | 16.5 ms per gather and 10.8 ms per scatter gradient on TPU, 8 + 2 per step | fused one-hot gather with a custom VJP (TPU only) |
| 5 | three 1M-update scatters per layer for usage counts; sort + segment-sum for row gradients | ~50 ms per step on TPU | one per-head scatter per layer, threshold counts for top-k membership; dense scatter-add + masked lazy Adam (`pool_row_grads`) |
| 6 | data parallel merged row gradients with a sort and an all-gather of every device's rows | ~2 GB gathered per step at 8 devices | dense per-device scatter + one all-reduce (both modes tested against one device) |
| 7 | `prepare_ultrafineweb.py` did not exit after writing everything | the HF streaming reader leaves non-daemon threads; a pipeline waiting on it hung (seen on Colab) | `os._exit(0)` after `main()` |
| 8 | `facts_in_text.build` took set B's fresh text from a fixed offset (50M) of the full file | with a larger training set the "fresh" text would have been training text | `--fresh_offset` with an overlap check |
| 9 | `text_study.launch` could only pin CUDA GPUs | no way to run one arm per TPU chip | `--gpus tpu0,...,tpu7` pins each process to one chip (own port) |
| 10 | Colab's image ships libtpu 0.0.21.1 with jax 0.7.2 (needs 0.0.23) | Mosaic (Pallas) kernels fail to load | documented; the router probes the kernel once and falls back to `lax.top_k` with a warning |
| 11 | the Pallas top-k was called directly under a data-parallel mesh | multi-chip TPU training crashed ("Mosaic kernels cannot be automatically partitioned"); only GPU data parallel had been tested | the trainer registers the mesh and the kernel runs per device under `shard_map`; 8-chip test |
| 12 | attention softmax computed in bfloat16 (Flax uses the module dtype) | late in training the first layers grew sharp: one of three 120k-step bfloat16 pool runs blew up (gradient norm 0.3 -> 2.8e4, Q/A 97% -> 39%), and so did the 368M run at 90k-95k (0.25 -> 2.4) with Adam beta2 0.95 | `ModelConfig.attn_fp32_softmax` (default on); the 368M run resumed from 90k with it stays at 0.25 |

Not changed, needs a decision: `.github/workflows/run.yml` runs on every push,
downloads an archive from an anonymous file host (`free.keep.sh`), builds it
and uploads the result there. It is unrelated to this project and runs
unverified code in CI; unless it is intentional it should be deleted.

Operational lessons from the runs: Colab reclaimed the TPU VM twice (all
unsaved results lost the first time; after that results were copied back
after every arm), and on the Kaggle v5e-8 writing GB-sized checkpoints to the
container disk stalled it (95% I/O pressure, processes stuck in
uninterruptible writes holding their TPU chips). Checkpoints now go to
`/dev/shm` there.


## Follow-up: recall through wordings never seen in training

Quick test in the last minutes of the v5e-8 session (6,000 steps, 2,000
people, one run per chip; recall on 300 people; results in
`experiments/results/v5e8/gen/`). Two fixes: *diverse* data (bios from 25-37
generated wordings per relation, possessive and Q/A forms included, the test
wordings never generated) and *late routing* (read the pool only in the last
layer, where the name has already been gathered).

| model | data | seen wordings | **new wordings** | never-seen people (chance) |
|---|---|---|---|---|
| dense d256 | diverse | 5.9% | 1.9% | 2.1% |
| pool d256 | diverse | 7.4% | 1.5% | 2.3% |
| pool d256, late routing | 4 wordings / 10 / diverse | 15.6% / 6.1% / 4.0% | 0.8% / 2.0% / 1.0% | 2.2-3.0% |
| dense d512 x 6 | diverse | 50.2% | 8.6% | 2.5% |
| pool d512 x 6, late routing | diverse | 26.5% | 5.4% | 2.5% |
| **pool d512 x 6** | diverse | **55.8%** | **14.8%** (0.0% with the pool shuffled) | 4.4% |

* With the d512 backbone and diverse data, the pool model answers **15% of
  facts through unseen wordings, 3.4x chance** (dense of the same size:
  8.6%), and this recall also disappears when the pool reads are shuffled,
  so it comes from the pool. That is the first above-chance transfer in
  this study.
* Late routing does not help; the d256 backbone does not transfer at all.
* Not a fix yet: single runs, 6,000 steps, and there is no d512 pool run on
  the 4-wording data at this scale to compare with (dense d512 there got
  11.2%, so part of the effect may be model size). Next: the same arms with
  2-3 seeds and long training, and Q/A-format training for half of the
  people with Q/A tested on the other half (Allen-Zhu & Li's mixed
  training).

### Q/A mixed training: the fix (2x T4, 6,000 steps)

Same 2,000 people and bios (10 wordings), plus question-answer documents
("Q: Where was X born? A: Lima.") for the **first 1,000 people only**. The
other 1,000 appear only in bios. The test asks them the same questions:
to answer, the model has to extract a fact it only read in a bio
(Allen-Zhu & Li, mixed training). Results in `experiments/results/qa_mix_2xT4/`.

| model (d512 x 6) | Q/A, people trained with Q/A | **Q/A, people seen only in bios** | Q/A, never-seen people (chance) | bio wordings | unseen statement wordings |
|---|---|---|---|---|---|
| dense | 98.8% | 84.8% | 3.1% | 81.1% | 12.0% |
| **pool** | 99.5% | **92.5%** | 2.8% | 89.3% | 16.5% |
| pool, reads shuffled / pool removed | 1.0% / 1.3% | **0.3% / 0.8%** | - | 0.5% / 1.3% | 0.3% / 0.3% |

* **Knowledge extraction works.** Trained on questions for half the people,
  the pool model answers them for the other half 92.5% of the time (dense of
  the same size: 84.8%). With the pool's reads shuffled or the pool removed
  this falls to 0.3-0.8%: the facts it extracts come from the pool, and the
  backbone has learnt a general "read the fact for this person" skill.
* The pool model is also better on the bios themselves (89.3% vs 81.1%) and
  on held-out web text (loss 4.33 vs 4.37).
* Statement wordings never trained in any form ("X's hometown is ...") stay
  low (16.5%). The model generalises to the *format it was taught to answer
  in* (questions), for people it never saw in that format; to cover other
  formats they need to be in the Q/A-style training for some people too.
* **Second seed confirms it:** pool 90.9% on bio-only people (seed 0: 92.5%),
  1.0% with the pool shuffled and 3.3% with it removed; dense 84.0% (seed 0:
  84.8%). Chance 2.4-3.4%. Still 6,000 steps and 2,000 people; the 80,000-fact
  scale is the next check.

### Mixed precision (GPU)

On a T4 the float32 step is bound by the backbone's matmuls (T4 has no fast
float32 path), not by the pool: d512 x 6 dense 425 ms, with the pool 517 ms.
New option `compute_dtype` runs the backbone's matmuls in float16 or
bfloat16; parameters, the residual stream, layer norms, routing and the pool
stay float32, and float16 trains with dynamic loss scaling (overflowing steps
are skipped). One T4:

| d512 x 6 | float32 | float16 |
|---|---|---|
| dense | 425 ms | 121 ms (3.5x) |
| pool, 8 reads per head | 517 ms | 226 ms (2.3x) |

Quality check, Q/A mixed training (6,000 steps, d512 x 6, one T4 each):

| | pool float32 (2 seeds) | pool float16 | dense float32 (2 seeds) | dense float16 |
|---|---|---|---|---|
| Q/A, people seen only in bios | 90.9-92.5% | 92.7% | 84.0-84.8% | 87.4% |
| same, pool shuffled / removed | 0.3-1.0% / 0.8-3.3% | 1.2% / 2.2% | - | - |
| held-out loss | 4.325-4.329 | 4.329 | 4.369-4.372 | 4.376 |
| training time | ~65 min | ~31 min | ~46 min | ~16 min |

float16 matches float32 within seed noise, with no skipped steps; the loss
scale grew from 2^15 to 2^17-2^18. (The float16 runs are on a separate
machine whose tokenizer came out slightly different, so the dense 87.4% is
not a like-for-like gain.)


### Inference shortcuts for a low-memory engine

The float16-trained pool model (trained reading 16 vectors per head) with the
pool stored in lower precision, or with fewer vectors read at inference; 300
people per group (`experiments/inference_robustness.py`,
`experiments/results/qa_mix_2xT4/inference_robustness_pool_4x_fp16.json`).

| pool storage | bytes per vector (256 dims) | bio recall | Q/A, bio-only people | held-out loss |
|---|---|---|---|---|
| float32 | 1024 | 91.9% | 92.7% | 4.3292 |
| bfloat16 | 512 | 91.8% | 92.8% | 4.3292 |
| float16 | 512 | 91.9% | 92.7% | 4.3293 |
| **int8** (per-row scale) | 256 + 4 | 91.8% | 92.8% | 4.3292 |
| int4 (per-row scale) | 128 + 4 | 91.6% | 91.9% | 4.3303 |

| vectors read per head at inference (trained with 16) | bio recall | Q/A, bio-only people | held-out loss |
|---|---|---|---|
| 16 | 91.9% | 92.7% | 4.329 |
| 8 | 89.5% | 87.7% | 4.370 |
| 4 | 77.6% | 71.4% | 4.464 |
| 2 | 50.6% | 44.4% | 4.669 |
| 1 | 21.6% | 17.3% | 5.036 |

* **int8 is lossless and int4 nearly so.** The 67M-parameter pool fits in
  67 MB at int8 (34 MB at int4) instead of 268 MB, 4-8x less to store and
  read from SSD per vector.
* **Reads must match training.** Reading fewer vectors than the model was
  trained with loses knowledge fast. The engine should read exactly what the
  model was trained for; to read fewer, train with fewer (8 per head in
  training reached the same recall as 16 in the earlier runs).

### Seed 2 and multi-GPU scaling (2x T4)

Q/A on bio-only people, float32, three seeds: pool 92.5 / 90.9 / 91.1%, dense
84.8 / 84.0 / 86.2%; with the pool reads shuffled 0.3-1.2%. On the same
machine and data, float32 seed 2 gives pool 91.1% / dense 86.2% and float16
gives 92.7% / 87.4%, so float16 training is confirmed.

Data parallel, d512 x 6, float16, 32 sequences of 256 tokens per GPU:

| model | 1 GPU | 2 GPUs | speed-up |
|---|---|---|---|
| dense | 59,972 tok/s | 107,566 tok/s | 1.79x |
| pool, sort-based row gradients | 32,715 tok/s | 44,237 tok/s | 1.35x |
| pool, dense row gradients (one all-reduce) | 34,212 tok/s | 42,383 tok/s | 1.24x |

The pool scales poorly because it is replicated: every step the devices
exchange the gradients of all rows read (or the whole table). Before a large
multi-device run the pool should be sharded by rows across devices (each
device owns a slice, reads are sent to the owner, only owned rows are
updated), which also lifts the one-device limit on pool size.


## 20,000 people with Q/A mixed training (TPU v5e-8, bfloat16, 3 seeds)

Date: 2026-09-30. Kaggle TPU v5e-8, JAX 0.11.2, one run per chip. Data: 380M
tokens of Ultra-FineWeb + 84M tokens of bios and Q/A documents: 20,000
people, 80,000 facts, bios in 10 wordings, Q/A documents ("Q: Where was X
born? A: Lima.") for the first 10,000 people only. The test asks the same
questions about the other 10,000, who appear only in bios. d512 x 6 backbone,
120,000 steps x 8,192 tokens, `compute_dtype=bfloat16`. Results, logs and
generations in `experiments/results/long_qa_v5e8/`.

Teacher-forced probes, 300 people per group, final checkpoints:

| model | Q/A, trained with Q/A | **Q/A, seen only in bios** | same, pool shuffled | bio recall | never seen (chance) | held-out loss (shuffled) | ms / step |
|---|---|---|---|---|---|---|---|
| dense_4x (same backbone, 3 seeds) | 98.3-99.1% | 91.2-93.8% | - | 85.6-87.1% | 3.3-3.6% | 3.544-3.548 | 13.5 |
| **pool_4x**, seeds 0 and 2 | 99.9-100% | **99.3%** | **0.0%** | 95.5-96.4% | 2.8-3.4% | 3.434-3.441 (4.46-4.49) | 50 |
| pool_4x, seed 1 (unstable, see below) | 48.0% | 38.7% | 0.0% | 40.1% | 3.1% | 3.787 | 50 |
| dense_8x (d768 x 8, 70M parameters, 3 seeds) | 99.8-99.9% | 98.8-99.3% | - | 95.3-96.1% | 2.5-3.3% | 3.359-3.363 | 30 |

Q/A on people seen only in bios during training:

| step | 20k | 30k | 40k | 50k | 60k | 80k | 120k |
|---|---|---|---|---|---|---|---|
| dense_4x | 4% | 4-5% | - | 14% | 26% | 74% | 91-94% |
| dense_8x | 4-5% | 6% | 28-34% | - | 76-81% | - | 99% |
| pool_4x | 4% | 5-6% | 26-30% | 64-71% | 82-87% | 96-97% | 99.3% |

**Generated text, not only probe numbers** (`experiments/inspect_outputs.py`:
greedy decoding, no teacher forcing, the answer cut at the first period and
compared with the truth; 200 people per group, 800 questions):

| model | Q/A, trained with Q/A | Q/A, seen only in bios | Q/A, never seen | bio, training wording |
|---|---|---|---|---|
| pool_4x (seed 0 / 2) | 100% / 100% | **99.4% / 99.5%** | 3.4% / 2.2% | 84.5% / 88.0% |
| same, pool reads shuffled | 0% | **0%** | 0% | 0-0.2% |
| dense_4x | 98.5% | 93.9% | 2.4% | 71.6% |
| dense_8x (seed 1) | 99.6% | 99.4% | 2.2% | 88.6% |

Examples from pool_4x (people seen only in bios):

```
Q: What is the job of Velosk Mirrofen? A:  ->  a veterinarian. Q: What food does V...   (truth: a veterinarian)
Q: Where was Korcletor Yarul born? A:      ->  Denver. Q: What food does Kor...          (truth: Denver)
same, pool reads shuffled:                 ->  a. The degree of the same way.            
Q: Where was Brais Ulkor born? A: (never seen)  ->  Muscat. Q: What is the job of ...     (truth: Salzburg)
```

* The answers are real: the model writes the value and stops at the period,
  then continues with the next question in the trained format. For people it
  never saw it gives a valid-looking value of the right kind (96-98% of
  answers), correct at chance (2-3%): it does not invent a format, and it
  does not know what it was not taught.
* With the pool reads shuffled the pool model answers nothing (0%) and
  writes broken text: every recalled fact comes from the pool.
* **Pool vs the same backbone without the pool:** 99.3% vs 91-94% on
  extraction, 96% vs 86% on bios, loss 3.44 vs 3.55, and the pool model gets
  there in fewer steps (60k steps: 82-87% vs 26%).
* **Pool vs a bigger dense model (dense_8x):** same extraction (99.3% vs
  98.8-99.3%) with 28.7M instead of 70M parameters on the accelerator
  (the 67M-value pool can live in host RAM or on SSD, and int8 storage is
  lossless: 99.2% at int8 and 99.0% at int4 vs 99.3% float32,
  `inference_robustness_pool_4x.json`), but dense_8x has the better held-out
  loss (3.36 vs 3.44) and a faster step (30 vs 50 ms). At equal compute the
  pool buys a smaller on-chip model, not better language modelling.
* Targeted deletion (200 facts): zeroing the 4-per-head vectors a fact reads
  (0.08% of the pool) leaves 6.8-8.9% of it; as many random vectors leave
  99.6-99.8%. Zeroing another person's vectors for the same relation also
  hurts (17-26% left): 34-37% of the vectors are shared per relation, so this
  control is not independent.
* Write test at this scale (update only the pool's vectors, backbone
  frozen): 1,000 new people learnt to 99.3%, old facts 86.6% (96.8% with
  replay), held-out perplexity 32.1.
* Statement wordings never trained in any form stay at 0-0.5%, as before:
  the model generalises to the format it was taught to answer in.

**Instability in one of three pool runs.** In seed 1 the gradient norm grew
from 0.25 (60k) to 0.8 (90k), 45 (100k) and 2.8e4 (117k); clipping kept each
update bounded but training degraded (held-out loss 3.51 -> 3.79, Q/A 97% ->
39%). Routing stayed healthy throughout (92% of the pool read, all sub-keys
used): this is an optimiser instability, not pool collapse. On the final
checkpoint 99% of the embedding gradient sits in 20 token rows and at
positions 0-3 of the windows (healthy seed: grad norm 0.37 vs 9.4 on the same
held-out batch). The dense parameters used Adam with beta2 = 0.999 (optax
default); long language-model runs normally use 0.95 for this reason, so
`TrainConfig.adam_b2` was added and the large run below uses 0.95. That
was not the cause: the large run blew up the same way with 0.95, and
resuming it with the attention softmax in float32 removed the growth (see
the next section). The attention softmax now runs in float32.

**Data parallel on TPU and the sharded pool.** Multi-chip training on TPU
failed: XLA cannot partition a Pallas call (`Mosaic kernels cannot be
automatically partitioned`), so the router's top-k now runs per device under
`shard_map` (tested on 8 chips, replicated and sharded pool). d512 x 6,
32 x 256 tokens per chip, bfloat16:

| model | 1 chip | 2 chips | 4 chips | 8 chips | speed-up at 8 |
|---|---|---|---|---|---|
| dense | 620k tok/s | 1.08M | 2.21M | 4.31M | 6.95x |
| pool, replicated | 165k tok/s | 237k | 508k | 960k | **5.83x** |
| pool, sharded by rows (capacity 1.5) | - | 159k | 361k | 597k | 3.6x |

On the TPU the replicated pool scales well (the table's gradient all-reduce
is cheap on the chip interconnect), and the row-sharded pool is slower: its
two all-to-alls per layer cost more than they save. Sharding stays on the
`pool-sharding` branch for pools that do not fit on one chip.

**Where the pool step's time goes** (bfloat16, one chip, 49.5 ms vs 13.2 ms
dense; `jax.profiler`): gradient scatter-add 8.6 ms, row gathers 6.4 ms (each
row is gathered once for forward and backward), masked Adam 3.0 ms, layout
copies 3.5 ms, router ~6 ms, top-k 3.2 ms. A microbenchmark shows the
gathers and scatters are bound by the number of rows touched (~14 ns per
row; 18 GB/s of the chip's 819 GB/s), not by bytes: a bfloat16 table saves
only 8% (3.69 -> 3.40 ms per gather). MFU: dense ~53%, pool ~15%.


## Larger model on 3B tokens (TPU v5e-8, all chips data parallel)

Date: 2026-09-30. d768 x 12 backbone (99M parameters) + a 1M-vector pool
(269M values, memory layers 4 and 8, 4 heads x top-8), bfloat16, trained
data parallel on all 8 chips (batch 256 x 256 tokens, replicated pool:
160 ms/step, 410k tokens/s; the row-sharded pool measured 189 ms). Adam
beta2 0.95, lr 6e-4 cosine to 6e-5, 100k steps = 6.6B tokens (2 passes).
Data: 3.0B tokens of Ultra-FineWeb (8 files) + 211M tokens of bios and Q/A
(6.6%): 50,000 people, 200,000 facts, Q/A documents for 25,000 of them.
Results in `experiments/results/big_v5e8/`.

Monitor (teacher-forced, 300 people per group) during training:

| step | Q/A, trained with Q/A | Q/A, seen only in bios | same, pool shuffled | bio recall | never seen | held-out loss (pool shuffled) |
|---|---|---|---|---|---|---|
| 20k | 3.3% | 3.6% | 0.2% | 3.1% | 4.1% | 3.21 (4.12) |
| 30k | 14.7% | 7.6% | 0.0% | 7.7% | 3.8% | 3.14 (4.10) |
| 40k | 70.8% | 38.2% | 0.2% | 45.5% | 3.7% | 3.08 (4.08) |
| 50k | 93.7% | 78.4% | 0.6% | 74.3% | 3.4% | 3.04 (4.07) |
| 65k | 97.9% | 94.1% | 0.1% | 85.8% | 3.4% | 2.99 (4.09) |
| 75k | 99.0% | 96.7% | 0.0% | 88.4% | 3.5% | 2.97 (4.07) |

Facts start to be recalled after about one pass over the bios (~30k steps
here: 50,000 people at 6.6% of the tokens), as in the smaller runs.

**The late instability again, and its cause.** From 89k steps the
gradient norm grew again (1k-step means 0.25, 0.33, 0.51, 1.05, 1.81, 2.44 at
89k-95k; held-out loss 2.751 at 90k -> 2.769 at 95k) although beta2 was
0.95, so beta2 was not the cause. Routing statistics stayed flat (temperature,
balance, spread, pick agreement); the gradient sat in the first layers
(layer-1 LayerNorm, embeddings, layer 0-1 attention) and at the first
positions, with the weights essentially unchanged (the learning rate was
~7e-5): the first layers had become sharp. The backbone's attention ran its
softmax in bfloat16 (Flax computes it in the module's dtype). Resumed from
the 90k checkpoint with the softmax in float32 and nothing else changed
(resume is exact: same data order and step RNGs):

| 1k-step mean gradient norm | 91k | 92k | 93k | 94k | 95k | held-out loss at 95k |
|---|---|---|---|---|---|---|
| bfloat16 softmax (original) | 0.33 | 0.51 | 1.05 | 1.81 | 2.44 | 2.769 |
| **float32 softmax** | 0.24 | 0.25 | 0.25 | 0.25 | 0.25 | **2.744** |

`ModelConfig.attn_fp32_softmax` (default on) now computes the attention
softmax in float32 under 16-bit compute; it costs nothing measurable (160.0
ms per step on 8 chips with and without). The earlier float32 runs (15 pool
models, 120k steps) never showed this, and the float16 runs were short.

The run was stopped at 95k and continued from the 90k checkpoint with the
float32 softmax to 100k steps (`pool_big_fix`; gradient norm 0.24-0.26 to
the end, held-out loss 2.741). Final model, 368M parameters (100M on the
accelerator without the pool values):

| test | pool_big_fix (100k) | 90k checkpoint |
|---|---|---|
| **Q/A, seen only in bios**, greedy generation (1,000 questions) | **98.9%** (pool shuffled 0.2%) | 97.7% (0.0%) |
| Q/A, trained with Q/A | 99.7% | 99.6% |
| Q/A, never seen (chance) | 2.0% | 3.0% |
| bio, training wording, greedy | 85.5% | 81.4% |
| bio recall, teacher-forced, 2,000 people x 10 wordings: normal / shuffled / removed | 92.5% / 0.1% / 0.9% | 90.5% / 0.1% / 0.8% |
| greedy "born in": normal / shuffled | 92.5% / 0% | 87.5% / 0% |
| held-out web loss: normal / shuffled / removed | 2.741 / 3.755 / 3.505 | 2.751 / 3.739 / 3.499 |
| targeted deletion, own 4-per-head vectors (0.02% of the pool) / as many random | 5.0% / 99.9% left | 6.8% / 99.0% |
| Q/A bio-only, pool stored float32 / int8 / int4 | 98.6% / 98.6% / 98.0% | 98.8% / 98.7% / 97.7% |
| Q/A bio-only reading 4 instead of 8 vectors per head | 90.2% | 86.8% |
| sub-keys used / pool read on held-out web text | 100% / 76% (of 1M, in 65k tokens) | 100% / 76% |

Generated text (people seen only in bios; greedy):

```
Q: What is the job of Miroskkor Vekzim? A:  ->  a librarian. Q: What did Mi...     (truth: a librarian)
Q: Where was Zimbrayar Korulfen born? A:     ->  Hanoi. Q: What did Zim...          (truth: Hanoi)
same, pool reads shuffled                    ->  a day. The job of Zimbray...
What is Stable Diffusion? Stable Diffusion is a free, open-source tool designed by Stability AI to create
high-quality images,  ->  videos, and other media. It is a powerful tool that allows users to create
high-quality videos, images, and other media.
```

* **The model trains well at this size on all 8 chips**, uses the pool
  (0.8-1.0 nats worse held-out loss without it) and answers questions about
  people it only read about in bios 98.9% of the time, with 0.2% when the
  pool reads are shuffled: the knowledge is in the pool and the backbone
  reads it.
* No collapse: every sub-key is used, the routing statistics were flat from
  20k steps to the end.
* **Write test** (1,000 new people, 3,000 steps at batch 32 on one chip;
  the 368M model's training batch of 256 does not fit one chip):

  | arm | old facts (before: 92.9%) | new people (before: 2.8%) | held-out ppl (before: 19.5) |
  |---|---|---|---|
  | **pool vectors only** (backbone, router, keys bit-identical) | 77.7% | **99.5%** | **19.9** |
  | pool vectors only, with replay of old bios | 90.4% | 98.9% | 19.9 |
  | full fine-tune | 5.1% | 99.9% | 60.5 |

* Not run at this size: the dense twin (no baseline for the same compute),
  and a second seed.
