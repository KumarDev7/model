# Navi review: TPU performance, bugs, and fixes

Review of [`theorionic/navi`](https://github.com/theorionic/navi) at commit `ee3128d`.

**Scope.** I read all of `navi/` and `main.py`, the TPU scripts `experiments/train500m.py`, `experiments/sweep_real.py`, `experiments/sweep9.py` and `experiments/prof_batch_scale.py`, and the verdict/analysis notes in `experiments/logs-4m/`. The RL scripts (`rl_*`, `sft_rl_*`, `instill_rl_*`) were skimmed, not reviewed line by line.

**How findings were checked.** Items marked **✅ verified** were reproduced by running the code (JAX 0.10.2 / Flax 0.12.8 on CPU); the commands are in the appendix. Items marked **🔍 profile to confirm** are TPU performance causes read from the code and hardware limits. I have no TPU profile, so confirm them with a trace before investing in the fix.

---

## Summary

| # | Area | Issue | Severity | Fix effort |
|---|---|---|---|---|
| T1 | TPU speed | Second-stage top-k scores a 64×64 grid; 8×8 gives the identical answer | High | 1 line |
| T2 | TPU speed | Value tables sharded on the slot axis while tokens are sharded on batch | High (🔍) | Medium |
| T3 | TPU speed | Everything runs in float32, no bf16 | High | Small |
| T4 | TPU speed | Dense optimizer update over all 537M pool params every step | Medium–High | Medium–Large |
| T5 | TPU speed | No buffer donation, extra full-size gradient copies in `step` | Medium | Small |
| T6 | TPU speed | Host stalls: un-jitted loss, re-jit on every validation, no-KV-cache generation, blocking pickle checkpoints | Medium | Small |
| T7 | TPU speed | `top_k` on TPU is sort-based; `approx_max_k` exists for this | Medium (🔍) | Small |
| T8 | TPU speed | Plain attention; no fused (Pallas) attention kernel | Low at seq 512 | Small |
| C1 | Correctness | **Lion moves pool rows that got no gradient, forever** | **Critical** | Medium |
| C2 | Correctness | Default config gives NaN loss (vocab 339 vs data vocab 1,299) | High | 1 line |
| C3 | Correctness | Hash memory hashes one token, not the `(key, n1, n2)` context | High | Small |
| C4 | Correctness | `python main.py roundtrip` crashes | Low | 1 line |
| C5 | Correctness | Memory grads ×10 under Lion do nothing | Low | 1 line |
| C6 | Correctness | Logged training loss is computed after the update, eagerly | Low | Small |
| E1 | Experiments | FFN vs Pool comparison is not parameter-matched (1.52M vs 2.89M) | High | Small |
| E2 | Experiments | "No collapse found" check never measures routing | High | Small |
| E3 | Experiments | 16.8M-fact runs are at chance by construction (2–4 exposures/fact) | High | Design |
| E4 | Experiments | No anti-collapse mechanisms in learned routing | Medium | Medium |
| E5 | Experiments | Weight decay applied to pool values in `navi/train.py` | Medium | 1 line |
| E6 | Experiments | Loss covers random tokens (3 of 4 unlearnable) | Low | Small |
| E7 | Experiments | Top-k recall unit test can't fail | Low | Small |
| E8 | Experiments | Config fields that do nothing (`value_noise`, `hash_k`, `dropout`, `train`) | Low | Small |
| H1–H5 | Hygiene | 131 MB dataset in git, hard-coded Kaggle paths, empty README, deprecated `orbax` package, key reuse | Low | Small |

**Recommended order:** C1 → T1 → C2/C3 → T3 → T5/T6 → E2 → T2 (profile first) → T4 → Pallas kernels.

C1 and T1 are the two findings most likely to change your results. C1 plausibly explains part of why recall stays at chance at scale. T1 removes most of the memory layer's top-k work with a one-line config change.

---

## Part 1: Why TPU training is slow

### How far from peak it is

The 500M run (`train500m.py`) logs **63k tokens/s** on a v5e-8. A rough FLOP count per trained token:

- Dense backbone (~19M params): ≈ 6 × 19M ≈ 114 MFLOP
- Attention score and value matmuls at seq 512, 8 layers, d 512: ≈ 3 × 4 · 8 · 512 · 512 ≈ 25 MFLOP (the QKV/output projections are already in the backbone count)
- Memory queries and subkey scoring (4 layers × 4 classes × 2 sides × 512 subkeys × 128 dims), forward + backward: ≈ 25 MFLOP

That totals ~165 MFLOP/token × 63k tokens/s ≈ **10 TFLOP/s**. A v5e-8's bf16 peak is about 8 × 197 ≈ 1,570 TFLOP/s, so the run uses **under 1%** of the chips' compute. The time is going to memory traffic, cross-chip communication, sorting and host stalls, not to math. The items below are ordered by expected impact.

### T1. The second-stage top-k grid is 64× larger than needed ✅ verified exactness

`train500m.py:182` uses `side_top=64, cand_k=8`. Every token, for every class and memory layer, the code builds a 64 × 64 = **4,096**-candidate grid, sums it, and runs `top_k(…, 8)` over it (`navi/pkm.py:69-76`).

The best 8 pairs by `s1[i] + s2[j]` always come from the top 8 of `s1` and the top 8 of `s2`. If `i` isn't in `s1`'s top 8, there are 8 rows `i'` with `s1[i'] > s1[i]`, so 8 pairs `(i', j)` all beat `(i, j)`. So **`side_top = cand_k` is exact**, not an approximation. I confirmed it matches brute force on 200 random cases.

Per training step at BS 256 × SEQ 512 = 131k tokens × 4 layers × 4 classes, the grid is **8.6 billion** candidate scores. That is materialized in HBM, passed through a sort-based top-k, and stored for the backward pass. With `side_top=8` it becomes 134 million, **64× less**.

**Fix:** set `side_top = cand_k` everywhere, and make it the default in `MemoryConfig`:

```python
# navi/config.py
cand_k: int = 8
side_top: int = 8   # exact: the top-k of pair sums lies inside top-k x top-k
```

If you want wider candidates for exploration, add noise to the scores instead (see E4).

### T2. Value tables are sharded along the axis the lookup indexes 🔍 profile to confirm

`train500m.py:47-53` shards every `values` table on the slot axis, `PartitionSpec(None, "cores")`, while the batch is sharded on `"cores"`. Each chip's tokens look up slots spread over **all 8 chips**. With an indexed axis that is sharded, XLA's SPMD partitioner usually does one of two things:

- **All-gathers the table** onto every chip before the gather. That's 537M floats ≈ 2.1 GB per chip per step across the 4 memory layers, over the inter-chip links.
- **Masked local gather + all-reduce.** Each chip gathers what it owns, zero-fills the rest, then all-reduces the full `(tokens, classes, k, dim)` output. The backward scatter-add then produces full-size gradients that need a reduce-scatter.

Either way, cross-chip traffic scales with the table or with gathered activations, not with the tiny index tensors. `prof_batch_scale.py`'s own comment ("29 s/it … sharding is the fix") points the same way.

**How to confirm:** capture a trace around a few steps and look for `all-gather`, `all-reduce` or `reduce-scatter` ops whose size matches the value tables or the gathered values:

```python
jax.profiler.start_trace("/tmp/trace")
for _ in range(5):
    p, o, _ = step(p, o, ids, tg)
jax.block_until_ready(p)
jax.profiler.stop_trace()          # open in TensorBoard / XProf
print(step.lower(p, o, ids, tg).compile().as_text()[:20000])   # grep for all-gather
```

**Fix options, simplest first:**

1. **Shard the value dimension instead of the slot axis:** `PartitionSpec(None, None, "cores")`. Every chip holds every slot but only `dim/8` of each row. All-gather the lookup **indices** and weights, which are small (tokens × classes × k int32). Each chip gathers its slice locally and computes its slice of the weighted sum; the output is all-gathered or kept dim-sharded into `w_o`. The backward pass is a local scatter. Communication scales with tokens × k instead of table size.
   - Caveat: at `per_class_dim=128` each chip gets 16-wide rows, which uses 1/8 of a 128-lane TPU vector register. Prefer `per_class_dim` ≥ 8 × 128 per class, or shard over fewer chips.
2. **All-to-all dispatch (MoE-style, "expert parallel" over slots).** Send each lookup to the chip that owns the slot, gather there, then all-to-all the results back. This is the standard design at 1B+ slots, and it composes with a Pallas gather kernel (Part 2).
3. **Replicate small tables.** Tables that fit in HBM (the CPU-scale configs) should just be replicated.

### T3. Everything runs in float32 ✅ verified by code search

No `dtype`/`param_dtype`/`bfloat16` appears anywhere in `navi/` or the TPU scripts. v5e matrix units run bf16 natively, and gathers and scatters are memory-bound, so f32 doubles every byte moved.

**Fix:**
- Compute in bf16 and keep f32 master weights: pass `dtype=jnp.bfloat16` to `nn.Dense`, `nn.Embed` and `nn.SelfAttention`.
- For the pool, store `values` in bf16, since the gather reads them, or keep f32 master values and cast the gathered rows.
- Keep softmax, LayerNorm and the loss in f32.

This roughly halves HBM traffic for the memory layers and speeds up every matmul.

### T4. The optimizer updates all 537M pool parameters every step

Each step reads and writes every pool parameter plus its Lion momentum, even though only the fetched rows got gradient. That's ~537M × (param + grad + momentum) read and ~537M × (param + momentum) written, ≈ 10 GB of HBM traffic per step before any other work. It also causes the correctness bug C1.

**Fix:** update only the rows that were touched.
- **Short term:** mask the update per row, `update = where(row_grad_nonzero, update, 0)`, and freeze the optimizer state for untouched rows. This fixes C1 but still moves dense bytes.
- **Real fix:** a sparse row update. Collect the touched slot IDs (`slots` is already returned), deduplicate them, `segment_sum` the gradients into a compact `(n_touched, dim)` array, run the optimizer only on those rows, and scatter them back. The original product-key memory work used sparse Adam this way. A Pallas kernel (Part 2) can fuse the scatter and update.

### T5. Wasted memory and copies inside `step` ✅ verified by code reading

`train500m.py:205-217`:
- `jax.jit(step)` has no `donate_argnums`. XLA must keep the old **and** new copies of 556M params and the optimizer state, which doubles HBM use and adds copies. Use `jax.jit(step, donate_argnums=(0, 1))`.
- `gm` and `gc_` build two more full-size gradient trees with `zeros_like` just to log two norms, and the `× 10.0` tree is a third (see C5). Compute the norms per subtree instead:

```python
mem_sq  = sum(jnp.sum(x * x) for kp, x in flat_grads if is_mem(kp))
core_sq = sum(jnp.sum(x * x) for kp, x in flat_grads if not is_mem(kp))
```

### T6. The host loop stalls the TPU ✅ verified by code reading

- **Un-jitted loss every 50 steps** (`train500m.py:252`): `float(loss_fn(model, p, ids, tg))` runs the 556M model eagerly, op by op with no fusion, on sharded arrays, and after the update (C6). Return the loss from the jitted `step` instead.
- **Validation recompiles every call** (`train500m.py:162-163`): `@jax.jit def ev` is defined inside `val_loss`, so each validation (every 1,000 steps) compiles a new function. Define it once at module level.
- **Generation without a KV cache** (`generate`): each of 256 tokens × 4 prompts runs a full 512-token forward at batch 1, which is 1,024 latency-bound forwards per generation round. Add a KV cache, or generate less often and with shorter outputs.
- **Blocking pickle checkpoints** (`train500m.py:271`): every 500 steps, all sharded params and optimizer state (~4+ GB) are pulled to the host and pickled while the TPU sits idle. Use `orbax.checkpoint.CheckpointManager` with async saving. It also writes sharded checkpoints without gathering everything to one host.
- **Data:** `feed.batch()` builds numpy windows on the host every step with no prefetch. It's small (0.5 MB/step), but put it behind a 2-deep prefetch queue so host time overlaps device time.

### T7. `top_k` on TPU is sort-based 🔍 profile to confirm

`jax.lax.top_k` lowers to a sort on TPU. Stage 1 (top-8 of 512 subkeys per side) is modest after T1, but at `c1 = c2 = 2048+` it grows. `jax.lax.approx_max_k(scores, k, recall_target=0.95)` is designed for this on TPU.
- Use it for **stage 1 only**, where approximate recall is acceptable.
- Keep stage 2 (k × k candidates, tiny after T1) exact.
- On CPU I measured `top_k` on >2-D inputs about 3–4× slower than on the same data reshaped to 2-D, so flatten leading axes before calling it. Check whether the same holds on TPU.

### T8. Attention is not fused

`nn.SelfAttention` materializes the full `(heads, T, T)` score matrix. At T = 512 that's fine. For longer contexts use the Pallas TPU kernels that ship with JAX, `jax.experimental.pallas.ops.tpu.flash_attention` or `splash_attention`.

---

## Part 2: Will Pallas help?

**Yes, for the memory layer specifically, but only after the fixes above.** Pallas lets you write the lookup as one fused TPU kernel. It does **not** fix wrong configuration or sharding.

| Problem | Does Pallas fix it? |
|---|---|
| T1 oversized grid | No. It's a config value; fix it first. |
| T2 sharding layout | No. Pallas runs per chip; cross-chip data movement is a sharding design choice. |
| T3 f32 | No. Just use bf16. |
| T5/T6 host and copy overhead | No. |
| C1 Lion drift | No. It's an optimizer bug. |
| **Unfused lookup** (scores → top-k → gather → weighted sum all go through HBM) | **Yes.** This is where Pallas pays off. |
| **Dense gradient scatter + dense optimizer (T4)** | **Yes.** A fused sparse scatter-add + row update kernel. |
| **Long-context attention (T8)** | **Yes.** Use the existing flash/splash kernels. |

### Kernel 1: fused gather + weighted sum (forward)

Today the forward pass writes the gathered `(tokens, classes, k, dim)` tensor to HBM, then reads it back to multiply by the weights. A Pallas kernel can use **scalar prefetch** to load the slot IDs into SMEM, DMA just the needed rows from HBM into VMEM, and accumulate `Σ w · value` on-chip, writing only `(tokens, dim)`.

The gather part looks like the standard TPU embedding-gather pattern. This is an untested sketch showing the shape of the API:

```python
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def _row_kernel(idx_ref, w_ref, val_ref, out_ref):
    # idx_ref: prefetched slot ids (SMEM); val_ref: the one row DMA'd for this grid step
    i = pl.program_id(0)
    j = pl.program_id(1)                     # which of the k fetched rows

    @pl.when(j == 0)
    def _():
        out_ref[...] = jnp.zeros_like(out_ref)

    out_ref[...] += w_ref[i, j] * val_ref[...].astype(jnp.float32)
    # j is the innermost grid axis, so the same output block is revisited k times
    # (mark it "arbitrary" in compiler_params dimension_semantics)

def fused_lookup(values, slots, weights):    # values [N, D], slots/weights [T, k]
    T, k = slots.shape
    D = values.shape[1]
    return pl.pallas_call(
        _row_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(T, k),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.SMEM),                      # weights
                pl.BlockSpec((1, D), lambda i, j, idx: (idx[i, j], 0)),     # the fetched row
            ],
            out_specs=pl.BlockSpec((1, D), lambda i, j, idx: (i, 0)),
        ),
        out_shape=jax.ShapeDtypeStruct((T, D), jnp.float32),
    )(slots, weights, values)
```

A production version would:
- process a block of tokens per grid step and double-buffer the row DMAs;
- keep `D` a multiple of 128;
- define a `custom_vjp` whose backward is Kernel 2;
- be tested against the plain JAX version for exact equality.

### Kernel 2: sparse backward + row update

The backward of the gather is a scatter-add of `w · grad_out` into the fetched rows. Fusing it with the optimizer update means untouched rows are **never read or written**. That fixes T4's bandwidth cost and makes C1 impossible by construction. Deduplicate slot IDs first, via sort + segment-sum in plain JAX, so the kernel doesn't need atomics.

### What to expect

I can't give a speed-up number without a profile. The sequence I'd use:
1. Fix T1, T3, T5, T6, then profile. This should be most of the gain.
2. Fix T2 based on what the profile shows.
3. Then the Pallas kernels, whose value is roughly the share of step time still in gather/scatter/HBM-bound memory-layer ops.

For a model where 97% of the parameters live in lookup tables, those ops are likely a large share of what's left.

---

## Part 3: Correctness bugs

### C1. Lion keeps moving pool rows that got no gradient ✅ verified (critical)

`train500m.py:101-102` trains the pool with `optax.lion(lr=3e-3, b1=0.9, b2=0.99)`. Lion's update is `sign(b1·m + (1−b1)·g)`. A row that isn't fetched this step has `g = 0` but non-zero momentum `m` from earlier, so `sign(m) = ±1` and **the row moves a full `lr` step in every dimension, every step**. The momentum decays, but its sign doesn't, so the drift never stops.

Measured: a row that got one gradient and was never fetched again, compared with value init scale std 0.02.

| steps since last fetch | Lion (navi 500m settings) | AdamW (b1 .9, b2 .95) |
|---|---|---|
| 1 | 0.003 | 0.003 |
| 10 | 0.030 | 0.013 |
| 100 | 0.300 | 0.020 |
| 1,000 | **3.0** | 0.020 |

With ~1M slots per class per layer and each slot fetched rarely, most stored values random-walk far away from what was written into them. This is a strong candidate for why recall doesn't build up at scale. Note the VERDICT's conclusion that "one gradient touch does not write a retrievable value" was measured under this optimizer.

**Fix, any of:**
- Use AdamW (with `weight_decay=0` for the pool) and mask updates to touched rows.
- Or a sparse row-wise optimizer (T4). Adagrad or sparse Adam are the usual choices for embedding tables.
- If you keep Lion for memory reasons, zero the update for rows whose gradient is zero this step. Wrap only the pool's optimizer (the `"mem"` branch of `multi_transform`), where every leaf is a table whose last axis is the row width:

```python
def mask_untouched_rows(inner: optax.GradientTransformation) -> optax.GradientTransformation:
    def update(grads, state, params=None):
        updates, state = inner.update(grads, state, params)
        touched = jax.tree.map(lambda g: jnp.any(g != 0, axis=-1, keepdims=True), grads)
        updates = jax.tree.map(lambda u, t: jnp.where(t, u, 0.0), updates, touched)
        return updates, state
    return optax.GradientTransformation(inner.init, update)

mem = mask_untouched_rows(optax.lion(learning_rate=3e-3, b1=0.9, b2=0.99, weight_decay=0.0))
```

The momentum of untouched rows still decays in the background, so the next time a row is fetched it gets a slightly stale push. That is mild; the unbounded drift is gone.

### C2. The default config gives NaN loss ✅ verified

`navi/config.py:8` sets `vocab_size = 339`, which only fits `NAVI_NONCE=64`. `navi/data.py:24` defaults `NAVI_NONCE` to `1024`, so the data vocabulary is **1,299**. Token IDs above 338 index past the embedding table, JAX fills out-of-range lookups with NaN, and `python main.py train` gives `loss: nan`. The experiment scripts avoid it only because they pass `vocab_size=VOCAB`.

**Fix:** derive it and fail loudly on mismatch:

```python
# navi/config.py
from navi.data import VOCAB
vocab_size: int = VOCAB
```

```python
# in the data loader / training loop (host side, before device_put)
assert batch.max() < model_cfg.vocab_size, f"token id {batch.max()} outside vocab {model_cfg.vocab_size}"
```

### C3. Hash memory hashes one token, not the context ✅ verified

The docstring (`navi/pkm.py:88-96`) says slots come from `hash((key, n1, n2) context)`, but `h = ctx_ids.astype(...)` (`pkm.py:99-106`) hashes **each position's own token ID**, and the model passes `ctx_ids=ids` (`navi/model.py:87,90`).
- I tested two triples with different key and n1 but the same n2. Both read the identical slots `[756, 3237, 1622, 7]`.
- The hash path is therefore a per-token lookup table and can't store a fact that depends on three tokens.
- `hash_k` is never used, and the hidden state `x` is ignored, so what is read doesn't depend on the input beyond one token ID.
- This invalidates VERDICT §5's conclusion that "hash placement = learned placement = chance". Context hashing was never tested. It also explains why 83.8% of anchor accuracy survives zeroing the pool: the pool wasn't carrying the facts.

**Fix:** hash an n-gram window ending at each position, and actually gather `hash_k` slots:

```python
def _ngram_hash(ids, n=3):
    # combine ids[t-n+1..t] with position-independent mixing
    h = jnp.zeros_like(ids, dtype=jnp.uint32)
    for s in range(n):
        tok = jnp.pad(ids, ((0, 0), (s, 0)))[:, : ids.shape[1]].astype(jnp.uint32)
        h = _mix32(h * jnp.uint32(0x9E3779B1) ^ (tok + jnp.uint32(s + 1)))
    return h

# _mix32 = the existing xorshift-multiply finalizer from pkm.py, factored into a function
def _hash_path(self, x, ctx_ids):
    cls = jnp.arange(self.cfg.n_classes)
    n_slots = self.cfg.c1 * self.cfg.c2
    h = _ngram_hash(ctx_ids, n=3)                                          # (b, l)
    probes = jnp.arange(self.cfg.hash_k, dtype=jnp.uint32)
    slots = (_mix32(h[..., None] + probes * jnp.uint32(0x85EBCA6B)) % n_slots).astype(jnp.int32)  # (b, l, hash_k)
    v = self.values[cls[:, None, None, None], slots[None]]                # (C, b, l, hash_k, D)
    ...
```

For the recall task, the value is predicted at the n2 position, so a 3-gram ending there covers `(key, n1, n2)`. For real text, use the attention output to make the read content-dependent. Otherwise the hash path is only an n-gram embedding.

### C4. `python main.py roundtrip` crashes ✅ verified

`main.py:147` calls `make_tx(cfg)`, but the signature is `make_tx(cfg, params_like)`, so it raises a `TypeError`. **Fix:** `tx = make_tx(cfg, params)`.

### C5. Memory gradients ×10 under Lion do nothing ✅ verified by code reading

`train500m.py:214` multiplies memory gradients by 10 before a Lion update. Lion uses only the **sign** of its momentum, which scales with the gradient, so ×10 changes nothing (weight decay is 0 for the pool) while costing a full-size tree op. Remove it. If the intent is a higher memory learning rate, that is already `lr=3e-3` vs `3e-4`.

### C6. Logged training loss is recomputed after the update

`train500m.py:252` evaluates the loss on the **updated** params with the same batch, eagerly (see T6). The logged "training loss" is therefore optimistic. Return the pre-update loss from `step`.

---

## Part 4: Experimental validity

### E1. The FFN vs Pool comparison isn't parameter-matched ✅ verified

`navi/train.py:3` says "at matched parameter budget". With the default config the FFN arm has **1,519,360** parameters and the Pool arm **2,893,696** (1.9×). A Pool win is confounded by extra parameters.

**Fix:** widen the FFN arm's hidden size until totals match (or report both matched-params and matched-FLOPs baselines), and print both counts in the summary.

### E2. The collapse check never measures routing ✅ verified by code reading

`experiments/diagnose_collapse.py` promises three checks, including "Routing collapse: distinct slots touched by 32k real queries, top-100 share, Gini". `gini()` is defined (line 56) and **never called**, and no queries are run. Only parameter statistics are computed, and those can't detect unused slots: a slot that is never fetched keeps its random initial values, so it passes the "dead row" and "effective rank" checks.
- VERDICT §6 ("NO collapse found") is not supported by this script.
- `sweep_real.slot_stats` does compute usage stats; that approach is the right one.

**Fix:** run real batches through the model with `return_aux=True`, count slot hits per layer, and report:
- coverage (% of slots hit at least once);
- active share (% of slots with ≥ 10% of a uniform share of hits);
- normalized usage entropy (`exp(H)/N`);
- top-1% traffic share.

Also track these during training, not only on saved checkpoints.

### E3. The 16.8M-fact results are at chance by construction

With 2–4 exposures per fact, ~13% of facts are never seen and most are seen once. Random facts carry no pattern, so an unseen fact can only be guessed. And one or two gradient updates are far below what memorization normally takes: dozens to hundreds of exposures per fact in the knowledge-capacity literature. "Fresh" eval at this scale is mostly unseen facts, so chance is the only possible outcome. The VERDICT's conclusion that "the failure is in the routing distribution" is not something these runs can show.

**Fix:** measure capacity the standard way:
- Fix exposures per fact at a level where the small anchor already reaches ~100% (for example ≥ 100).
- Grow the number of facts at a fixed pool size, and report **facts stored** (accuracy × facts), not accuracy alone.
- Run matched dense baselines at each point.
- Then scale the pool and see if facts stored scales with it.

Fix C1 first; otherwise the optimizer drift confounds the result.

### E4. No anti-collapse mechanisms in learned routing

Scores are raw dot products of unnormalized queries and keys (`pkm.py:62-66`), with no load balancing, exploration noise, or dead-slot revival. Whether routing spreads is left to chance. Mechanisms that work (all implemented in my model at `KumarDev7/cloud`, `memory_pool_model/memory.py`):
1. Cosine scoring (L2-normalize queries and subkeys) with a learned temperature, so a key can't win by growing its norm.
2. Gumbel noise on scores during training, affecting selection only, not the mixing weights.
3. A Switch-style load-balancing loss on each subkey codebook, computed on the noise-free router.
4. A key-diversity loss (penalize off-diagonal subkey cosine).
5. Reviving dead subkeys onto recent real queries, with their optimizer state reset.

### E5. Weight decay is applied to pool values in `navi/train.py`

`make_tx` (`navi/train.py:28-49`) applies AdamW `weight_decay=cfg.weight_decay` to **all** params, including pool values and keys. Decoupled decay shrinks rows every step whether or not they were fetched, which slowly erases rarely used knowledge. The 500m script already sets pool decay to 0; the library default should too. Also exclude LayerNorm scales and biases, which is standard practice:

```python
mask = jax.tree_util.tree_map_with_path(lambda kp, x: x.ndim >= 2 and not _is_mem(kp), params)
optax.adamw(schedule, weight_decay=cfg.weight_decay, mask=mask)
```

### E6. Loss covers random tokens

`navi/train.py:17-20` averages the loss over every position. In the recall data, the key and nonce tokens are random, so 3 of every 4 predicted tokens are unlearnable noise, which dilutes the gradient for the fact positions. For the recall task, weight the loss to the value positions (`targets[:, 2::4]`), or report recall-only loss alongside.

### E7. The top-k recall test can't fail

`main.py:37-41` draws `q1` and `q2` from the **same** key (`qk`), and `k1` and `k2` from the same key (`kk`), so both sides are identical. With `side_top ≥ cand_k` the filter is exact by construction anyway (see T1), so `recall ≥ 0.95` always passes. **Fix:** use independent keys and assert the slot sets are **exactly equal** to `exact_topk_slots`.

### E8. Config fields and flags that do nothing

- `MemoryConfig.value_noise` (config.py:34): never read.
- `MemoryConfig.hash_k` (config.py:41): never read (see C3).
- `ModelConfig.dropout` (config.py:12): never used; attention is `deterministic=True`.
- The `train` argument to the memory layer and `sample_batch(train=...)`: ignored.

Either implement them or delete them. Silent no-op flags make sweeps report conclusions about settings that never took effect.

---

## Part 5: Repository hygiene

- **H1. Large data in git:** `tmp/data/enwik8` (35 MB) and `tmp/data/enwik8.raw` (96 MB) are committed, and `.git` is 71 MB. Remove them, add `tmp/` to `.gitignore`, and download the data in a setup script. Rewriting history with `git filter-repo` would be needed to shrink existing clones.
- **H2. Hard-coded paths:** `/kaggle/working` is hard-coded in many experiments. Use an env var or CLI flag.
- **H3. Two checkpoint systems:** experiments pickle checkpoints while `navi/checkpoint.py` uses Orbax. Standardize on Orbax (async, sharded; see T6).
- **H4. Packaging:** `README.md` is empty, and the `pyproject.toml` description is a placeholder. The `orbax` PyPI package is deprecated; depend on `orbax-checkpoint`. `grain` is listed but unused in the code I read.
- **H5. JAX key reuse:** `navi/data.py:43-45` uses `rng` for `keys` and then splits the same `rng` for `n1`/`n2`. Split once into three keys.
- Minor: `from navi.config import …` at the bottom of `navi/train.py` (line 222) works only because `run()` resolves names at call time. Move it to the top.

---

## Suggested plan

1. **Correctness first:** C1 (optimizer on untouched rows), C2 (vocab), C3 (context hash), C4, C5, E5, E7.
2. **Cheap speed wins:** T1 (`side_top = cand_k`), T3 (bf16), T5 (donation, no extra trees), T6 (jit the loss, compile validation once, async Orbax, KV cache).
3. **Measure:** add slot-usage metrics during training (E2), then profile a TPU step (T2 instructions).
4. **Fix sharding** based on the profile (T2), and add sparse row updates (T4).
5. **Re-run the science:** a parameter-matched A/B (E1), then a capacity sweep with enough exposures per fact (E3), with anti-collapse on and off (E4).
6. **Pallas:** a fused lookup forward plus a sparse backward/update kernel, benchmarked against the plain-JAX path for identical outputs.

---

## Appendix: verification

Run from a clone of `theorionic/navi` on CPU:

```
1) data VOCAB = 1299  ModelConfig().vocab_size = 339
   max token id in batch: 1297  loss: nan
2) make_tx signature: (cfg, params_like)
   make_tx(cfg) -> make_tx() missing 1 required positional argument: 'params_like'
3) hash slots at n2 positions (different key/n1, same n2): [756, 3237, 1622, 7] [756, 3237, 1622, 7] identical: True
4) memory_every=0: params = 1,519,360
4) memory_every=2: params = 2,893,696

Row fetched once, then never again (position after N steps; init std 0.02):
lion  lr=3e-3 b1=.9 b2=.99: [(1, -0.003), (10, -0.030), (100, -0.300), (1000, -2.996)]
adamw lr=3e-3 b1=.9 b2=.95: [(1, -0.003), (10, -0.013), (100, -0.020), (1000, -0.020)]

side_top = cand_k = 8 matches brute-force top-8 over 512x512 on 200 random cases: True
```
