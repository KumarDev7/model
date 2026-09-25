# Navi review: `routing-fix` branch

**Repository:** `theorionic/navi`
**Branch:** `routing-fix` at commit `30d854a` ("fix(pool): E9 usage-balance eliminates coverage collapse - TPU verified", 2026-09-24).
There is no branch called `router-fix`; `routing-fix` is the closest name, so this review covers it.

**Read in full:** `navi/pkm.py`, `navi/model.py`, `navi/train.py`, `navi/config.py`, `navi/data.py`, `navi/pallas_pkm.py`, `main.py`, `experiments/train500m_bpe.py` (the 575M trainer), `experiments/profile_pool.py` and its log, `experiments/diagnose_collapse.py`, `ISSUES.md`, `FIX-01`, `DOC-01`, `DOC-08`.
**Skimmed:** `navi/autopilot.py`, `navi/stream_feed.py`, `tests/`, and the other experiment scripts.

**How items were checked.** Items marked **✅ reproduced** were confirmed by running the branch's own code on CPU with JAX 0.10.2 / Flax 0.12.8. The checks are small and use the repo's modules directly; the script is in the appendix. Everything else comes from reading the code, with line numbers given.

**Severity levels**
- 🔴 **Critical:** results or claims are wrong, or a feature silently does nothing.
- 🟠 **High:** large speed cost, or a design flaw that limits the model.
- 🟡 **Medium:** a smaller bug or a methodology gap.
- ⚪ **Low:** hygiene.

---

## Summary

| ID | Sev | Area | Problem | Checked |
|---|---|---|---|---|
| P1 | 🟠 | TPU speed | The pool read is **~97% of step time** (2,093 ms with pool vs 58.6 ms without) and grows linearly with batch size | repo's profile log |
| P2 | 🟠 | TPU speed | Value table is sharded by slot while tokens are sharded by batch. The lookup crosses shards and likely adds per-token communication | code + profile |
| P3 | 🟠 | TPU speed | Row lookups and their backward scatters are slow on v5e, which has no SparseCore | code |
| P4 | 🟠 | TPU speed | E9 usage counting uses `one_hot(i, c1).sum()`, which does work proportional to tokens × side_top × c1 | code |
| P5 | 🟡 | TPU speed | Returning the full gradient tree from the step, an optimizer that updates the whole pool every step, attention that isn't flash, fp32 activations, no `scan` over layers | code |
| P6 | 🟡 | TPU speed | Eval re-compiles every call and runs a forward pass without `jit` | code |
| K1 | 🟠 | Pallas | The kernel targets the wrong stage: selection is 20 ms of a 385 ms read | repo's DOC-01 |
| C1 | 🔴 | trainer | **The compiled step never sees later changes.** The dense→sparse switch, the E9 usage refresh, and autopilot's model/LR rebuilds are all ignored | ✅ reproduced |
| C2 | 🔴 | pkm dense | The dense read's einsum **mixes all classes together** | ✅ reproduced (error 0.103; fixed version 3e-9) |
| C3 | 🔴 | pkm E9 | Balancing offsets **leak into the readout weights**. Side-2 usage is averaged over classes, and usage counts candidates instead of actual reads | ✅ reproduced |
| C4 | 🔴 | pkm E6 | The learned injection head **gets zero gradient** and has O(c1·c2) parameters per token | ✅ reproduced |
| C5 | 🔴 | pkm hash | Hash slots come from the **current token only**, not the context | ✅ reproduced |
| C6 | 🟠 | trainer | Validation, generation and coverage run at temperature **4.0**; training ends at **1.0** | code |
| C7 | 🟠 | pkm lb | The entropy "balance" loss flattens each token's routing instead of spreading usage | code |
| C8 | 🟡 | pkm | `_hash_path` adds the residual twice | code |
| C9 | 🟡 | pkm E4 | Exploration offsets override E9 and change the raw scores used by the balance loss and hash rows | code |
| C10 | 🟡 | pkm E2 | The 3-factor read is not exact with default settings, despite its docstring | code |
| C11 | 🟡 | pkm E3 | The hash row's score is computed then ignored; `lb_eps` is silently overridden | code |
| C12 | 🟡 | model | The `NAVI_REMAT` toggle does nothing; remat is never applied | code |
| C13 | 🟡 | defaults | `python main.py train` gives **NaN loss**: model vocab is 339, data vocab is 1,299 | ✅ reproduced |
| C14 | 🟡 | CLI | `python main.py roundtrip` crashes: `make_tx(cfg)` is missing an argument | ✅ reproduced |
| C15 | 🟡 | A/B | The "parameter-matched" A/B is 1.52M vs 2.90M parameters | ✅ reproduced |
| C16 | 🟡 | optimizer | `navi/train.py` applies weight decay to pool values and keys | code |
| M1–M6 | 🟡 | method | The collapse diagnostic can't detect collapse; the 16.8M-fact regime; the E9 and dense-warmup results depend on C1–C3; the unit test can't fail | code |
| H1–H6 | ⚪ | hygiene | Unused config fields, prints during tracing, hard-coded `/kaggle` paths, a 71 MB git history | code |

**Most important:** the compiled training step is **frozen at its first trace** (C1). Every runtime mechanism that relies on mutating a Python variable (dense warmup, E9 balancing, autopilot) has no effect in `train500m_bpe.py`. Together with C2 and C3, the recent routing-fix conclusions need to be re-run before anyone relies on them.

---

## Part 1: TPU performance. Why training is slow

### The evidence already in the repo

`experiments/logs-4m/profile_pool.log` (8 × v5e cores, d_model 512, 8 layers, 4 memory layers, c1 = c2 = 512, 4 classes, BS × SEQ = 256 × 512):

| Run | Step time (fwd + bwd) |
|---|---|
| Full model, BS 256 | **2,093 ms** |
| Full model, BS 128 | 1,056 ms |
| Full model, BS 64 | 555 ms |
| Same model **without the pool** | **58.6 ms** |
| Without the balance loss | 2,077 ms |
| Values frozen (no value gradient) | 2,107 ms |

What this shows:
1. **The pool is ~97% of step time** (35× slower than no pool).
2. **Cost scales linearly with tokens** (555 → 1,056 → 2,093 ms). The slow part is work done for every token, not a fixed per-step cost such as updating or all-gathering the whole table.
3. **It isn't the value gradient or the balance loss.** Freezing values or removing the balance loss changes almost nothing.
4. `DOC-01` measured the pieces of one memory-layer read: **selection 20 ms of 385 ms**. Top-k isn't the bottleneck either. What remains is the row lookup, the operations around it, and their backward pass.

The profile script crashed before its last section (`TypeError` in `pkm_step`, `profile_pool.py:170`), and its HLO census printed an empty list (`hlo top: []`). So nobody has yet seen which operations take the time. **Profile properly first**; see "How to profile" at the end of Part 1.

### P1/P2 🟠 Lookups across sharded memory (most likely main cost)

`train500m_bpe.py:78-95` shards `values` along the slot axis, `PartitionSpec(None, "cores", None)`. Everything else is replicated and tokens are split across cores by batch. The read at `pkm.py:313`, `self.values[class_idx, slots]`, looks up **data-dependent rows along the sharded axis**. The compiler (GSPMD) can't know which core holds a row, so it has to choose one of two strategies:
- **Copy the whole table to every core** each step (all-gather) and reduce-scatter its gradient, or
- **Look up rows on every core with masking, then all-reduce** the (tokens × k × D) results. This redoes the per-token work on every core and adds per-token communication.

Cost that grows linearly with batch fits the second strategy. **Confirm it before changing anything:**
```python
lowered = step_jit.lower(p, o, ids, tg, jnp.float32(1.0), jnp.float32(0.15), vt)
hlo = lowered.compile().as_text()
import re; print(sorted(set(re.findall(r"(all-gather|all-reduce|reduce-scatter|all-to-all|collective-permute)", hlo))))
```
Also record an XProf trace (see below) and check how much time the collectives take.

**Fixes (pick based on the profile):**
1. **Quick A/B test:** replicate `values` (`PartitionSpec()`) at a smaller pool that fits, and compare step time. If it drops sharply, sharding is the problem.
2. **Replicated values with sharded optimizer state (ZeRO-1 style):** reads are local; gradients are reduce-scattered; each core updates only its shard of the optimizer; updated values are all-gathered **once per step**. Communication then depends on table size, not token count.
   - Memory: values are 268 MB per layer in bf16, so 4 layers = 1.07 GB replicated.
   - Adam state for values is 2.1 GB per layer in fp32. It must stay sharded, or be kept in bf16.
3. **Route each token's request to the core that owns the row** (like expert-parallel MoE). Use `shard_map` and `all_to_all` to send slot requests to their owner, look up locally, then `all_to_all` the rows back. Communication becomes tokens × k × D, and nothing is computed twice. This is the design that scales to 16.8M+ slots per class.
4. **Remove duplicate requests** in a batch before communicating (`jnp.unique` with a fixed `size`). Pool reads cluster heavily (Gini 0.73–0.86), so this can cut traffic several-fold.

### P3 🟠 Row lookups and their backward pass on v5e

- **v5e has no SparseCore.** v5p and v6e do; SparseCores exist to make embedding lookups fast. On v5e, gathers and scatters run on the TensorCore, which is slow for random rows.
- **Volume:** each step does 131k tokens × 4 classes × `cand_k` (8–16) × 4 layers ≈ **17–34M random row reads forward**, each only 256 bytes (D = 128 in bf16).
- **The backward pass adds scatters.** Every `take_along_axis` on float scores (`g1`, `g2`, `g2_vals`, `scores`; `pkm.py:229-258`) becomes a scatter in the backward pass. So does the value gather itself.

**Fixes**
- **Get read scores directly from the chosen keys.** Select slots with a stop-gradient top-k, then compute the read scores as `q1·k1[pi1] + q2·k2[pi2]`, gathering rows from the small key tables. That removes three float scatters over large score tensors. Key-table gradients go to tables of c1 × D, which are cheap.
- **Use fewer, bigger rows.** Rows of 512 bytes to 1 KB suit DMA better than 256 bytes. For example, 2 classes × D = 256 moves the same number of bytes in half as many rows.
- **Remove duplicate slots before looking them up** (as in P2) and scatter the results back to the tokens.
- After that, a Pallas lookup kernel may help. See Part 2.

### P4 🟠 E9 usage counting does far more work than needed

`pkm.py:366-369`:
```python
hits1 = jax.nn.one_hot(i1, c.c1).sum(axis=(0, 1, 3))   # (b,l,C,side) -> (b,l,C,side,c1)
```
- This does b·l·C·side_top·c1 work: 131k × 4 × 64 × 512 ≈ **1.7 × 10¹⁰ compare operations per memory layer per step**, forward only.
- If the compiler doesn't fuse it, it also needs 69 GB of scratch memory.
- The trainer enables it by default (`NAVI_BALANCE_BETA=0.5`).

The profile above ran without it, so the production step is slower than the profile shows. **Fix:** a scatter-add over flat indices, which is O(tokens × side):
```python
flat = (i1 + jnp.arange(C)[None, None, :, None] * c1).reshape(-1)
hits1 = jnp.zeros(C * c1).at[flat].add(1.0).reshape(C, c1) / n_tok
```
(C3 also argues that final picks should be counted, not side candidates. That makes this about 8× cheaper again.)

### P5 🟡 Other costs inside the training step

| Where | Problem | Fix |
|---|---|---|
| `train500m_bpe.py:498` | The step returns the **full gradient tree `g`** every iteration. That keeps an extra params-sized buffer alive (including the value tables) and blocks buffer reuse | Compute the two gradient norms inside the step and return 2 scalars |
| `make_tx` (`:189-228`) | Adam updates **every row of the pool** every step: 4 layers × (params + m + v) read and written, even though a step touches only a few % of rows. Cost grows with pool size and will dominate at 16.8M slots per class | Sparse update: find the touched rows, update only those, write them back ("lazy Adam") |
| `model.py:79` | `jax.nn.dot_product_attention(..., is_causal=True)` on TPU uses the plain XLA implementation. The fused path is only cuDNN on GPU. The comment "lowers to the fused TPU FlashAttention kernel" is wrong: the full (l × l) scores are materialized | Use Pallas **splash attention** (`jax.experimental.pallas.ops.tpu.splash_attention`). Modest gain at SEQ 512; large at 2k+ |
| everywhere | Activations are fp32 (`Dense` and `Embed` have no `dtype`). TPU matmuls already run in bf16, but every activation stored and read from memory is fp32, which doubles traffic and remat pressure | `dtype=jnp.bfloat16` for compute, keep fp32 params (mixed precision) |
| `model.py:109` | The `NAVI_REMAT` toggle chooses `Block` or `_identity(Block)`, which are the **same class**. `nn.remat` is never applied (see C12). ISSUE-03's "remat off" speedup can't be real, and big configs aren't getting the memory savings they assume | `BlockCls = nn.remat(Block) if REMAT else Block` |
| `model.py:110-150` | The Python loop over blocks unrolls the whole graph, so every relaunch recompiles for 3–6 minutes (ISSUE-03) | `nn.scan` over (memory block, FFN block) pairs. The compile cache helps but doesn't remove this |
| `:584-617` | Per-step work on the host: unigram histogram and KL drift in numpy on the main thread, `jnp.asarray(visits)` uploading 1 MB each step, and two separate `device_put`s per batch | Compute drift in the prefetch thread, keep the visit table on the device, upload one window and slice it on the device |

### P6 🟡 Evaluation is slow

- **`val_loss` recompiles every call.** It defines `@jax.jit def ev` inside the function (`:280-285`), so each call creates a new jitted function and compiles again. That's a full forward compile every `GEN_EVERY` steps. Define `ev` once at module level.
- **`pool_coverage` runs without `jit`.** It calls `model_ra.apply` directly (`:317`), so it runs one operation at a time on TPU.
- **Eval inputs have no sharding.** `jax.device_put(win[:, :-1])` is placed on a single device while `params` are sharded, so XLA has to reshuffle data. Use `BATCH` sharding like training does.

### How to profile (do this first)

```python
jax.profiler.start_trace("/tmp/xprof")
for _ in range(5):
    p, o, l, g = step(p, o, ids, tg, temp_at(i), None)
jax.block_until_ready(p)
jax.profiler.stop_trace()
# open with TensorBoard / XProf: look at "Op profile" and the collectives lane
```
Also fix `profile_pool.py:170`: the `pkm_step` lambda takes 1 argument but is called with 2. Fix the HLO census, which returned `[]`.

### Recommended order for speed work
1. Profile and read the HLO collectives (P2).
2. Fix C1 so the dense read doesn't silently run all training. **This may be the biggest single speed issue if `NAVI_DENSE_STEPS` > 0.**
3. Fix P4 (usage counting) and P5 (returning gradients, remat).
4. Change the sharding strategy (P2), then remove duplicate lookups and use bigger rows (P3).
5. Add splash attention, bf16 activations, and `scan` over layers.
6. Only then consider the Pallas lookup kernel (K2).

---

## Part 2: Will Pallas help?

### K1 🟠 The existing kernel targets the wrong stage

`navi/pallas_pkm.py` replaces **only the final top-`cand_k` selection** over the small (side × g2k) grid. `DOC-01` measured that stage at **20 ms out of a 385 ms read**, so even an infinitely fast kernel would save about 5%. The kernel also has problems of its own:
- **Tiny blocks:** `PB = 8` rows per program (`:191`) means about 65k grid steps of 8 × 512 tiles. Per-step overhead dominates.
- **Unused inputs:** `i1_ref` and `i2t_ref` are passed in and copied to the chip, but the kernel body never reads them (`:200-203`). `c2` is unused too.
- **Stale docstrings:** the docstrings still describe a bitonic network and "fuses q·k → top-k → gather". The code only does iterative argmax.
- DOC-01's conclusion ("XLA top_k wins, don't integrate") is **correct for this kernel**. It doesn't mean Pallas can't help the pool.

### Where Pallas can help

| Candidate | Expected value | Notes |
|---|---|---|
| **Splash / flash attention** (ready-made `jax.experimental.pallas.ops.tpu.splash_attention`) | Medium at SEQ 512, high at 2k+ | Drop-in replacement. Removes the (l × l) scores, which also cuts remat and memory pressure |
| **Fused bag-of-rows lookup** (value read) | Potentially high, **but only after P2 is fixed** | Pre-load the slot IDs as scalars (`PrefetchScalarGridSpec`), keep `values` in HBM (`memory_space=pl.ANY`), and copy the k rows per token into VMEM with double-buffered `pltpu.make_async_copy`. Do the weighted sum in VMEM and write `h`. Tile 128–512 tokens per program so copies overlap compute. **Template:** JAX's own paged-attention kernel (`jax.experimental.pallas.ops.tpu.paged_attention`) uses exactly this pattern: an index table in SMEM plus async copies from HBM |
| **Backward: value-gradient accumulation** | High at large pools | Avoid conflicting scatters: sort (slot, token) pairs, add up equal slots (sum by segment), and write each touched row once. This naturally feeds the sparse optimizer update in P5 |
| Fused `q·kᵀ` + side top-k | Low | XLA already runs the matmul on the MXU and `top_k` well (selection is 20 ms) |
| Final top-k selection (what DOC-01 tried) | **None** | Measured: 11× slower than XLA |

**Order matters.** If each lookup crosses cores (P2), a kernel can't remove that communication. Fix the sharding, remove duplicates, and profile again. Write the lookup kernel only if the gather is still a large share of step time.

---

## Part 3: Correctness bugs and their fixes

### C1 🔴 The compiled step never sees later changes ✅ reproduced

**What happens.** `jax.jit` caches its trace by argument shapes and dtypes. **Python variables that the function captures are read once, when it first traces.** Changing them later has no effect unless the function is compiled again.

Tiny repro:
```python
box = [True]; f = jax.jit(lambda x: x * (2.0 if box[0] else 3.0))
f(1.0)  # 2.0
box[0] = False
f(1.0)  # still 2.0
```

**Where this breaks things in `experiments/train500m_bpe.py`:**
- **Dense→sparse switch** (`:455-457`, `:632-634`). `dense_box[0] = False` never reaches `step_jit`. With the default `NAVI_DENSE_STEPS=1000`, **the whole run stays on the dense read**. The log even says "(one retrace expected)". To check a run: after `DENSE->SPARSE switch` there must be a new `[trace] PKM sparse two-sided top-k path compiling` line. If it's missing, the switch never happened.
- **E9 usage refresh** (`:479`, `:490`, `:724`). `usage_box[0] = refresh_usage(...)` replaces the tables, but the step keeps using the **initial zeros**.
  - With all-zero tables, `rel = clip(0/(0+1e-9) - 1) = -1` for every subkey. That's the same offset everywhere, so the ranking doesn't change.
  - **E9 therefore does nothing in the production trainer.**
  - The updated tables the pool returns in `aux` are also thrown away (`loss_fn` returns only the loss).
- **Autopilot** (`:443-452`, `navi/autopilot.py:12-18`).
  - `rebuild_models()` and `rebuild_tx()` swap `model_ra` and `tx`, but `step_jit` keeps the originals.
  - The docstring's claim that "mutating the dataclass field takes effect on next jitted call" is wrong. `MemoryConfig` is a frozen `struct.dataclass` and is fixed into the trace.
  - The autopilot's LR and `lb_weight` actions are logged but never applied.

**Fix.** Make every changing value an explicit input or output of the jitted step:
```python
@partial(jax.jit, static_argnames=("dense",), donate_argnums=(0, 1, 2))
def step(params, opt, usage, ids, tg, temp, eps_t, visits, *, dense):
    (loss, (usage_new, lb)), g = jax.value_and_grad(loss_fn, has_aux=True)(
        params, ids, tg, temp, eps_t, visits, usage, dense)
    upd, opt = tx.update(g, opt, params)
    return optax.apply_updates(params, upd), opt, usage_new, loss, grad_norms(g)
```
- `loss_fn` must **return** the per-block `usage` from `aux`, and the loop feeds it back every step.
- For autopilot changes, either pass the numbers (lb_weight, eps, LR scale) as traced scalars, or rebuild `step_jit` after `rebuild_models()` or `rebuild_tx()`.

### C2 🔴 The dense read mixes classes ✅ reproduced

`pkm.py:113`:
```python
h = jnp.einsum("bln,cnd->bcd", w, vals)   # w: (chunk, C, n_slots), vals: (C, n_slots, D)
```
`l` labels the class axis of `w` but is missing from the output, so it is **summed**. Every class plane gets the **sum of all C class softmaxes**, whose total weight is C, instead of its own.

Measured against the exact full-softmax read (sparse path with `cand_k` = all slots): **max error 0.103**. With the fix below: **2.8e-9**.

**Fix:**
```python
h = jnp.einsum("bcn,cnd->bcd", w, vals)
```
Also, the `print` on `pkm.py:83` sits *before* the docstring, which turns the docstring into an ignored string. Move the print below it.

**Impact:** DOC-04 ("dense warm-up fix") was measured with this bug, and together with C1 possibly for the whole run. Re-run it.

### C3 🔴 E9 balancing distorts the read ✅ reproduced

**Bug 1: offsets reach the read weights.** The comment at `pkm.py:189-193` says selection offsets don't affect the readout softmax, which "keeps RAW scores". But `g1` and `g2` are gathered from the offset scores `sel1` and `sel2` (`:229-230`), and the readout softmax uses those (`:319`). In the test, readout scores differ from the raw `s1[i1]+s2[i2]` by **1.0** (β = 0.5, so ±0.5 per side). Hot slots get lower read weight, and the gradient is biased.

**Bug 2: side 2 is averaged over classes.** `r2 = clip(u2s.mean(axis=0) / ...)` (`:198`) averages side-2 usage across classes, while side 1 is per class.

**Bug 3: usage counts candidates, not reads.** Usage counts every **side_top candidate** (`i1`, `i2`: 64 per side), not the slots actually read (`pi1`, `pi2`: cand_k). Balancing candidate lists is not the same as balancing reads.

**Fix:**
```python
# select with offsets...
i1 = jax.lax.top_k(sel1, c.side_top)[1]
i2 = jax.lax.top_k(sel2, c.side_top)[1]
# ...but READ with raw scores
g1 = jnp.take_along_axis(s1, i1, axis=-1)
g2 = jnp.take_along_axis(s2, i2, axis=-1)
# per-class relative usage on both sides
r2 = jnp.clip(u2s / (u2s.mean(axis=-1, keepdims=True) + 1e-9) - 1.0, -1.0, 1.0)
sel2 = s2 - c.balance_beta * r2[None, None]
# count final reads (pi1/pi2), via scatter-add (see P4)
```
If you want the offsets to change which slots win the final top-k too, build `flat` twice: once from `sel` scores to choose `f_idx`, once from raw scores to read.

### C4 🔴 The learned injection head (E6) never learns ✅ reproduced

`_read_inject` (`pkm.py:380-459`) uses `inj_scores` only through `top_k` **indices**. The injected rows get a fixed weight (`hybrid_eps / inject_k`), and their scores `hsc` come from `s1` and `s2`. So `inj_in` and `inj_out` get **zero gradient**: measured |grad| = 0.0, versus 0.16 for `w_q`. The head stays at its random initialization.

It also doesn't scale:
- `inj_out` has `inject_hidden × C·c1·c2` parameters, which is **67M per memory layer** at c1 = c2 = 512.
- It produces **131k × 4 × 262,144** logits per step, about 550 GB in fp32. That undoes the product-key design.

**Fix:** give the head a path to the loss, for example by adding its score to the read logits of the injected rows (`scores_inj = hsc + inj_scores[islots]`). Make it factorized, like the product keys (two heads over c1 and c2), instead of one dense layer over c1·c2. Or drop E6.

### C5 🔴 Hash slots use only the current token ✅ reproduced

`_hash_path` (`pkm.py:531-538`) and the hybrid hash (`:273-279`) hash `ctx_ids` element by element, so the slot depends only on **the token at that position** (plus the constant sequence length).

In the test, two triples with different `key` and `n1` but the same `n2` read the **same 4 slots** `[756, 3237, 1622, 7]`. The docstring's "(key, n1, n2) context maps to fixed value rows" is false. A lookup keyed on one token can't store facts keyed on three tokens. It's a unigram embedding table. This undermines:
- the master VERDICT §5 conclusion that hash placement doesn't work (context hashing was never tested), and
- the E3/E7 hybrid-hash coverage results.

`hash_k` is also never used.

**Fix:** hash an n-gram window, and read `hash_k` slots:
```python
def ngram_hash(ids, n=3):
    h = jnp.zeros_like(ids, dtype=jnp.uint32)
    for j in range(n):                                   # tokens t-j
        tok = jnp.pad(ids, ((0, 0), (j, 0)))[:, : ids.shape[1]].astype(jnp.uint32)
        h = (h ^ (tok + jnp.uint32(0x9E3779B9) + (h << 6) + (h >> 2))) * jnp.uint32(0x85EBCA6B)
    return h ^ (h >> 16)
# slots_j = (ngram_hash(ids) + j * PRIME_j) % n_slots   for j in range(hash_k)
```
For the fact task, `n = 3` covers (key, n1, n2) at the n2 position.

### C6 🟠 Evaluation runs at a different temperature than training

- **Training** passes `mem_temp = temp_at(i)`, which ramps from `NAVI_TEMP_START` to **1.0** (`:148-155`, `:167`).
- **Validation, generation and coverage** call `model.apply(...)` without `mem_temp` (`:249`, `:284`, `:317`), so they use `score_temp = TEMP_END = 4.0` (`:350`).

The read is **4× sharper at eval than in training**. Slot selection is unchanged, since top-k doesn't depend on temperature, but the mixing weights differ, so reported validation bpc and generations don't measure the trained model.

`NAVI_TEMP_END` has no effect on training.

**Fix:** pass the same `mem_temp` to every eval call (or make `temp_at` end at `TEMP_END`), and log the temperature used in each eval line.

### C7 🟠 The entropy "balance" loss works against sharp routing

`lb = -(H(softmax(s1)) + H(softmax(s2)))` per token (`pkm.py:346-358`, duplicated in the dense, inject and read3 paths) **maximizes each token's routing entropy**. That flattens every query's scores toward uniform, which makes selection closer to random and fights specialization. It does **not** directly spread usage across the pool.

The comment's claim that the Switch `n·Σ f_i·P_i` loss "is vacuous for sparse top-k (f_i over the gathered subset is always 1/k)" is a misreading. `f_i` must be the fraction of **all tokens** whose hard top-k picked subkey *i*, over the **full** c1 table. That's informative, and it's the standard load-balancing loss.

**Fix: use the aggregate distribution.**
```python
# per class, side 1 (same for side 2)
P = jax.nn.softmax(s1, -1).mean(axis=(0, 1))                                  # (C, c1)  mean router prob
f = jax.lax.stop_gradient(count_hits(i1_final) / (n_tok * k))                 # (C, c1)  fraction of picks
switch = c1 * (f * P).sum(-1).mean()                                          # = 1 when balanced
# or maximize the MARGINAL entropy H(mean_t softmax(s1_t)), optionally minus per-token entropy
```

### C8 🟡 `_hash_path` adds the residual twice

`pkm.py:546` returns `x + self.w_o(hsum)`, and `Block` adds `x` again (`model.py:94`), so the result is `2x + w_o(...)`. In pure-hash mode every memory layer doubles the residual stream. **Fix:** return `self.w_o(hsum)`.

### C9 🟡 E4 exploration overrides E9 and changes raw scores

When `explore_beta > 0` (`pkm.py:218-228`):
- It **replaces** `s1` and `s2` (raw scores) with offset scores, so the balance loss (`:354`) and hybrid hash scores (`:309`) later use them.
- It recomputes `i1` and `i2` from `s1 + offsets`, which **discards E9's `sel1`/`sel2`** and wastes the first `top_k`.
- `g1` is then gathered from `sel1` using indices ranked by a different score.

**Fix:** keep one `sel = s - beta9*rel + beta4*visit_off` for selection and use raw `s` for everything else.

### C10 🟡 The 3-factor read (E2) isn't exact with defaults

The docstring (`pkm.py:461-468`) says exact. That requires every factor's candidate count to be ≥ `cand_k`. The defaults are `cand3 = 4 < cand_k = 8` (`config.py`), so top-k over the 3-factor grid can miss slots.

**Fix:** use `max(cand2, cand_k)` and `max(cand3, cand_k)`, or change the docstring. `_read3` also ignores E9, hybrid, `lb_eps` and `dense`. Either raise an error when those are enabled together, or implement them.

### C11 🟡 Hybrid hash (E3): its score is computed then ignored

- `hs` is computed and appended to `scores` (`:309-312`), but the hybrid readout uses a fixed `eps` for the hash row (`:336-339`).
- The comment "scored by the same q·k product … fair competition in the softmax" (`:269-272`) is no longer true.
- When `hybrid_hash` is on, the `lb_eps` floor (`:325-326`) is overwritten by the recomputed `w`.

**Fix:** either drop `hs`, or use it: `w = softmax(concat(t*scores_r, t*hs + log(eps)))`. Raise an error if `lb_eps` is set together with `hybrid_hash`.

### C12 🟡 The remat toggle does nothing

`model.py:21-22,109`: `_identity(Block)` returns `Block`, so both branches are the same class. `nn.remat` is never applied.

**Fix:** `BlockCls = nn.remat(Block, static_argnums=(2,)) if REMAT else Block`. Mark boolean arguments such as `train` and `dense` static. Then re-measure ISSUE-03.

### C13 🟡 The default setup gives NaN loss ✅ reproduced (unchanged from master)

`config.py:8` sets `vocab_size = 339`, which fits `NAVI_NONCE=64`, but `data.py:24` defaults to `NAVI_NONCE=1024`, so `VOCAB = 1299`. IDs ≥ 339 index past the embedding table. JAX fills out-of-range lookups with NaN, and the loss becomes `nan`.

**Fix:** `ModelConfig(vocab_size=navi.data.VOCAB)` in `navi/train.py:83` and `main.py`, or make the default `NAVI_NONCE=64`.

### C14 🟡 `main.py roundtrip` crashes ✅ reproduced

`main.py:152` calls `make_tx(cfg)`, but the signature is `make_tx(cfg, params_like)`. **Fix:** `make_tx(cfg, params)`.

### C15 🟡 The A/B isn't parameter-matched ✅ reproduced

The `navi/train.py` docstring says "matched parameter budget", but the FFN model has **1,521,664** parameters and the Pool model **2,896,000** (1.9×).

**Fix:** widen the FFN model, or shrink the pool, until totals match within ~5%. Report both numbers and also match compute (FLOPs per token).

### C16 🟡 `navi/train.py` decays the pool

`make_tx` (`navi/train.py:31-52`) applies AdamW `weight_decay` to all parameters, including `values`, `k1` and `k2`. Decoupled decay shrinks rows even when they weren't read, so rarely used knowledge fades. The 575M trainer already excludes the pool from decay; the core library doesn't.

**Fix:** `optax.adamw(..., mask=lambda p: tree_map_with_path(lambda kp, x: x.ndim >= 2 and not _is_mem(kp), p))`.

---

## Part 4: Problems with the experiments and conclusions

| ID | Problem | Fix |
|---|---|---|
| M1 | **`diagnose_collapse.py` can't detect routing collapse.** The docstring promises distinct slots, top-100 share and Gini, but `gini()` (`:56`) is never called. The script only checks key cosines, value norms and rank (`:41-55`). A slot that is never read keeps its random starting values, so `dead_frac` is 0 whether or not it's used. The "NO collapse found" result is unsupported | Count actual reads from real queries: distinct slots, share of reads going to the top 1%, Gini, and `exp(entropy)/N` from the final top-k indices at the eval temperature |
| M2 | **16.8M facts at 2–4 exposures each.** At that budget, random facts can't be memorized with any placement method, and "fresh" draws are mostly unseen facts, so chance accuracy is guaranteed. The claim that the cause is "the routing distribution" doesn't follow | Capacity sweep: fixed exposures per fact (≥ 50–100), growing numbers of facts, pool vs a parameter-matched dense model |
| M3 | **The E9 "coverage collapse eliminated" result** (DOC-08, commit `30d854a`) can't come from `train500m_bpe.py` as written: C1 means the usage tables never update, and C3 means offsets distort the read. If the TPU proof used another harness, write down which one | Re-run in the production trainer after fixing C1, C3 and P4 |
| M4 | **DOC-04 dense warmup** results were produced with the class-mixing dense read (C2), and possibly with no dense→sparse switch at all (C1) | Re-run after the fixes |
| M5 | **The recall unit test can't fail.** `main.py:38-41` uses the same random key for `q1`/`q2` and for `k1`/`k2`. The two-sided filter is exact whenever `side_top ≥ cand_k`, so `recall ≥ 0.95` always passes | Separate keys, and assert *equality* with brute force, including cases where `side_top < cand_k` |
| M6 | **Small validation set.** ISSUES.md already notes 741 tokens and ±0.05 bpc noise. Synthetic `navi/train.py` also scores all tokens, three quarters of which are random, which buries the recall signal | ≥ 4k validation tokens at multiple offsets, several seeds, and loss only on answer positions for fact tasks |

---

## Part 5: Hygiene

- **H1 Unused config.** `value_noise`, `hash_k`, `ModelConfig.dropout` and `NAVI_TEMP_END` (see C6) do nothing. The `train` argument to the pool and `sample_batch(train=...)` are ignored. Remove them or implement them.
- **H2 Prints during tracing.** `print("[trace] …")` inside traced functions (`pkm.py:83,186,394,470`, `train500m_bpe.py:523`) fires on every retrace, which is fine, but on `pkm.py:83` it also disables the docstring. Use `logging` and put prints after docstrings.
- **H3 Hard-coded paths.** `/kaggle/working` appears in `train500m_bpe.py`, `diagnose_collapse.py` and `run_slot_stats.py`. Experiments save pickle checkpoints while the library uses Orbax. Use one checkpoint path and format, set by environment variable or CLI.
- **H4 Repo size.** The enwik8 files (131 MB) are deleted on this branch but remain in git history, so `.git` is 71 MB. Many `tmp/*.log` files and an 80k-line tokenizer JSON are committed. Remove large files from history (`git filter-repo`) and keep data and logs out of git.
- **H5 Packaging.** `README.md` is empty on master, and the `pyproject.toml` description is `"Add your description here"`. The `orbax` PyPI package is deprecated; use `orbax-checkpoint`.
- **H6 Random-key reuse.** `data._draw` uses `rng` directly and then `split(rng)`. Split first and use separate subkeys.

---

## Part 6: Suggested order of work

1. **C1:** make usage, dense mode and hyperparameters explicit inputs to the step. This single fix decides whether E9, dense warmup and autopilot do anything.
2. **C2, C3, C6:** one-line to few-line fixes that invalidate current measurements.
3. **P4, P5** (returning gradients, remat) and **P6:** cheap speed wins.
4. **Profile** (Part 1), then change the pool's **sharding strategy** (P2) and **remove duplicate lookups** (P3).
5. **C5** (n-gram hash) and **C7** (proper load-balancing loss). Then re-run M1–M4 with read-based coverage metrics.
6. Add **splash attention** and bf16 activations. Consider a **Pallas lookup and backward kernel** only if the profile still shows the lookup dominating after step 4.
7. Clean-ups: C8–C16 and H1–H6.

---

## Appendix: reproduction script

Run from the `navi` checkout on `routing-fix` (CPU is enough):

```python
import jax, jax.numpy as jnp, numpy as np
from navi.config import MemoryConfig
from navi.pkm import ProductKeyMemory
x = jax.random.normal(jax.random.PRNGKey(0), (1, 3, 16))

# C1: jit ignores mutated closure state
box = [True]; f = jax.jit(lambda v: v * (2.0 if box[0] else 3.0))
print(float(f(1.0))); box[0] = False; print(float(f(1.0)))          # 2.0, 2.0

# C2: dense read vs exact full-softmax read (sparse path with cand_k = all slots)
cfg = MemoryConfig(c1=4, c2=4, n_classes=2, cand_k=16, side_top=4)
m = ProductKeyMemory(cfg=cfg, per_class_dim=8, value_dim=16); p = m.init(jax.random.PRNGKey(1), x)
print(float(jnp.abs(m.apply(p, x, dense=True)[0] - m.apply(p, x)[0]).max()))   # ~0.10 (should be ~0)

# C4: learned-inject head gradient
c6 = MemoryConfig(c1=8, c2=8, n_classes=2, cand_k=4, side_top=4, learned_inject=True, inject_k=2)
m6 = ProductKeyMemory(cfg=c6, per_class_dim=8, value_dim=16); p6 = m6.init(jax.random.PRNGKey(1), x)
g = jax.grad(lambda q: m6.apply(q, x)[0].sum())(p6)["params"]
print(float(jnp.abs(g["inj_in"]["kernel"]).sum()), float(jnp.abs(g["inj_out"]["kernel"]).sum()))  # 0.0 0.0

# C5: hash slots ignore context (vocab_size must cover the ids)
from navi.model import Navi; from navi.config import ModelConfig
mh = Navi(ModelConfig(memory_every=2, vocab_size=1299), MemoryConfig(hash_slots=True), return_aux=True)
ph = mh.init({"params": jax.random.PRNGKey(0)}, jnp.zeros((1, 8), jnp.int32))
_, aux, _ = mh.apply(ph, jnp.array([[5, 30, 40, 50, 7, 31, 40, 50]]))
s = aux["mem_0"][0]; print(s[2].tolist(), s[6].tolist())                        # identical
```
C3 (read scores ≠ raw scores under E9), C13 (NaN loss), C14 (roundtrip TypeError) and C15 (parameter counts) were checked the same way. The numbers are quoted in their sections.
