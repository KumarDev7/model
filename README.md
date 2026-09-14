# FP8 (E4M3) Multiply–Accumulate — pure transistor-level, GEMM-optimized

A single, fully custom **FP8 E4M3 multiply–accumulate (MAC)** unit built
**entirely at the MOSFET level** and optimized for AI/ML matrix-multiply (GEMM)
workloads, with the **lowest practical transistor count** while remaining
functionally correct and energy-efficient.

Everything — datapath, control, primitives — is expanded to individual NMOS/PMOS
devices with explicit `W/L`. There is **no** behavioural Verilog/VHDL, **no**
standard cells, and **no** gate primitive that is not itself built from
transistors. The design is constructed with **PySpice**, dumped as a pure SPICE
transistor netlist, and simulated in **ngspice**.

```
Result = A · B + C          A, B : FP8 E4M3        C, Result : fixed-point accumulator
```

* **Default (minimum-area) configuration: 2216 MOSFETs.**
* Verified two independent ways: a built-in **switch-level (transistor) simulator**
  against a bit-exact golden model, **and** real **ngspice** DC operating-point +
  transient analog simulation.

---

## 1. Quick start

```bash
pip install PySpice numpy          # PySpice pulls in numpy/scipy/matplotlib
sudo apt-get install -y ngspice    # or: brew install ngspice

python3 fp8_mac_spice.py                 # build + verify + ngspice + reports
python3 fp8_mac_spice.py --no-spice      # skip analog sim (switch-level only)
python3 fp8_mac_spice.py --exhaustive    # also verify ALL 65536 A·B pairs (slow)
python3 fp8_mac_spice.py --dump-netlist fp8_mac.sp
```

The script **degrades gracefully**: with PySpice/ngspice absent it still builds
the netlist, counts transistors and runs the switch-level verification.

Files:

| file | contents |
|------|----------|
| `fp8_mac_spice.py` | the complete self-contained implementation (build, sim, verify, report) |
| `fp8_mac.sp`       | dumped pure-transistor SPICE netlist (generated) |
| `VERIFICATION.txt` | captured end-to-end run log (transistor report, verification, ngspice, GEMM) |

---

## 2. Format & functional specification

**FP8 E4M3**: `1` sign · `4` exponent (bias 7) · `3` mantissa (implicit leading 1).

| case | handling | why (transistor rationale) |
|------|----------|----------------------------|
| ±0, `exp==0` | value 0 | zero significand falls out for free |
| subnormals (`exp==0`) | **flush-to-zero** | deletes the whole subnormal-align datapath |
| NaN (`S.1111.111`) | **flush-to-zero** | E4M3's single NaN; cheaper than propagation, valid for GEMM |
| ±Inf | **N/A** | E4M3 has no infinities (per OCP) — nothing to encode |
| overflow | **saturate** to accumulator max | one OR row, no Inf logic |
| rounding | **truncation** (round-toward-zero) | no sticky-OR tree, no rounding incrementer |

**Accumulator.** `C` and the output are a **signed fixed-point accumulator**, not
FP8. This is exactly how real GEMM / systolic MAC arrays accumulate, and it is the
single biggest transistor saving because it **removes the leading-one detector,
the normalization shifter and the rounder** from the replicated processing
element (PE). Width/precision are parameters (`set_accumulator(W_ACC, KL)`):

| config | width | range | LSB | mean GEMM error (K=16) | transistors |
|--------|-------|-------|-----|------------------------|-------------|
| **default (min-area)** | 16-bit Q9.6 | ±512 | 2⁻⁶ ≈ 0.0156 | ≈ 1.1 LSB | **2216** |
| balanced | 20-bit | ±2048 | 2⁻⁸ ≈ 0.0039 | ≈ 0.5 LSB | 2574 |
| full-range | 24-bit | ±32768 | 2⁻⁸ | ≈ 0.5 LSB | 2846 |

(The exact numbers are reproduced by the `[tradeoff]` table the script prints.)

---

## 3. Micro-architecture (combinational core)

```
 A(8) ─┬─ decode ─► sA, SA[4], eA[4]           (FTZ subnormal & NaN)
 B(8) ─┴─ decode ─► sB, SB[4], eB[4]
        sP = sA xor sB                          (1 XOR)
        SP[8] = SA · SB                          4x4 Dadda array multiplier
        eSum[5] = eA + eB                        5-bit ripple adder
        Lc = eSum − LC_SUB ; drop = (eSum<LC_SUB) ; sat = (eSum≥SAT_THRESH)
        T = SP << Lc                             log barrel shifter (TG muxes)
        M = gate( T[OFF : OFF+MAG] )             force 0 on drop/zero, all-1s on sat
        OUT = C ± M                              two's-complement saturating adder
```

*Alignment maths.* A product equals `SP · 2^(eA+eB−20)`. Placing it at accumulator
LSB `2^KL` is a left shift by `Lc = eSum − (14+KL)`; bits below the LSB are simply
dropped (correct under truncation — the sticky bit can never round up). Products
too small to reach the LSB assert `drop`; products past the top assert `sat`.
Both comparisons are **carry-outs of adders already in the datapath**, so the
comparators are nearly free.

---

## 4. Transistor-count report (default 16-bit config)

| block | NMOS | PMOS | total | what it is / why it's small |
|-------|-----:|-----:|------:|-----------------------------|
| accumulate | 310 | 310 | **620** | 16-bit two's-complement **saturating** adder; conditional negate = 1 XOR row + carry-in (no separate subtractor) |
| multiplier | 258 | 258 | **516** | 4×4 unsigned significand multiply; Dadda column compression with 8T half-/24T full-adders (vs 28T mirror array) |
| shifter | 172 | 172 | **344** | logarithmic barrel shifter (stages 1,2,4,8) from **4T transmission-gate** 2:1 muxes — no decoder, no dual-rail |
| shift_ctrl | 133 | 133 | **266** | exponent compare/subtract; `drop` and `saturate` reuse adder carry-outs |
| shift_gate | 98 | 98 | **196** | forces addend to 0 (underflow/zero operand) or all-ones (overflow): 1 AND + 1 OR per bit |
| decode | 73 | 73 | **146** | significand gating + FTZ of subnormals & NaN (removes those datapaths) |
| exp_add | 60 | 60 | **120** | single 5-bit ripple adder (bias folded into a constant subtract) |
| sign | 4 | 4 | **8** | one XOR gate |
| **TOTAL** | | | **2216** | |

Estimated placed area (λ design rules, dense custom): **≈ 577 µm² @ 65 nm**,
**≈ 107 µm² @ 28 nm**. Active gate area Σ(W·L) ≈ 39.9 µm² (60 nm draw). Supply
0.8–1.0 V (default 1.0 V).

### Primitive cell library (each fully expanded to transistors)

| cell | transistors | style |
|------|-------------|-------|
| INV | 2 | static CMOS |
| NAND2 / NOR2 | 4 | static CMOS |
| NAND3 / NOR4 | 6 / 8 | static CMOS |
| transmission gate | 2 | N+P pass pair (full-swing) |
| 2:1 MUX | 4 (+shared inv) | transmission-gate |
| XOR2 | 4 (+shared inv) | transmission-gate |
| half adder | ~10 | TG XOR + AND |
| **full adder** | **24** | TG XOR/XNOR sum + TG-mux carry, **restored** carry |

The full adder is **24T** (restored carry keeps drive across a ripple chain),
chosen over the 28T mirror adder for area and over unreliable 10T pass-only
designs for correctness. Transmission gates use parallel N+P so every node
swings rail-to-rail — no threshold-drop, which is why the analog ngspice runs
resolve to correct logic levels.

---

## 5. Verification

**(a) Switch-level (transistor) simulation.** A built-in solver treats every
MOSFET as a gate-controlled switch, unions nodes joined by conducting devices,
and assigns each component the value of the strong source (VDD/GND/driven input)
it touches — iterating to a fixed point. This derives logic **purely from the
transistor topology**, not from any parallel behavioural spec, and is checked
against the bit-exact `mac_hw()` golden model over structured corner vectors
(zero, subnormal, normal, max, NaN; drop / in-range / saturate regions),
random vectors, and (optionally) the **exhaustive** 65 536 `A·B` space.
Result: **0 failures**.

**(b) Real analog ngspice.** The identical netlist is built as a PySpice
`Circuit` and driven through `ngspice -b` (PySpice 1.5's shared-lib bridge is
incompatible with ngspice 42, so the emitted deck is run through the binary).
A **DC operating-point** of the full ~2200-transistor network is solved per
vector and thresholded to logic; edge cases (subnormal→FTZ, NaN→FTZ,
448×448→saturate, min×min→underflow) all match the golden model.

**(c) Transient — power / delay.** An `ngspice` `.tran` toggles a full input
vector; the supply current is integrated for **energy/op**, the settled current
gives **static power**, and the last output crossing gives a combinational
**settling delay** estimate.

Representative measured figures (generic 1.0 V model, see `VERIFICATION.txt`):
**energy ≈ 0.1 pJ/op**, **static power ≈ 0.16 µW**, **settle ≈ 0.1 ns**. These
scale with the model card — swap in a foundry BSIM4 PDK for sign-off numbers.

---

## 6. GEMM accuracy

Dot products of normalized FP8 operands (values ≈ [−2, 2], the post-normalization
inference regime) are accumulated through the bit-exact MAC and compared to ideal
arithmetic. Mean absolute error is **≈ 1 LSB** for the 16-bit config and **≈ 0.5
LSB** for 20/24-bit — i.e. the error is just the truncation quantization, with no
drift, exactly what a fixed-point GEMM accumulator should deliver. (Relative error
is only meaningful for dot products whose true sum is bounded away from zero;
cancellation-to-≈0 cases are excluded from the relative metric.)

---

## 7. Models & process

Generic low-voltage CMOS `.model` cards (`LEVEL=1`, `VTO=±0.35`, sized for 65/28 nm
digital behaviour) are used so ngspice stays robust and fast; they are clearly
marked **educational** and are a drop-in replacement for a foundry **BSIM4** card
(`.include 65nm_bulk.pm`) — the `W/L` are already on every `M` line. Device sizes:
`Wn = 200 nm`, `Wp = 400 nm`, `L = 60 nm` (min-length; PMOS 2× for β-matching).

---

## 8. Why this beats a textbook / standard-cell FP8 MAC

A conventional standard-cell FP8 **FMA** with full IEEE features (subnormals,
round-to-nearest-even, infinities/NaN propagation, a wide normalize+round path)
and pipeline flip-flops runs to **several thousand** transistors per PE. This
design reaches **2216** by:

1. **Fixed-point accumulate** (no LOD / normalize / rounder in the PE) — the
   biggest saving and the correct choice for GEMM.
2. **Truncation** — no sticky tree, no rounding incrementer; out-of-window bits
   are just discarded.
3. **Flush-to-zero of subnormals *and* NaN** — deletes those datapaths entirely.
4. **No Infinity logic** (E4M3 has none); overflow saturates via one OR row.
5. **Transmission-gate muxes / shifters** (4T) instead of static or dual-rail.
6. **24T restored TG full adder**; 8T half adders where a column has two bits.
7. **Shared arithmetic** — `eA+eB`, the `−LC_SUB`/`−SAT_THRESH` constants,
   `drop` (an inverted carry) and `saturate` (a spare carry-out) all reuse the
   same ripple adders.
8. **Conditional two's-complement negate** = one XOR row + carry-in.
9. **Flop-free combinational PE** — a systolic array supplies its own registers,
   so the replicated cell carries none.

---

## 9. Further transistor reduction (roadmap)

* **Serialize the multiplier** (bit-serial / 2-cycle 4×2) to roughly halve the
  516-transistor array at a 2× latency cost — attractive when the PE is
  replicated N² times.
* **Carry-select or carry-skip** only where the accumulate path is timing-critical;
  otherwise keep the dense ripple.
* **Drop output saturation** (≈ 60–90 T) if the surrounding tile guarantees the
  accumulator never overflows.
* **Dynamic/precharge** multiplier partial-product AND array where leakage is
  tolerable (domino), trading static robustness for ~30–40 % fewer devices in
  that block.
* **Share one physical multiplier/adder across time** in a multi-cycle PE to cut
  peak device count further (resource sharing).

---

## 10. Reproducing every number

`python3 fp8_mac_spice.py` prints, in order: the transistor report + area, the
switch-level verification result, the ngspice DC functional table, the transient
energy/power/delay, the per-K GEMM accuracy, and the accumulator-width tradeoff
sweep — all of which are captured in `VERIFICATION.txt`.
