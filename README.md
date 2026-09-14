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

* **Default (minimum-area) configuration: 1454 MOSFETs** (down from 2216 after an
  aggressive constant-folding / hardware-sharing optimization pass — see §9).
* Verified two independent ways: a built-in **switch-level (transistor) simulator**
  against a bit-exact golden model, **and** real analog simulation driven **by
  PySpice's own ngspice binding** (DC operating-point) plus an ngspice transient
  for power/delay.

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
| **default (min-area)** | 16-bit Q9.6 | ±512 | 2⁻⁶ ≈ 0.0156 | ≈ 1.1 LSB | **1454** |
| balanced | 20-bit | ±2048 | 2⁻⁸ ≈ 0.0039 | ≈ 0.5 LSB | 1720 |
| full-range | 24-bit | ±32768 | 2⁻⁸ | ≈ 0.5 LSB | 1906 |

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

| block | NMOS | PMOS | total | before | what it is / why it's small |
|-------|-----:|-----:|------:|------:|-----------------------------|
| accumulate | 271 | 271 | **542** | 620 | 16-bit two's-complement **saturating** adder (20T FAs); negate = 1 XOR row + carry-in; overflow = XOR of top two carries |
| multiplier | 149 | 149 | **298** | 516 | implicit leading-1 is a **constant** → the 4×4 folds to a 3×3 (9 ANDs); Dadda compression |
| shifter | 104 | 104 | **208** | 344 | **pruned** log barrel shifter (4T TG muxes): only occupied bits + last-stage read window are built |
| shift_gate | 77 | 77 | **154** | 196 | gates the 8-bit product to 0 (underflow/zero) at the shifter input, ORs saturation into M |
| decode | 53 | 53 | **106** | 146 | FTZ subnormals & NaN + **constant** implicit-1 → no significand-gating gates |
| exp_add | 37 | 37 | **74** | 120 | single 5-bit ripple adder (top bits fold away) |
| shift_ctrl | 32 | 32 | **64** | 266 | `eSum ± constant` folds half its full-adders to half-adders; `drop`/`sat` are carry-outs |
| sign | 4 | 4 | **8** | 8 | one XOR gate |
| **TOTAL** | | | **1454** | 2216 | **−34 %** |

Estimated placed area (λ design rules, dense custom): **≈ 380 µm² @ 65 nm**,
**≈ 70 µm² @ 28 nm**. Supply 0.8–1.0 V (default 1.0 V). (Exact figures are
printed by the script.)

### Primitive cell library (each fully expanded to transistors)

| cell | transistors | style |
|------|-------------|-------|
| INV | 2 | static CMOS |
| NAND2 / NOR2 | 4 | static CMOS |
| NAND3 / NOR4 | 6 / 8 | static CMOS |
| transmission gate | 2 | N+P pass pair (full-swing) |
| 2:1 MUX | 4 (+shared inv) | transmission-gate |
| XOR2 / XNOR2 | 4 (+shared inv) | transmission-gate |
| half adder | ~10 | TG XOR + AND |
| **full adder** | **20** | TG-XOR sum + TG-mux carry |

**Every cell constant-folds.** An input tied to VDD/GND collapses the cell to a
wire / constant / inverter, so no transistor is ever spent driving a rail. That
single rule is what removes the "idle" transistors: constant operands (the
implicit leading 1s, the bias/threshold constants, zero-padding) cost nothing,
which is why the 4×4 multiplier is really a 3×3 and `shift_ctrl` collapsed from
266→64. The full adder is **20T** (no per-bit carry restore); a restoring buffer
is inserted only every 4th bit of a long ripple. Transmission gates use parallel
N+P so every node swings rail-to-rail — no threshold drop, which is why the
analog runs resolve to correct logic levels.

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

**(b) Real analog simulation, driven by PySpice.** The identical netlist is built
as a PySpice `Circuit` and simulated through **PySpice's own `NgSpiceShared`
binding** (see §5.1 below on the ngspice-42 compatibility). A **DC operating-point**
of the full 1454-transistor network is solved per vector and thresholded to logic;
edge cases (subnormal→FTZ, NaN→FTZ, 448×448→saturate, min×min→underflow) all match
the golden model: `DC functional: ALL OK`.

**(c) Transient — power / delay.** An `ngspice` `.tran` toggles a full input
vector; the supply current is integrated for **energy/op**, the settled current
gives **static power**, and the last output crossing gives a combinational
**settling delay** estimate.

Representative measured figures (generic 1.0 V model, see `VERIFICATION.txt`):
**energy ≈ 74 fJ/op**, **static power ≈ 0.10 µW**, **settle ≈ 79 ps** (all lower
than the pre-optimization design, thanks to the smaller device count). These
scale with the model card — swap in a foundry BSIM4 PDK for sign-off numbers.

### 5.1 Why PySpice — and the ngspice-42 shim

PySpice **is** used, on both ends: it constructs the `Circuit` object and emits
`fp8_mac.sp`, and its `NgSpiceShared` binding runs the DC verification above.
PySpice 1.5 ships a hard-coded list of "supported" ngspice versions that excludes
v42 (the current Ubuntu build) and also misreads v42's benign `run` return status,
so the stock `circuit.simulator().operating_point()` raises. Two small,
well-contained shims in `_patch_pyspice()` fix it: (1) set
`NgSpiceShared.NGSPICE_SUPPORTED_VERSION = 42`, and (2) wrap `exec_command` to
swallow the spurious status. With those, PySpice's native simulator runs the
whole MAC and returns node voltages normally. The `ngspice -b` batch path is kept
as an automatic fallback (and for the transient), so the script works whether or
not PySpice is importable.

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
design reaches **1454** by:

1. **Fixed-point accumulate** (no LOD / normalize / rounder in the PE) — the
   biggest architectural saving and the correct choice for GEMM.
2. **Truncation** — no sticky tree, no rounding incrementer; out-of-window bits
   are just discarded.
3. **Flush-to-zero of subnormals *and* NaN** — deletes those datapaths entirely.
4. **No Infinity logic** (E4M3 has none); overflow saturates via a carry XOR.
5. **Transmission-gate muxes / shifters** (4T) instead of static or dual-rail.
6. **20T TG full adder** with restoration only every 4th ripple bit.

The optimization pass then removed a further **762 transistors (2216 → 1454, −34 %)**
by attacking idle transistors directly:

7. **Constant folding in every cell** — a rail-tied input collapses the gate, so
   no transistor drives a constant. This makes the **4×4 multiplier a 3×3** (the
   implicit leading 1 is a constant VDD → its partial products are wires: 516→298),
   and folds the **exponent/shift-control** adders that add `eSum` to constants
   (`shift_ctrl` 266→64, `exp_add` 120→74, `decode` 146→106).
8. **Pruned barrel shifter** — each stage builds muxes only for indices that can
   be occupied, and the last stage only the read window (344→208).
9. **Product-zero gated at the 8-bit shifter input**, not the 15–23-bit magnitude.
10. **Overflow = XOR of the top two ripple carries** — no 17-bit sign-extended
    add just to detect saturation (accumulate 620→542).
11. **Shared arithmetic / free comparators** — `drop` is an inverted carry,
    `saturate` a spare carry-out; the conditional negate is one XOR row + carry-in.
12. **Flop-free combinational PE** — a systolic array supplies its own registers.

---

## 9. Further transistor reduction (roadmap)

* **Serialize the multiplier** (bit-serial / 2-cycle 4×2) to roughly halve the
  ~300-transistor array at a 2× latency cost — attractive when the PE is
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
