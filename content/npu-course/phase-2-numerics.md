---
title: "Phase 2 — Fixed-Point Numerics"
description: "Q-formats and the requantization pipeline — the place where FPGA ML projects silently produce garbage."
tags:
  - npu
  - hardware
---

# Phase 2 — Numerics

**2 weeks** · Part of [[index|Zero to NPU]] · Prev: [[phase-1-baseline-llm]] · Next: [[phase-3-mac-engine]]

> [!info] What this phase is for
> Honestly, this is the least glamorous phase in the course, and the one most likely to quietly ruin your project. FPGAs have no float — every number in your accelerator is an integer wearing a fraction as a costume, and the costume only works if all your hardware agrees on exactly where the decimal point sits. Phase 1 gave you a working int8 model in C; this phase proves, bit for bit, that the arithmetic you build in Verilog in [[phase-3-mac-engine]] produces the *exact same answers*. The failure mode is uniquely nasty: get the rounding direction wrong on one right-shift and your accelerator doesn't crash or error — it quietly produces worse and worse outputs with nothing to point you at the cause. Two weeks feels like a long time to spend shifting integers, but almost everyone who rushes this phase pays for it later, in a debugging session that costs far more. Go slowly — this is the foundation everything else stands on.

## Why fixed point at all

Every multiply your accelerator does — weight times activation, millions of times per inference — has to happen in real silicon, and silicon isn't free. A float multiplier is a genuinely complicated circuit: align exponents, multiply wide mantissas, renormalize, handle NaN and infinity. An int8 multiplier, by contrast, is small enough that a single FPGA [[concepts#Fixed point|DSP]] slice does it in one shot. Your board has roughly 60 DSPs total, and useful throughput needs many multiplies per cycle. Spend those DSPs on float hardware and you'll fit a handful of multipliers; spend them on int8 and you'll fit an array. This phase is the reason your accelerator fits in that budget at all.

"Fixed point" is less mysterious than it sounds. Instead of letting the decimal point float around (as IEEE float does, via the exponent), you *fix* it at a known position, once, up front — and from then on you use ordinary integer arithmetic. The hardware never knows a fraction is involved. It only exists in your head and in the software that decodes integers back into real values for a human to read.

A useful analogy: a price of $19.99 is often stored as the integer 1999 — cents instead of dollars. Nobody builds special "cents" hardware; you just agree, once, that the last two digits mean cents, and every addition and comparison behaves correctly *as long as everyone agrees where the point is*. Fixed-point binary arithmetic is this same trick, generalized to base 2. The entire discipline of this phase is making sure you and your future Verilog module agree, with zero ambiguity, on where that point sits at every stage.

## Q-format, properly explained

The standard notation for "where the point sits" is **[[concepts#Q-format|Qm.n]]**: `m` integer bits, `n` fractional bits, plus one sign bit, packed into an ordinary two's-complement register. Total width is `1 + m + n` bits. The bits look no different from a normal signed integer — Q-format is purely a *convention for interpretation* that you must apply consistently.

Take **Q1.15** in a 16-bit signed word (1 sign + 1 integer + 15 fractional bits — common for audio and neural-net activations). Its range is `[-1.0, 0.999969...]`, resolution `2^-15 ≈ 0.0000305`. To read a Q1.15 integer as a real number, divide by `2^15 = 32768`. To encode a real number into Q1.15, multiply by `32768` and round.

Here's the part that trips people up: **multiplication moves the point.** Multiply two Q1.15 values (two 16-bit integers) and you get a 32-bit product that is not Q1.15 anymore — it's **Q2.30** (integer bits add: 1+1=2; fractional bits add: 15+15=30). The point has physically moved. This is the single most important fact in this section, so let's walk it through with real numbers.

Encode `0.5` and `0.25` in Q1.15:
- `0.5 × 32768 = 16384`
- `0.25 × 32768 = 8192`

Multiply the raw integers, exactly as your hardware will:
- `16384 × 8192 = 134,217,728`

Now interpret that 32-bit result as **Q2.30** — divide by `2^30 = 1,073,741,824`:
- `134,217,728 / 1,073,741,824 = 0.125`

And `0.5 × 0.25 = 0.125` — it checks out, but only because you tracked the output format as Q2.30, not Q1.15. Treat that same 32-bit product as if it were still Q1.15 (dividing by `2^15` instead of `2^30`) and you'd decode it as `4096.0` — wildly wrong, even though every bit was correct. **The bits are never wrong. Your bookkeeping about the format is what goes wrong**, and it goes wrong silently.

Addition has the opposite requirement: both operands must already be in the *same* Q format. You can't add a Q1.15 value to a Q2.30 value directly, any more than you can add 5 cents to 5 dollars without agreeing on units first — you'd have to shift one to line up the binary points.

| Format | Total bits | Range | Resolution | Typical use |
|---|---|---|---|---|
| Q0.7 | 8 | [-1, 0.992] | 2^-7 | normalized int8 activation |
| Q1.15 | 16 | [-1, 0.99997] | 2^-15 | audio, normalized activations |
| Q1.31 | 32 | [-1, ~1.0] | 2^-31 | high-precision audio |
| Q3.13 | 16 | [-4, 3.9999] | 2^-13 | small dynamic range with headroom |
| Q8.8 | 16 | [-128, 127.996] | 2^-8 | coarse values needing bigger integer range |
| Q2.30 | 32 | [-2, ~2.0] | 2^-30 | product of two Q1.15 values |

## Quantization: mapping floats onto int8

[[concepts#Quantization|Quantization]] is the separate step of taking the *floating-point* weights and activations your Phase 1 model already has and mapping them onto int8. It uses the same "agree where the point is" idea as Q-format, but expressed through an explicit **scale** factor instead of a fixed bit position, which turns out to be more convenient once floats are involved.

For symmetric quantization (the scheme this course uses, with zero-point fixed at 0), the scale is:

```
scale = max(|W|) / 127
```

taken over whatever set of values you're quantizing together. To quantize a float `x`, you compute `q = round(x / scale)` and clamp the result into `[-128, 127]` (a [[concepts#Saturation|saturating]] clamp — more on that below). To recover an approximation of the original float, you compute `x ≈ q × scale`. That `≈` is doing real work: quantization is lossy, and the whole point of this phase is making sure the *only* place you lose information is this deliberate, controlled rounding — not an accidental mismatch between your C model and your hardware.

Weights are quantized **per-channel**: every output channel (roughly, every row of a weight matrix) gets its *own* scale, computed from just that channel's values. Activations are quantized **per-tensor**: one scale for the whole tensor. The reason is dynamic range — different output channels often have very different weight magnitudes (one might peak at 0.02, another at 1.5), and sharing one scale would waste most of int8's 256 values on the small-magnitude channels. Per-channel scales let each channel use its full 8-bit range. Activations don't need that structure, and per-tensor keeps the runtime MAC hardware simple, since every term in a row shares one activation scale.

Symmetric quantization (zero-point locked to 0) matters for hardware, not just convenience. Asymmetric quantization adds a zero-point offset term that has to be carried through every multiply — a cross-term your MAC array would otherwise not need, meaning extra adders, extra bit width, extra chances to get it wrong. Symmetric quantization keeps the MAC array exactly as simple as "multiply two signed int8s, accumulate in int32" — the hardware you build in Phase 3.

## The requantization pipeline, step by step

Here is the chain you'll implement twice, in C and in Verilog — the single most important piece of arithmetic in this course:

```
int32 accumulator  →  × M0 (fixed-point multiplier)  →  >>n (rounding shift)  →  saturate to int8
```

This is the [[concepts#Requantization|requantization]] step that runs between every matmul and the next layer, following the same approach as gemmlowp and the Jacob et al. 2018 integer-arithmetic-only-inference paper. Let's understand *why* it looks like this before touching code.

Your MAC array accumulates `int8 × int8` products into an `int32` accumulator, whose true value in real units is `acc × scale_w × scale_a`. To get back to your output tensor's int8 representation you divide by `scale_out`, so the correction factor you actually need is:

```
real_multiplier = (scale_w × scale_a) / scale_out
```

That `real_multiplier` is a **float**, almost always less than 1 (accumulators tend to be "too big" relative to the output range). You can't compute with a float on this hardware, so instead you represent it as an integer `M0` times a power of two:

```
real_multiplier ≈ M0 × 2^-n
```

where `M0` is a 32-bit integer (conventionally normalized so it uses close to the full 32 bits of precision, i.e. `M0` is roughly in `[2^30, 2^31)`) and `n` is a right-shift amount. Computing `(M0, n)` from a float in C is a normalization loop very much like the mantissa/exponent split `frexp()` does for IEEE floats:

```c
// Decompose a positive float multiplier < 1 into M0 * 2^-n,
// with M0 normalized into roughly [2^30, 2^31).
void quantize_multiplier(double real_multiplier, int32_t *m0, int *n) {
    int shift = 0;
    // frexp gives us frac in [0.5, 1.0) and the power-of-two exponent
    double frac = frexp(real_multiplier, &shift); // real_multiplier = frac * 2^shift
    int64_t q = (int64_t)round(frac * (1LL << 31)); // scale frac up into a 32-bit integer
    if (q == (1LL << 31)) { q /= 2; shift++; }        // handle the round-up-to-2^31 edge case
    *m0 = (int32_t)q;
    *n  = 31 - shift; // total right-shift needed after the 32-bit multiply
}
```

Once you have `M0` and `n`, applying the pipeline is: multiply `acc × M0` (needs a 64-bit intermediate or you silently truncate), round-and-shift right by `n` bits, then saturate into `[-128, 127]`.

The rounding step is where almost everyone gets bitten. It's tempting to write `result = prod >> n` and move on — "shift right by n" *is* division by `2^n`, after all. But a plain arithmetic right shift (`>>>` in Verilog, `>>` on a signed value in C) rounds toward negative infinity, not toward zero, and not toward the nearest value either. Concretely: `-3 >> 1` evaluates to **-2**, not **-1**. If your mental model was "shifting is truncating division, so -3/2 truncates to -1," you're already wrong — and you won't find out from a compiler warning, you'll find out from a model that mysteriously drifts worse the deeper into the network you look.

The fix is **round-half-away-from-zero**, applied identically on both sides of the C/Verilog boundary: before shifting, add a rounding bias equal to "half" of what's about to be discarded (`1 << (n-1)`) to the *magnitude* of the product, then reapply the sign. Do this the same way in both languages — the identical sequence of operations, not just "morally similar" — and they will agree bit-for-bit. Exercise 2.2 builds exactly that, and proves it with a million random vectors.

## 2.1 Q-format fluency

> [!abstract] Goal
> Build a small C toolkit that converts between real numbers and Q-format integers, and reports what format a multiply produces — until you can do the Q1.15×Q1.15 example above on a whiteboard without hesitating.

**What's going on:** The Q-format math above is something you followed on the page. This exercise makes you the one doing the encoding, decoding, and format bookkeeping, on numbers you pick yourself, until it stops feeling like a trick and starts feeling like arithmetic.

**Steps:**
1. Write `float_to_q(x, n)`: multiply by `2^n`, round to nearest (ties away from zero), return the integer.
2. Write `q_to_float(q, n)`: divide by `2^n`, return the double.
3. Write `print_product_format(m1, n1, m2, n2)`: given two Qm.n formats, print the Qm.n format of their product (integer bits add, fractional bits add) and the total bit width needed to hold it without truncation.
4. Reproduce the `0.5 × 0.25` worked example from above in code and confirm you get `0.125` back out.
5. Pick three more pairs of numbers yourself, run them through, and check the decoded result against a calculator.

```c
#include <stdio.h>
#include <stdint.h>
#include <math.h>

// Encode a real number as a Qm.n integer (only n matters for the math;
// m only affects what range is "valid," which we don't enforce here).
int32_t float_to_q(double x, int n) {
    double scaled = x * (double)(1LL << n);
    return (int32_t)(scaled >= 0 ? scaled + 0.5 : scaled - 0.5); // round, ties away from 0
}

// Decode a Qm.n integer back to a real number.
double q_to_float(int32_t q, int n) {
    return (double)q / (double)(1LL << n);
}

// Report the format (and required width) of a Qm1.n1 x Qm2.n2 product.
void print_product_format(int m1, int n1, int m2, int n2) {
    int m = m1 + m2, n = n1 + n2;
    printf("Q%d.%d x Q%d.%d -> Q%d.%d (%d-bit product, no truncation)\n",
           m1, n1, m2, n2, m, n, 1 + m + n);
}

int main(void) {
    int32_t a = float_to_q(0.5, 15);   // expect 16384
    int32_t b = float_to_q(0.25, 15);  // expect 8192
    int64_t prod = (int64_t)a * (int64_t)b; // expect 134217728, a Q2.30 value
    printf("a=%d b=%d prod=%lld -> %.4f\n", a, b, (long long)prod, q_to_float((int32_t)prod, 30));

    print_product_format(1, 15, 1, 15); // Q1.15 x Q1.15 -> Q2.30
    print_product_format(3, 13, 1, 15); // Q3.13 x Q1.15 -> Q4.28
    return 0;
}
```

| Value | Format | Stored integer | Decoded back |
|---|---|---|---|
| 0.5 | Q1.15 | 16384 | 16384 / 32768 = 0.5 |
| 0.25 | Q1.15 | 8192 | 8192 / 32768 = 0.25 |
| 0.5 × 0.25 | Q2.30 (product) | 134,217,728 | 134217728 / 2^30 = 0.125 |
| -0.75 | Q1.15 | -24576 | -24576 / 32768 = -0.75 |

> [!success] Done when
> You can hand-derive the Qm.n output format of any Qm1.n1 × Qm2.n2 multiply in your head, and your code confirms the worked example above returns exactly `0.125`.

> [!question]- Mismatches between C and Verilog?
> This exercise is pure C, but the habits you build here are what save you in 2.2 — so before moving on, check:
> 1. Are you always widening to a 64-bit intermediate before multiplying two 32-bit-range values?
> 2. Does your rounding in `float_to_q` handle negative numbers the same way it handles positive ones (ties away from zero, both directions)?
> 3. Did you actually test a negative input, not just the positive worked example?
> 4. Does `print_product_format` match what you'd get if you multiplied the raw integers by hand?
> 5. Do you know, for every number in your test set, exactly which Q format it's supposed to be in at every line of your code?

> [!danger] Gotcha
> It is very easy to write `float_to_q` correctly for positive numbers and subtly wrong for negative ones, because `(int)(scaled + 0.5)` silently does the wrong thing when `scaled` is negative (it rounds toward zero instead of away from it). Test negative values explicitly — this is a preview of the much bigger version of this bug waiting in 2.2.

## 2.2 The requantization pipeline, in C and in Verilog

> [!abstract] Goal
> Implement the full `int32 → M0 multiply → rounding shift → saturate` chain in both C and Verilog, and prove they agree bit-exactly across a million random inputs and 100 different `(M0, n)` pairs.

**What's going on:** This is the exercise the whole phase has been building toward. Everything downstream in Phase 3 assumes this pipeline is correct and that your Verilog and C model produce identical outputs. If they silently disagree here, nothing you build afterward will be debuggable — the bug will look like it's somewhere else entirely.

**Steps:**
1. Implement `requantize()` in C exactly as below — 64-bit intermediate, round-half-away-from-zero, saturate.
2. Implement the equivalent `requantize` module in Verilog, matching the C rounding logic operation-for-operation, not just "in spirit."
3. Write a Verilator testbench that drives both with the same random `(acc, M0, n)` triples and asserts the outputs match.
4. Generate 100 different `(M0, n)` pairs (mix of small and large `n`, positive and negative `acc`, edge values like `acc = INT32_MIN`) and run 1,000,000 random `acc` values through each — or 1,000,000 total across all 100 pairs, whichever you find easier to script — and confirm zero mismatches.

**(a) C reference implementation:**

```c
#include <stdint.h>

// int32 acc -> int8, matching int32 -> xM0 -> >>n -> saturate.
int8_t requantize(int32_t acc, int32_t m0, int n) {
    // Full-precision product. acc and m0 are each up to 32 bits, so
    // the product needs up to 64 bits -- a 32-bit multiply here would
    // silently truncate and every downstream bit would be wrong.
    int64_t prod = (int64_t)acc * (int64_t)m0;

    int64_t result;
    if (n == 0) {
        // Nothing to shift, so nothing to round.
        result = prod;
    } else {
        // "half" is 0.5 in the units about to be shifted away --
        // i.e. the value of bit (n-1) of the product.
        int64_t half = (int64_t)1 << (n - 1);

        if (prod >= 0) {
            // Standard round-to-nearest: add half, then shift.
            result = (prod + half) >> n;
        } else {
            // Negative case: do NOT shift prod directly. A plain
            // arithmetic right shift rounds toward -infinity, which
            // is NOT the same as round-half-away-from-zero. Instead,
            // mirror the positive-case math on the magnitude, then
            // reapply the sign, so positive and negative accumulators
            // round symmetrically.
            result = -(((-prod) + half) >> n);
        }
    }

    // Saturate into int8 range. Without this, an out-of-range result
    // wraps around instead of clamping, which is a second, unrelated
    // way to silently corrupt output.
    if (result > 127)  result = 127;
    if (result < -128) result = -128;
    return (int8_t)result;
}
```

**(b) Equivalent synthesizable Verilog:**

```verilog
module requantize (
    input  wire signed [31:0] acc,   // MAC accumulator
    input  wire signed [31:0] m0,    // fixed-point multiplier
    input  wire [5:0]         n,     // right-shift amount, 0-31
    output wire signed [7:0]  q_out  // requantized int8 result
);
    // Full-precision product -- the Verilog equivalent of C's int64_t prod.
    wire signed [63:0] prod = acc * m0;

    // 0.5 in the units about to be shifted away. Zero when n == 0,
    // exactly mirroring the C "if (n == 0)" special case.
    wire signed [63:0] half = (n == 0) ? 64'sd0 : (64'sd1 <<< (n - 1));

    // Round the magnitude, then reapply the sign -- same mirror trick
    // as the C code's negative branch, applied uniformly here so one
    // code path covers both signs.
    wire                 sign      = prod[63];
    wire signed [63:0]   abs_prod  = sign ? -prod : prod;
    wire signed [63:0]   rounded   = (n == 0) ? abs_prod : ((abs_prod + half) >>> n);
    wire signed [63:0]   result    = sign ? -rounded : rounded;

    // Saturate to int8 [-128, 127].
    assign q_out = (result >  64'sd127)  ?  8'sd127  :
                   (result < -64'sd128)  ? -8'sd128  :
                   result[7:0];
endmodule
```

**(c) Verilator testbench sketch:**

```cpp
// tb_requantize.cpp -- drives both models with identical random vectors.
#include "Vrequantize.h"
#include "verilated.h"
#include <random>
#include <cstdint>
#include <cassert>
#include <cstdio>

extern "C" int8_t requantize(int32_t acc, int32_t m0, int n); // linked from requantize.c

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vrequantize* dut = new Vrequantize;

    std::mt19937_64 rng(0xC0FFEE);
    std::uniform_int_distribution<int32_t> acc_dist(INT32_MIN, INT32_MAX);
    std::uniform_int_distribution<int32_t> m0_dist(1, INT32_MAX);
    std::uniform_int_distribution<int>     n_dist(0, 31);

    const int NUM_PAIRS = 100, TRIALS_PER_PAIR = 10000; // 1,000,000 total
    for (int pair = 0; pair < NUM_PAIRS; pair++) {
        int32_t m0 = m0_dist(rng);
        int     n  = n_dist(rng);
        for (int t = 0; t < TRIALS_PER_PAIR; t++) {
            int32_t acc = acc_dist(rng);

            dut->acc = acc; dut->m0 = m0; dut->n = n;
            dut->eval(); // combinational module -- no clock needed

            int8_t hw = (int8_t)dut->q_out;
            int8_t sw = requantize(acc, m0, n);
            if (hw != sw) {
                printf("MISMATCH acc=%d m0=%d n=%d hw=%d sw=%d\n", acc, m0, n, hw, sw);
                assert(false);
            }
        }
    }
    printf("1,000,000 vectors matched bit-exactly across 100 (M0, n) pairs.\n");
    delete dut;
    return 0;
}
```

> [!success] Done when
> The Verilator testbench runs to completion with zero mismatches across 1,000,000 random `acc` values and 100 distinct `(M0, n)` pairs, including pairs with `n = 0`, `n = 31`, and accumulator values at `INT32_MIN` and `INT32_MAX`.

> [!question]- Mismatches between C and Verilog?
> Roughly sorted from "most common" to "most obscure":
> 1. **Sign extension** — is `acc` treated as signed everywhere in the Verilog (`signed` on every intermediate wire, not just the port), or did an unsigned wire silently zero-extend somewhere?
> 2. **Rounding direction on negatives** — did you implement the mirror trick (negate → round magnitude → negate back) identically on both sides, or did one side use a plain `>>>`/`>>` and hope for the best?
> 3. **Saturation bounds** — off-by-one is easy: is it `> 127` / `< -128`, not `>= 127` / `<= -128`?
> 4. **Shift amount mismatch** — is `n` computed the same way on both sides? A common bug is deriving `n` one way in `quantize_multiplier` and a slightly different way when hand-picking Verilog test values.
> 5. **Test coverage** — does your random generator actually hit negative `acc`, `acc = 0`, `n = 0`, and `INT32_MIN`/`INT32_MAX`? A test that only exercises positive, mid-range values will pass even with a broken negative-rounding branch.

> [!danger] Gotcha
> **Rounding is the bug that will make your model output mush, and it will not announce itself.** Arithmetic right shift of a negative number rounds toward $-\infty$, not toward zero — `-3 >> 1` is `-2`, not `-1`. If C and Verilog implement rounding even slightly differently — say C mirrors the magnitude but Verilog naively does `(prod + half) >>> n` on the raw signed value — they'll agree on most inputs and silently diverge on a subset of negative values near shift boundaries. Your accelerator won't crash; it will run, produce plausible-looking-but-wrong numbers, and by the time you notice the model has degraded you'll have dozens of more-complicated modules to rule out first. Get this bit-exact now, while it's still two functions you can compare directly.

## Check your understanding

> [!question]- What does "Qm.n" actually specify, and what's the total register width?
> > [!success]- Answer
> > A two's-complement integer register split into `m` integer bits and `n` fractional bits, plus one sign bit, total width `1 + m + n`. The bits are ordinary integer bits — Qm.n is a convention for *interpreting* them, not a different kind of hardware.

> [!question]- What Q format results from multiplying a Q3.13 value by a Q1.15 value?
> > [!success]- Answer
> > **Q4.28.** Integer bits add: `3 + 1 = 4`. Fractional bits add: `13 + 15 = 28`. Track that the result's binary point has moved to bit 28 — nowhere close to either input's original point.

> [!question]- Why does hardware quantize weights per-channel but activations per-tensor?
> > [!success]- Answer
> > Output channels of a trained weight matrix often have very different dynamic ranges — one might peak at 0.02, another at 1.5. Sharing one scale would waste most of int8's 256 codes on the small-magnitude channels, so per-channel scales let each one use its full range. Activations don't need that structure, and per-tensor keeps the MAC hardware simple — every term in a row shares one activation scale.

> [!question]- Why does `-3 >> 1` evaluate to `-2` instead of `-1`, and why does it matter for this phase?
> > [!success]- Answer
> > Arithmetic right shift on a negative number rounds toward $-\infty$ (floor division), not toward zero: `-3 / 2 = -1.5`, and flooring gives `-2`. If you'd modeled `>>` as "truncating division" you'd expect `-1` and be wrong. It matters because the requantization pipeline shifts values that are frequently negative — if C and Verilog don't apply the identical round-half-away-from-zero correction instead of a raw shift, they'll disagree exactly on the inputs this bug hides in.

## What's next

With a bit-exact requantization pipeline proven in both languages, you have the one piece of arithmetic every MAC unit in your accelerator depends on. [[phase-3-mac-engine]] builds the array of multiply-accumulate units that feeds this pipeline and turns it into an actual systolic compute engine.

<script src="/tutor.js" defer></script>
