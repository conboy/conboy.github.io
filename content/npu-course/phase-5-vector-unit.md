---
title: "Phase 5 — The Vector Unit"
description: "RMSNorm, softmax, SiLU and RoPE in fixed point — and Amdahl's Law arriving on schedule."
tags:
  - npu
  - hardware
---

# Phase 5 — Everything that isn't matmul

**4 weeks** · Part of [[index|Zero to NPU]] · Prev: [[phase-4-memory]] · Next: [[phase-6-sequencer]]

> [!info] What this phase is for
> Back in [[phase-1-baseline-llm|Phase 1]] you profiled stories260K on the ARM core and found matmul eating 85–95% of every millisecond. In [[phase-3-mac-engine|Phase 3]] you built an 8×4 array of PEs and made matmul roughly **5× faster**. That sounds like a clean win, and for matmul itself it is — but [[concepts#Amdahl's Law|Amdahl's Law]] says the *rest* of the runtime doesn't shrink just because matmul did. Everything that used to be a rounding error next to matmul — normalization, attention weighting, activation functions, position encoding — is now roughly **35% of your runtime**, and none of it runs on your PE array. This phase builds a small, shared "vector unit": one modest datapath (about 12 DSP slices, a few BRAMs for lookup tables, under 2,000 LUTs) that handles the four operations a transformer layer needs besides matmul — [[concepts#RMSNorm|RMSNorm]], [[concepts#Softmax|Softmax]], [[concepts#SiLU|SiLU/SwiGLU]], and [[concepts#RoPE|RoPE]]. None of these are individually hard. What makes them worth four weeks is that each one hides a fixed-point trap that will silently corrupt your model's output if you don't understand *why* the trick works, not just that it exists.

---

## Amdahl's Law, concretely

[[concepts#Amdahl's Law|Amdahl's Law]] is really just one piece of arithmetic, and it's worth doing by hand once so the rest of this phase makes sense instead of feeling like a rule someone told you to follow.

Say your baseline run takes **100 units of time** total. Based on your Phase 1 profile, about **90 units are matmul** and **10 units are everything else** — RMSNorm, softmax, SiLU, RoPE, and the residual adds. That "everything else" bucket is what this phase is about.

Now you speed matmul up 5×, exactly like Phase 3 did. Matmul's 90 units become 90 / 5 = **18 units**. The other 10 units haven't changed at all — you didn't touch them — so your new total is 18 + 10 = **28 units**. Two things happen here that trip people up:

1. **Your overall speedup is not 5×.** It's 100 / 28 ≈ **3.6×**. You made one part of the system five times faster and got less than four times faster overall, because the part you *didn't* speed up is now a much bigger fraction of what's left.
2. **The "other" bucket's share of runtime nearly quadrupled.** It was 10% of 100 units before (10/100). Now it's 10 units out of 28 total — about **36%**. Same absolute cost, much bigger relative cost, purely because you shrank the thing next to it.

This is the whole mechanism behind the phase. If you now build a vector unit that makes that "other" bucket 4× faster, it drops from 10 units to 2.5 units, and your total becomes 18 + 2.5 = **20.5 units** — an overall speedup of 100/20.5 ≈ **4.9×**, much closer to matmul's own 5×. You cannot get there by making matmul faster again; matmul is already down to 18 units and squeezing it further has rapidly diminishing returns while the vector-unit work is still cheap and unclaimed. That's the entire argument for spending four weeks on operations that individually look tiny next to a 32-PE systolic array: once matmul stops dominating, everything else becomes the bottleneck by definition, whether or not it "feels" important.

---

## Where these four operations live in a transformer layer

It helps to see exactly where in a single transformer layer these four operations sit, because they are not random — they are the specific glue between matmuls that Phase 1's reference model needs at every layer, five times over for stories260K (`n_layers=5`).

```
x (residual stream, dim=64)
  │
  ├─► RMSNorm ──► [Q,K,V projections]  (matmul, Phase 3)
  │                       │
  │                RoPE on Q and K
  │                       │
  │              [attention scores]    (matmul, Phase 3)
  │                       │
  │                   Softmax
  │                       │
  │            [weighted sum over V]   (matmul, Phase 3)
  │                       │
  └───────────────────► add (residual)
                          │
                    RMSNorm ──► [FFN gate, FFN up]  (matmul, Phase 3)
                          │             │
                          │         SiLU(gate)
                          │             │
                          │      elementwise multiply  (SwiGLU)
                          │             │
                          │       [FFN down]           (matmul, Phase 3)
                          │             │
                          └───────────► add (residual)
```

Every box labeled "matmul" runs on the 8×4 PE array you built in Phase 3 — that hardware only knows how to multiply-and-accumulate. Every other box — RMSNorm, RoPE, Softmax, SiLU, and the elementwise multiply and adds — is this phase's job. Notice the pattern: RMSNorm always appears *before* a block of matmuls (it conditions the input), RoPE modifies Q and K right after they're produced and before they're used in attention, softmax turns raw attention scores into weights right after the score matmul, and SiLU/SwiGLU sits between the two halves of the feed-forward network. With `n_heads=8`, `n_kv_heads=4`, and `head_size=8`, this exact sequence repeats once per attention block per layer, and the FFN sequence repeats once per layer — five times end to end for stories260K. None of these operations are optional extras; skip any one of them and the model's output stops matching the golden reference almost immediately, because each one is doing real numerical work the matmuls depend on.

---

## 5.1 RMSNorm

**What it does:** RMSNorm rescales a vector so its values sit at a consistent, predictable magnitude before the next matmul touches them. You can think of it as a cousin of LayerNorm, but simpler: it skips the mean-subtraction step and skips the learned bias term, using only a per-channel scale. That simplicity is exactly why it's cheap enough to build in fixed point.

**The math:**

$$\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\dfrac{1}{n}\sum_{j=1}^{n} x_j^2 + \epsilon}} \cdot g_i$$

- $x_i$ — the $i$-th element of the input vector (here, $n = 64$, matching `dim=64`)
- $n$ — the vector length being normalized (64)
- $\epsilon$ — a tiny constant added for numerical safety, so you never divide by exactly zero
- $g_i$ — a learned per-channel scale (a weight from the trained model, one per dimension)

**Why it's here:** Without it, the scale of activations can drift layer to layer — five layers deep, small biases compound. A matmul fed activations that are sometimes tiny and sometimes huge will alternately lose precision to rounding and overflow its accumulator. RMSNorm keeps every layer's input in a known, bounded range so the rest of your fixed-point pipeline can be sized correctly instead of defensively oversized.

**How to build it in fixed point:** Your input is 64 int8 values (`dim=64`). Square each one: an int8 value maxes out at ±127, and $127^2 = 16{,}129$, which fits easily in an int16 but you should accumulate the running sum in **int32**. Sum all 64 squares: worst case is $64 \times 127^2 \approx 1{,}032{,}256$ — about **1.03 million**, which fits comfortably inside a 32-bit accumulator (max ~2.1 billion) with enormous headroom. This is the overflow analysis for this op, and it's the reassuring one in this phase: there is no realistic way for this particular sum to overflow int32.

Divide that sum by $n=64$ to get the mean of squares — since 64 is a power of two, this is a free right-shift by 6 bits, not a real division. Add $\epsilon$ (represented as a small fixed-point constant), and now you need $1/\sqrt{\cdot}$ of the result. Don't compute a real square root in hardware if you can avoid it — build a small **reciprocal-square-root lookup table** indexed by the (mean-square + eps) value, get an approximate answer $y_0$, then sharpen it with **one Newton-Raphson iteration**, the same trick the famous "fast inverse square root" hack uses in software:

$$y' = \frac{y\,(3 - x \cdot y^2)}{2}$$

Here $x$ is the value you're taking the reciprocal square root of (mean-square + eps), $y$ is your LUT's initial guess, and $y'$ is the refined result — one multiply-heavy iteration turns a coarse LUT into something accurate enough to be bit-exact against the golden model after requantization.

```c
// RMSNorm reference, fixed-point style (int32 accumulate, LUT + 1 NR step)
int32_t sumsq = 0;
for (int i = 0; i < 64; i++) {
    int32_t xi = (int32_t)x[i];          // int8 -> int32, sign-extended
    sumsq += xi * xi;                    // max ~1.03M, safe in int32
}
int32_t mean_sq = sumsq >> 6;            // divide by 64 (free shift)
int32_t arg = mean_sq + EPS_FIXED;       // add epsilon

int32_t y0 = rsqrt_lut[quantize_index(arg)];   // 256 or 512-entry LUT, coarse
// One Newton-Raphson refinement: y' = y*(3 - arg*y^2)/2
int32_t y1 = fixed_mul(y0, (3*ONE - fixed_mul(arg, fixed_mul(y0, y0)))) / 2;

for (int i = 0; i < 64; i++) {
    int32_t scaled = fixed_mul((int32_t)x[i], y1);   // x_i * rsqrt(mean_sq+eps)
    out[i] = requantize(fixed_mul(scaled, g[i]));    // apply learned g_i, requant (Phase 2)
}
```

```verilog
// RMSNorm datapath sketch: accumulate squares, then hand off to shared rsqrt LUT+NR unit (5.5)
module rmsnorm_accum #(parameter N = 64) (
    input  wire              clk, rst_n, start,
    input  wire signed [7:0] x_in,       // one element per cycle
    input  wire               x_valid,
    output reg  signed [31:0] sumsq,     // int32 accumulator
    output reg                done
);
    reg [6:0] count;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin sumsq <= 0; count <= 0; done <= 0; end
        else if (start) begin sumsq <= 0; count <= 0; done <= 0; end
        else if (x_valid && count < N) begin
            sumsq <= sumsq + ($signed(x_in) * $signed(x_in)); // widens to 32b
            count <= count + 1;
            done  <= (count == N-1);
        end
    end
endmodule
```

> [!success] Done when
> Your RMSNorm output is bit-exact against the golden model for every test vector, and specifically: the int32 `sumsq` you compute matches the golden model's sum-of-squares exactly (not just the final normalized output) — that catches accumulator bugs before they hide behind the LUT/NR stage.

> [!question]- It's not working
> 1. Are you accumulating `sumsq` in int32, or did a temporary somewhere truncate to int16 and silently wrap?
> 2. Is your right-shift-by-6 actually dividing by 64, or did you shift by the wrong amount (a classic off-by-one on `log2(64)=6`)?
> 3. Does your rsqrt LUT's index calculation match the *exact* quantization scheme the golden model used to build that LUT — a mismatched scale factor here won't crash anything, it'll just be subtly wrong?
> 4. Did you actually apply the Newton-Raphson refinement, or are you comparing against the coarse LUT value alone (close but not bit-exact)?
> 5. Are you applying the per-channel `g_i` weight *after* the rsqrt multiply, in the right order, and requantizing (Phase 2) at the right point in the chain?
> 6. Does `eps` use the same fixed-point scale as `mean_sq`, or are you adding a raw float-derived constant into an integer domain?

---

## 5.2 Softmax

**What it does:** Softmax takes a row of raw attention scores — arbitrary signed numbers — and turns them into a proper probability distribution: values between 0 and 1 that sum to 1, with larger scores getting disproportionately more weight. It's the step that decides how much attention each token pays to every other token.

**The math:**

$$\text{softmax}(x)_i = \frac{e^{x_i - \max(x)}}{\displaystyle\sum_{j} e^{x_j - \max(x)}}$$

- $x_i$ — the $i$-th raw attention score in the row
- $\max(x)$ — the largest score in that row
- $e^{x_i - \max(x)}$ — the exponentiated, shifted score (always $\le 1$ after shifting)
- the denominator — the sum of all shifted-and-exponentiated scores in the row, used to normalize

**Why it's here:** Attention scores by themselves aren't weights — they're arbitrary magnitudes. Softmax is what converts "bigger score" into "proportionally more attention," and it's what makes the weighted-sum-over-V matmul that follows actually meaningful instead of using raw, unbounded numbers as weights.

**How to build it in fixed point:** The subtract-the-max step is not optional polish — it is load-bearing. $e^x$ grows explosively: $e^{21} \approx 1.3 \times 10^9$, already brushing up against int32's ceiling of about $2.1 \times 10^9$, and your raw attention scores can easily exceed that in magnitude before shifting. Without subtracting the row max first, a single large score anywhere in the row overflows your fixed-point exponential and every downstream weight in that row becomes garbage — silently, with no crash, just wrong numbers propagating into the next matmul. Subtracting the max guarantees every exponent argument is $\le 0$, so every $e^{x_i-\max}$ is $\le 1$, and overflow simply cannot happen in that step.

Once every value is $\le 0$ and bounded below by your fixed-point range, build a **256-entry exp lookup table**: index it by the (quantized, non-positive) shifted score, get back a fixed-point approximation of $e^{x_i-\max}$. Sum those 256-or-fewer results in an int32 accumulator (same style of safety margin as RMSNorm), then get the reciprocal of that sum — either with its own small LUT or by reusing the same LUT+Newton-Raphson machinery from 5.1, since a reciprocal is a simpler cousin of reciprocal-square-root. Multiply each exponentiated value by that reciprocal and requantize.

If you want to avoid two full passes over the row (one to find the max, one to exponentiate-and-sum), look at **online softmax** (Milakov & Gimelshein, *"Online normalizer calculation for softmax"*): it updates a running max and a running sum together in a single pass, rescaling the running sum whenever a new max is found. That's a real engineering option here, not just an academic curiosity — it's exactly the kind of one-pass streaming trick that matters when your row is arriving from BRAM one element per cycle instead of sitting fully resident in registers.

```c
// Two-pass fixed-point softmax with max-subtraction
int8_t row_max = x[0];
for (int i = 1; i < len; i++) if (x[i] > row_max) row_max = x[i];

int32_t sum = 0;
int32_t exp_vals[MAX_LEN];
for (int i = 0; i < len; i++) {
    int shifted = (int)x[i] - (int)row_max;      // always <= 0, no overflow risk
    exp_vals[i] = exp_lut[quantize_index(shifted)]; // 256-entry LUT
    sum += exp_vals[i];                          // int32 accumulate
}
int32_t recip = reciprocal_lut_with_nr(sum);     // reuse 5.1's LUT+NR machinery
for (int i = 0; i < len; i++) {
    out[i] = requantize(fixed_mul(exp_vals[i], recip));
}
```

```verilog
// Softmax exp-LUT stage: one shifted score in, one LUT-approximated exp value out
module softmax_exp_lut (
    input  wire signed [7:0] shifted_score,  // x_i - row_max, always <= 0
    output wire       [15:0] exp_val         // fixed-point e^(shifted_score)
);
    // 256-entry ROM, initialized at synth time from the golden model's table
    (* rom_style = "block" *) reg [15:0] rom [0:255];
    initial $readmemh("exp_lut.mem", rom);
    assign exp_val = rom[shifted_score[7:0]];  // index by the negative offset
endmodule
```

> [!success] Done when
> Your softmax output is bit-exact against the golden model row by row, and specifically: feed a row containing your model's largest realistic raw attention score, confirm your fixed-point sum never overflows int32, and confirm the output row still sums to your fixed-point representation of 1.0 within your defined rounding tolerance.

> [!question]- It's not working
> 1. Did you subtract the row max *before* indexing the exp LUT, or are you feeding raw (possibly positive-and-large) scores into it?
> 2. Is your exp LUT's domain correctly bounded to non-positive inputs only — what happens if a rounding bug lets a slightly-positive value through?
> 3. Is the sum of exponentiated values accumulated in int32, not a narrower type that could wrap for a long row?
> 4. Does your reciprocal step use the same fixed-point scale the golden model's requantization expects, or is there a scale mismatch that would show up as "close but off by a constant factor"?
> 5. If you implemented online softmax, are you correctly *rescaling* the running sum every time a new running max is found, not just updating the max and leaving old sum terms unrescaled?
> 6. Does row length vary (e.g., causal attention masks shortening later rows), and if so, is your LUT/accumulator logic correctly bounded to the valid row length instead of reading garbage past it?

---

## 5.3 SiLU / SwiGLU

**What it does:** SiLU is a smooth, non-linear activation function — the thing that gives the feed-forward network its actual expressive power. Without a non-linearity between matmuls, stacking multiple matmuls back to back is mathematically equivalent to one single matmul, which would make five layers pointless. SwiGLU, as used in this model, is SiLU applied to a "gate" projection and then multiplied elementwise by a separate "up" projection — a gating mechanism that lets the network learn to selectively pass information through.

**The math:**

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}, \qquad \text{SwiGLU}(\text{gate}, \text{up}) = \text{SiLU}(\text{gate}) \odot \text{up}$$

- $x$ — the input value (here, the "gate" projection's output, per element)
- $\sigma(x) = \tfrac{1}{1+e^{-x}}$ — the logistic sigmoid function
- $\text{up}$ — the separate "up" projection, same shape as gate
- $\odot$ — elementwise multiplication

**Why it's here:** This is the non-linearity every transformer FFN needs. It's also the gate in SwiGLU: multiplying by `up` lets the network scale how much of each channel passes through, which is strictly more expressive than a plain non-gated FFN.

**How to build it in fixed point:** This is the easy win of the whole phase, and it's worth saying plainly: because your input is **int8**, there are only **256 possible input values, full stop**. That means you don't need an approximation of SiLU at all — you can precompute the *exact* output for every one of those 256 inputs (using the real floating-point SiLU formula, then requantizing to int8 the same way the golden model does) and store the whole function as a **256-entry int8-to-int8 lookup table**. That's one small BRAM, essentially free in terms of DSPs, and it is not an approximation — it is the exact answer for every input your hardware will ever actually see, because the input domain is small enough to enumerate completely. Once you have `SiLU(gate)` from the LUT, the SwiGLU elementwise multiply against `up` is a single int8×int8 multiply (one DSP) followed by the usual requantization from Phase 2.

```c
// Building the exact SiLU LUT offline (host-side, once, from the same
// quantization scheme the golden model uses)
void build_silu_lut(int8_t lut[256], float scale, int32_t zero_point) {
    for (int q = -128; q <= 127; q++) {
        float x = dequantize(q, scale, zero_point);       // int8 -> float
        float y = x / (1.0f + expf(-x));                  // real SiLU
        lut[q & 0xFF] = quantize(y, scale, zero_point);    // float -> int8, exact for this input
    }
}

// Runtime: SwiGLU = SiLU(gate) * up, elementwise
for (int i = 0; i < hidden_dim; i++) {                     // hidden_dim = 172
    int8_t silu_gate = silu_lut[(uint8_t)gate[i]];
    out[i] = requantize(fixed_mul((int32_t)silu_gate, (int32_t)up[i]));
}
```

```verilog
// SiLU LUT + elementwise multiply, one element per cycle
module silu_swiglu (
    input  wire signed [7:0] gate_in,
    input  wire signed [7:0] up_in,
    output wire       [15:0] product_out   // feeds requantize stage (Phase 2)
);
    (* rom_style = "block" *) reg signed [7:0] silu_lut [0:255];
    initial $readmemh("silu_lut.mem", silu_lut);   // exact table, all 256 inputs

    wire signed [7:0] silu_val = silu_lut[gate_in[7:0]];
    assign product_out = silu_val * up_in;          // one DSP
endmodule
```

> [!success] Done when
> Every one of the 256 possible int8 inputs produces the exact int8 output the golden model's LUT produces (this should be a trivial, total match — if it isn't, the LUT was built with the wrong quantization parameters), and the elementwise SwiGLU product matches the golden model bit-exactly across a full `hidden_dim=172` vector.

> [!question]- It's not working
> 1. Did you build the LUT using the *same* scale and zero-point your model's quantization actually uses, or a generic/default one?
> 2. Are you indexing the LUT with a signed int8 treated as unsigned bits (`gate_in[7:0]`), consistently on both the C reference and the Verilog ROM?
> 3. Since the LUT should be exact, did you check all 256 entries against the golden model at least once, rather than spot-checking a handful and assuming the rest are fine?
> 4. Is the elementwise multiply happening between `SiLU(gate)` and `up`, not accidentally `gate` and `up` (forgetting to apply SiLU first)?
> 5. Is requantization happening after the multiply, with the correct scale for a product of two int8-derived quantized values (this is a Phase 2 concept — a mismatched combined scale here is an easy silent bug)?

---

## 5.4 RoPE

**What it does:** [[concepts#RoPE|RoPE]] (Rotary Position Embedding) is how the model knows word order. Plainly: it takes each pair of dimensions inside a query (Q) or key (K) vector and rotates that pair by an angle that depends on the token's position in the sequence. Tokens at different positions get different rotation angles, so the same word appearing early versus late in a sentence produces measurably different Q/K vectors — that difference is what lets attention scores reflect position, without needing separate positional-embedding vectors added anywhere.

**The math:** for one pair of dimensions $(x_0, x_1)$ at position $\text{pos}$, with per-pair angle $\theta_{\text{pos},i}$:

$$
\begin{aligned}
x_0' &= x_0 \cos(\theta_{\text{pos},i}) - x_1 \sin(\theta_{\text{pos},i}) \\
x_1' &= x_0 \sin(\theta_{\text{pos},i}) + x_1 \cos(\theta_{\text{pos},i})
\end{aligned}
\qquad
\theta_{\text{pos},i} = \frac{\text{pos}}{10000^{\,2i / \text{head\_size}}}
$$

- $(x_0, x_1)$ — one adjacent pair of dimensions within a head
- $\text{pos}$ — the token's position in the sequence (0, 1, 2, …)
- $i$ — which pair within the head this is (pairs are indexed $0 \dots \text{head\_size}/2 - 1$)
- $\text{head\_size}$ — dimensions per attention head (8, for this model)
- $\theta_{\text{pos},i}$ — the rotation angle for this pair at this position; lower-indexed pairs rotate faster with position, higher-indexed pairs rotate more slowly

**Why it's here:** matmul and the other three ops in this phase are all position-agnostic — they'd produce the identical result if you shuffled the token order. Language obviously isn't position-agnostic. RoPE is the one piece of the whole pipeline that injects "where in the sequence am I" into Q and K before they're compared in attention.

**How to build it in fixed point:** The formula has a $\sin$ and $\cos$ for every (position, pair) combination, and computing trig functions in hardware per-token would be wasteful and slow. The fix is exactly what the phase brief says: **precompute every sin/cos value you'll ever need into a BRAM table once, at initialization** — not per token, not per inference call. With `head_size=8`, each head has $8/2 = 4$ pairs, and with a 128-token context window (`128-tok context`, per your board notes), the table needs $128 \text{ positions} \times 4 \text{ pairs} \times 2 \text{ values (sin, cos)} = 1{,}024$ fixed-point entries. At, say, 2 bytes per entry, that's **2,048 bytes** — about 2 KB, trivially small next to your ~4 BRAM-block budget, and it comfortably fits in a single block. Compute it once when the model loads, index it by `(position, pair_index)` at runtime, and apply the rotation formula above as two multiplies and an add/subtract per pair — cheap, and it never touches a trig function at inference time.

```c
// Precompute the RoPE sin/cos table once, at init (NOT per token)
// 128 positions x 4 pairs x {sin, cos}, fixed-point Q-format values
void build_rope_table(int16_t sin_tab[128][4], int16_t cos_tab[128][4]) {
    const int head_size = 8;
    for (int pos = 0; pos < 128; pos++) {
        for (int i = 0; i < head_size / 2; i++) {           // 4 pairs
            float theta = pos / powf(10000.0f, (2.0f * i) / head_size);
            sin_tab[pos][i] = float_to_fixed(sinf(theta));
            cos_tab[pos][i] = float_to_fixed(cosf(theta));
        }
    }
}

// Apply RoPE to one (x0, x1) pair at runtime — pure table lookup + rotate
void rope_apply_pair(int8_t *x0, int8_t *x1, int pos, int pair_idx,
                      const int16_t sin_tab[128][4], const int16_t cos_tab[128][4]) {
    int16_t s = sin_tab[pos][pair_idx];
    int16_t c = cos_tab[pos][pair_idx];
    int32_t new_x0 = fixed_mul(*x0, c) - fixed_mul(*x1, s);
    int32_t new_x1 = fixed_mul(*x0, s) + fixed_mul(*x1, c);
    *x0 = requantize(new_x0);
    *x1 = requantize(new_x1);
}
```

```verilog
// RoPE table lookup + rotate, one pair per cycle
module rope_rotate (
    input  wire [6:0]        pos,        // 0..127
    input  wire [1:0]        pair_idx,   // 0..3 (head_size/2 - 1)
    input  wire signed [7:0] x0_in, x1_in,
    output wire signed [7:0] x0_out, x1_out
);
    // 1024-entry sin/cos tables, precomputed at init, loaded via $readmemh
    (* rom_style = "block" *) reg signed [15:0] sin_rom [0:511]; // 128*4
    (* rom_style = "block" *) reg signed [15:0] cos_rom [0:511];
    initial begin
        $readmemh("rope_sin.mem", sin_rom);
        $readmemh("rope_cos.mem", cos_rom);
    end

    wire [8:0] idx = {pos, pair_idx};          // 128*4 = 512 entries
    wire signed [15:0] s = sin_rom[idx];
    wire signed [15:0] c = cos_rom[idx];

    wire signed [23:0] new_x0 = (x0_in * c) - (x1_in * s);
    wire signed [23:0] new_x1 = (x0_in * s) + (x1_in * c);

    assign x0_out = requantize24_to_8(new_x0);   // Phase 2 requant stage
    assign x1_out = requantize24_to_8(new_x1);
endmodule
```

> [!success] Done when
> Rotated Q and K vectors are bit-exact against the golden model at every tested position, and specifically: position 0 must produce an **identity rotation** ($\theta = 0$, so $\cos=1$, $\sin=0$, meaning $x_0' = x_0$ and $x_1' = x_1$ exactly) — that's a free, cheap sanity check before you trust any other position.

> [!question]- It's not working
> 1. Does position 0 produce an exact identity (no change to the pair) — if not, your table generation or indexing is off before you even get to harder positions.
> 2. Are you pairing dimensions correctly — pair $i$ should be dimensions $(2i, 2i+1)$ within the head, not some other grouping?
> 3. Is `pair_idx` reset per head (0..3 within each 8-wide head), not accidentally running 0..31 across the whole `dim=64` vector?
> 4. Did you build the table with the exact same $\theta_{\text{pos},i}$ formula and base (10000) the golden model uses — a different base gives plausible-looking but wrong rotations?
> 5. Are you applying RoPE to **both** Q and K (not just Q), and at the right point in the pipeline — after the Q/K/V projection matmul, before the attention-score matmul?
> 6. Is the table indexed by absolute sequence position, or did an off-by-one creep in from 0-indexed vs. 1-indexed token counting?

---

## 5.5 Wiring it together

You don't need four separate pieces of hardware for four operations — that would burn far more than your ~12-DSP, ~4-BRAM budget. Instead, build **one small shared datapath**: a modest ALU (a handful of adders and multipliers, reused across ops) plus a single LUT-holding BRAM block, with a `mode` select signal that decides what that cycle's ALU/LUT combination actually computes. RMSNorm and softmax's reciprocal step reuse the same LUT+Newton-Raphson hardware (a coarse LUT plus one refinement multiply-add). Softmax's exp LUT, SiLU's LUT, and RoPE's sin/cos tables are three different *contents* loaded into the same *kind* of BRAM lookup structure — same address-in, data-out shape, different tables. The accumulator (int32, used by both RMSNorm's sum-of-squares and softmax's sum-of-exponentials) is one more shared resource, cleared and reused between ops. This is why the phase fits its tiny budget: you're not building four units, you're building one flexible one and pointing it at different tables and control sequences depending on which of the four operations the command sequencer ([[phase-6-sequencer|Phase 6]]) asks for next.

```verilog
// Shared vector-unit top level: one ALU + one LUT BRAM, mode-selected
module vector_unit (
    input  wire [2:0]  mode,     // 0=RMSNorm, 1=Softmax, 2=SiLU, 3=RoPE
    input  wire signed [7:0] a_in, b_in,
    input  wire [8:0] lut_addr,
    output reg  signed [15:0] result
);
    wire [15:0] lut_data;
    lut_bram u_lut (.addr(lut_addr), .mode(mode), .data_out(lut_data));

    always @(*) begin
        case (mode)
            3'd0: result = rmsnorm_step(a_in, b_in, lut_data);  // sumsq / rsqrt+NR
            3'd1: result = softmax_step(a_in, b_in, lut_data);  // exp LUT / recip+NR
            3'd2: result = a_in * lut_data[7:0];                // SiLU LUT * up
            3'd3: result = rope_step(a_in, b_in, lut_data);     // sin/cos rotate
            default: result = 16'sd0;
        endcase
    end
endmodule
```

---

## Check your understanding

> [!question]- Why does skipping the max-subtraction step in softmax cause silent, catastrophic errors instead of an obvious crash?
> > [!success]- Answer
> > Because $e^x$ grows so fast that even moderate raw scores overflow a fixed-point accumulator — $e^{21} \approx 1.3\times10^9$, already close to int32's ~2.1×10^9 ceiling, and real attention scores can exceed that. Overflow in fixed-point hardware doesn't throw an exception; it wraps or saturates silently, producing a plausible-looking but wrong number that then flows straight into the next matmul. Subtracting the row max first guarantees every exponent argument is ≤ 0, so every term is ≤ 1 and overflow becomes structurally impossible, not just unlikely.

> [!question]- Why is the SiLU LUT described as "exact" rather than "a good approximation," when every other LUT in this phase (rsqrt, exp) is explicitly an approximation refined by Newton-Raphson?
> > [!success]- Answer
> > Because the input to SiLU is int8, and int8 has exactly 256 possible values — full stop, no more. That means you can precompute the true SiLU output for literally every input your hardware will ever see and store all 256 answers directly. There's no rounding error introduced by the lookup itself (only whatever quantization error already exists from representing the value as int8 in the first place). RMSNorm's rsqrt and softmax's exp, by contrast, operate over a wider effective range (sums of squares, sums of exponentials), so a 256-entry table there can only be a coarse starting point — hence the Newton-Raphson refinement step to sharpen it.

> [!question]- Why do RMSNorm and softmax both need a Newton-Raphson refinement step, but RoPE and SiLU don't?
> > [!success]- Answer
> > RMSNorm's reciprocal-square-root and softmax's reciprocal both operate on values from a wide dynamic range (a sum of up to 64 squared values, or a sum of many exponentiated values), so a compact LUT can only approximate the answer coarsely — Newton-Raphson cheaply sharpens that guess to bit-exact precision. RoPE's sin/cos values only ever need to be computed for a small, fully enumerable set of (position, pair) combinations — 128 × 4 in this model — so the table itself can just store the exact precomputed answer directly, no refinement needed. SiLU is exact for the same reason: its input domain (256 int8 values) is small enough to enumerate completely.

> [!question]- Why does this whole phase exist given that matmul made up 85–95% of runtime back in Phase 1?
> > [!success]- Answer
> > Because that percentage described the *old* runtime, before Phase 3's ~5× matmul speedup. Amdahl's Law means shrinking the dominant piece automatically inflates the relative size of everything else — the "everything else" bucket goes from roughly 10% of total time to roughly 35–36%, even though its absolute cost never changed. Once matmul stops dominating, the vector-unit operations become the next real bottleneck, and squeezing more speed out of the already-fast matmul array yields far less benefit than making this newly-dominant 35% chunk faster.

---

## What's next

You now have a working vector unit that handles every non-matmul operation a transformer layer needs, wired to share one ALU and one LUT BRAM across four modes. [[phase-6-sequencer]] is where you build the control logic that actually sequences matmul and vector-unit operations correctly, layer after layer, so the whole pipeline runs itself instead of needing you to single-step it.
