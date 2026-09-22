---
title: "Concepts — Glossary"
description: "Plain-English definitions for every term in the NPU course, from LUT to KV cache."
tags:
  - npu
  - reference
---

# Concepts

This is the glossary for the course: every term used building an int8 NPU on the Blackboard and running `stories260K`, assuming zero prior ML, fixed-point, AXI, or systolic-array knowledge. Read it front to back or jump straight to a linked word. Each entry ends with a *Used in* line pointing at the phase where it does real work.

## FPGA & hardware basics

### LUT
**A tiny truth table that computes any boolean function of a handful of inputs.**

The basic unit of FPGA logic. Each LUT on your Blackboard's fabric has 6 inputs, so it implements any boolean function of up to 6 bits. You have 14,400 total, shared across your MAC engine, vector unit, and command sequencer. "LUT utilization: 62%" means that much of the budget is spent.

*Used in: [[phase-0-toolchain]], [[board-specs]]*

### Flip-flop
**A single bit of memory that updates once per clock edge.**

A flip-flop latches its input on a clock edge and holds it, turning combinational logic into a circuit with state — a counter, an FSM's current state. Every `always @(posedge clk)` register becomes flip-flops. Each LUT on 7-series parts is paired with one.

*Used in: [[phase-0-toolchain]], [[phase-3-mac-engine]]*

### BRAM
**Dedicated on-chip RAM blocks, separate from the LUT fabric.**

Fast, dual-port memory built as hard blocks rather than assembled from LUTs. Each block holds 36 Kb (4.5 KB); you have 50, for 225 KB total. Dual-port lets one port fill a buffer from DDR while another feeds the MAC engine. Budget it carefully — the KV cache alone eats about 9 blocks.

*Used in: [[phase-3-mac-engine]], [[phase-4-memory]], [[board-specs]]*

### DSP slice
**A hardened multiply-accumulate unit, far smaller and faster than the same thing built from LUTs.**

Each DSP48E1 contains a 25×18-bit signed multiplier feeding a 48-bit accumulator, in dedicated silicon. You have about 60. That accumulator is why int8×int8 products can sum thousands of times without overflow. Your MAC engine's throughput is capped by how many of these 60 you keep busy each cycle.

*Used in: [[phase-3-mac-engine]], [[board-specs]]*

### PS
**The hard ARM processor system baked into the chip, separate from the FPGA fabric.**

PS is the single-core ARM Cortex-A9 at 666 MHz plus its DDR3 controller and peripherals — ordinary embedded Linux territory. Your baseline `llama2.c` runs here in Phase 1, and it's where you write the driver code that talks to the accelerator over AXI.

*Used in: [[phase-1-baseline-llm]], [[board-specs]]*

### PL
**The programmable logic fabric — the reconfigurable half of the chip you write Verilog for.**

LUTs, flip-flops, BRAM, and DSP slices wired together by synthesizing a bitstream; nothing runs until it's loaded. The PL is where your accelerator lives — MAC engine, vector unit, command sequencer — while the PS feeds it work over AXI. Internalizing this split is the key idea before Phase 3.

*Used in: [[phase-3-mac-engine]], [[phase-0-toolchain]], [[board-specs]]*

### Fabric
**Shorthand for the programmable logic itself — the sea of LUTs, flip-flops, and routing.**

"How much fabric does that use" means LUTs, flip-flops, and routing, as distinct from the hard PS or the BRAM/DSP blocks embedded in the PL. It's used interchangeably with PL. Watching usage matters — 14,400 LUTs disappears fast once a MAC array and AXI plumbing compete for it.

*Used in: [[phase-3-mac-engine]], [[board-specs]]*

### FSM
**A finite state machine: a circuit whose next state and outputs depend only on its current state and inputs.**

How you sequence hardware — "do this, then that, then wait, then repeat." You'll write plenty: an AXI read sequence, a command sequencer walking a descriptor ring, a softmax unit stepping through find-max, exponentiate, normalize. State lives in flip-flops; next-state logic is combinational.

```verilog
always @(posedge clk) case (state)
  IDLE:      if (start) state <= SEND_ADDR;
  SEND_ADDR: if (awready) state <= WAIT_DATA;
endcase
```

*Used in: [[phase-3-mac-engine]], [[phase-6-sequencer]]*

### Timing closure
**Getting your design to run correctly at its target clock frequency, with no signal arriving late.**

Every path between two flip-flops must settle within one clock period or the receiver latches garbage. Timing closure means every path meets that deadline, checked after place-and-route. Failing it means simplifying the path, pipelining it, or lowering the clock.

*Used in: [[phase-3-mac-engine]], [[phase-8-optimization]]*

### Interrupt
**A hardware signal asking the CPU to stop and run a specific handler.**

Instead of the ARM core polling "is it done," the PL raises an interrupt the instant a job finishes, freeing the CPU to sleep or do other work. You'll wire your command sequencer's "descriptor complete" flag to an interrupt and write a driver that waits on it instead of spinning.

*Used in: [[phase-6-sequencer]], [[phase-7-end-to-end]]*

## Buses & memory

### AXI
**ARM's on-chip bus standard connecting the PS to the PL, built from five independent handshaking channels.**

Five channels — write address, write data, write response, read address, read data — each with its own VALID/READY handshake: a transfer happens when both are high on the same edge. Channels run simultaneously and out of order, which makes AXI fast but fiddly to get right.

*Used in: [[phase-4-memory]], [[phase-6-sequencer]]*

### AXI-Lite
**The simplified AXI subset: no bursts, one transfer at a time, used for control rather than bulk data.**

Same five channels, but every transaction moves exactly one data beat — right for a start bit or status register. You'll expose your accelerator's control registers over AXI-Lite, while weight and activation data moves over full, burst-capable AXI.

*Used in: [[phase-6-sequencer]]*

### Burst
**One address phase followed by many data beats, amortizing per-transaction overhead over more data.**

Sending an address for every 4-byte word wastes as much bus time on addresses as data. A burst sends one address and a length, then streams up to 256 beats — how you approach your DDR3's ~1.2 GB/s. A burst may never cross a 4 KB boundary, so your DMA logic must split any transfer that would.

*Used in: [[phase-4-memory]]*

### Bandwidth
**How many bytes per second you can actually move, as opposed to operations per second you can compute.**

Your Blackboard's 16-bit DDR3 delivers roughly 1.2 GB/s. Every matmul pulls weights across that same pipe, and eventually the MAC array waits on data rather than the reverse. Bandwidth and compute throughput are the two numbers that decide if a layer is compute- or memory-bound — what a roofline plot shows.

*Used in: [[phase-4-memory]], [[phase-8-optimization]], [[board-specs]]*

### Cache coherency
**The guarantee that the CPU's cached view of memory and the accelerator's view agree.**

If your accelerator writes to DRAM while the CPU holds stale cached data for that address, the CPU reads garbage unless the line is flushed. Some AXI ports on this Zynq auto-snoop; a non-coherent one needs explicit flush/invalidate around every accelerator access.

*Used in: [[phase-4-memory]], [[phase-7-end-to-end]]*

### Double buffering
**Using two buffers so one fills while the other is consumed, hiding transfer latency behind compute.**

The MAC engine reads from buffer A while DMA fills buffer B with the next tile; they swap once both finish, so the engine never stalls on DDR. It's the standard fix for a "wait, compute, wait" pattern, at the cost of doubling the BRAM for whatever's buffered.

*Used in: [[phase-4-memory]], [[phase-8-optimization]]*

### Tiling
**Splitting a large matrix into chunks small enough to fit in on-chip BRAM, processed one at a time.**

Weight matrices don't fit in 225 KB of BRAM alongside everything else, so you never load a whole matrix at once — you pull in a tile, compute its partial contribution, accumulate, move on. Too small wastes DMA overhead; too large won't fit next to your double buffers.

*Used in: [[phase-3-mac-engine]], [[phase-4-memory]]*

### Roofline
**A plot of achievable performance against operational intensity, showing whether you're compute-bound or bandwidth-bound.**

Plots performance (ops/sec) against intensity (ops per byte moved). At low intensity you're bandwidth-bound, capped by a slanted line set by your ~1.2 GB/s DDR3; at high intensity you're compute-bound, capped by a flat line set by your ~60 DSP slices.

```
Performance = min(Peak Compute, Operational Intensity × Bandwidth)
```

*Used in: [[phase-8-optimization]]*

## Arithmetic

### Fixed point
**Fractional numbers represented as plain integers with an implied, fixed binary point.**

Your fabric has no floating-point worth using here, so every value is an integer implicitly scaled by a power of two — e.g. int16 value 16384 might mean 0.5 if the point sits after bit 15. Fixed-point math is ordinary integer add/multiply/shift; the extra work is tracking where the point sits after each op.

*Used in: [[phase-2-numerics]]*

### Q-format
**A notation, Qm.n, meaning m integer bits and n fraction bits, plus an implicit sign bit.**

Q1.15 means 1 integer bit, 15 fraction bits in a signed 16-bit value: -1.0 to just under +1.0 in steps of 1/32768. Multiplying two Q1.15 numbers adds the bit counts to Q2.30, wider than either input — you must explicitly shift it back down for Q1.15 again.

> [!tip] The bit count always grows on multiply
> Qa.b × Qc.d = Q(a+c).(b+d). You decide where to truncate the wider result back down.

*Used in: [[phase-2-numerics]]*

### Quantization
**Converting a float32 weight or activation into a low-bit integer — here, int8 — that approximates it.**

`stories260K`'s weights start as float32. Quantization picks a scale per tensor so the float range maps onto -128..127, then rounds each value to the nearest integer — what lets the model fit in ~60 DSP slices and 225 KB of BRAM, at a small accuracy cost you measure with perplexity.

```
q = round(x / scale)   where scale ≈ max(|x|) / 127
```

*Used in: [[phase-2-numerics]]*

### Requantization
**Converting an int32 MAC-accumulator result back to int8 after a matmul, so the next layer can consume it.**

Each MAC accumulates in int32 via the DSP slice's accumulator. Before the next layer can use that sum, it multiplies by a per-tensor scale M0, applies a rounding right shift, then saturates to -128..127.

*Used in: [[phase-2-numerics]], [[phase-3-mac-engine]]*

### Saturation
**Clamping an out-of-range result to the nearest representable value instead of letting it wrap around.**

Plain overflow wraps: 127 + 1 becomes -128 in signed int8, flipping sign silently. Saturating arithmetic clamps instead — above 127 becomes 127, below -128 becomes -128. Every requantization needs a saturate stage or one large activation corrupts everything downstream.

```
sat_int8(x) = max(-128, min(127, x))
```

*Used in: [[phase-2-numerics]], [[phase-3-mac-engine]]*

### MAC
**Multiply-accumulate: compute a×b and add it to a running sum, the single most repeated operation in the project.**

A MAC is `acc += a * b`. Matrix multiplication — nearly everything a transformer does — is a huge number of these done as dot products. Phase 3 is, at its core, keeping as many of your ~60 DSP slices busy as possible every cycle.

*Used in: [[phase-3-mac-engine]], [[board-specs]]*

## Accelerator architecture

### PE
**A processing element: one small, repeated compute-plus-storage unit tiled to build a bigger array.**

One MAC unit plus a little local storage (a stationary weight, small I/O registers) that passes data to its neighbors each cycle. You build one PE, get it timing-closed, then instantiate it N times — thinking "one PE, replicated" rather than "one giant block" is what makes closure achievable.

*Used in: [[phase-3-mac-engine]]*

### Systolic array
**A grid of PEs that pass operands to their neighbors every cycle, so each value is read from memory once and reused many times.**

Instead of re-reading a weight or activation from BRAM per MAC, data pipes through a grid in lockstep — each PE reads from its neighbor, computes, and passes data onward next cycle. This lets a few BRAM ports feed many more DSP slices without BRAM becoming the bottleneck.

*Used in: [[phase-3-mac-engine]]*

### Weight-stationary
**A systolic dataflow where each PE holds one weight fixed in a register while activations stream past it.**

Each PE is loaded with one weight at the start of a tile and keeps it there; activations flow through cycle by cycle, and each PE multiplies the passing activation by its resident weight and accumulates. This minimizes reloading weights from BRAM, since weight matrices are read far more often than any single activation.

*Used in: [[phase-3-mac-engine]]*

### Descriptor
**A small memory structure describing one unit of work for the accelerator — source, destination, size, and flags.**

Rather than the PS poking a dozen registers per matmul, it writes one descriptor: "read this many bytes from here, write results there, then interrupt." Your command sequencer fetches and interprets descriptors and drives the actual AXI transactions and MAC array.

*Used in: [[phase-6-sequencer]]*

### Ring buffer
**A fixed-size circular queue of descriptors the CPU writes into and the accelerator reads out of, wrapping at the end.**

The PS pushes descriptors onto the ring; the sequencer walks it in order and wraps to index 0 past the last slot. A write pointer (PS) and read pointer (PL) track progress. This lets the CPU queue several matmuls without waiting for the first to finish.

*Used in: [[phase-6-sequencer]]*

### Doorbell
**A register write telling the accelerator "new work is available" without it having to poll.**

After writing descriptors, the PS "rings the doorbell" with a single AXI-Lite write, waking the sequencer's FSM. Without one, your PL logic would poll memory continuously, wasting power; with one, it sits idle until told there's work.

*Used in: [[phase-6-sequencer]]*

### Amdahl's Law
**Overall speedup is capped by the fraction of the work you didn't speed up, no matter how much you accelerate the rest.**

If matmuls get 20× faster but were only 80% of runtime, overall speedup is nowhere near 20×. With `p` the sped-up fraction and `s` its speedup: speedup = 1 / ((1-p) + p/s). Plug in p=0.8, s=20: ≈ 4.17×. This is why Phase 5 and Phase 8 exist.

> [!tip] Do the arithmetic before you optimize
> Ask what fraction of wall-clock time the MAC engine owns before spending a week speeding it up further.

*Used in: [[phase-8-optimization]]*

## Machine learning

### Token
**A chunk of text the model treats as one discrete unit — not necessarily a whole word.**

`stories260K` has a vocabulary of 512 tokens: every text is built from a fixed dictionary of 512 chunks. The model never sees raw text — a tokenizer maps text to integers in [0, 511], the model operates on those integers, and a detokenizer maps output back to text.

*Used in: [[phase-1-baseline-llm]]*

### Embedding
**A lookup table mapping each of the 512 possible tokens to a learned vector of numbers.**

Shape (vocab=512, dim=64): row `i` is a 64-number vector representing what the model has learned about token `i`. Looking one up is just indexing — a memory read, no multiplication — cheap next to the matmuls that follow.

*Used in: [[phase-1-baseline-llm]]*

### Logits
**The model's raw, unnormalized output scores — one per vocabulary entry — before they become probabilities.**

After a token passes through all 5 layers, the model produces 512 raw scores, higher meaning more likely. Logits can be negative and don't sum to 1 — softmax turns them into a probability distribution you sample from to pick the next token.

*Used in: [[phase-1-baseline-llm]]*

### Transformer
**The neural network architecture behind `stories260K`: alternating attention and feed-forward blocks, stacked in layers.**

Pushes token embeddings through a stack of identical blocks — 5 here — each with an attention sub-layer (a token looks at previous tokens) and a feed-forward sub-layer (SwiGLU), plus RMSNorm and a residual skip around each. Every weight matrix inside gets quantized and fed through your MAC engine.

*Used in: [[phase-1-baseline-llm]]*

### Attention
**The mechanism letting each token look back at previous tokens and weight how much each one matters.**

For each token, attention compares its query against every earlier token's key to score relevance, turns scores into weights via softmax, then sums the corresponding value vectors. `stories260K` uses 8 query heads sharing 4 key/value heads (grouped-query attention) in parallel.

*Used in: [[phase-1-baseline-llm]], [[phase-5-vector-unit]]*

### KV cache
**Stored key and value vectors from previous tokens, so you never recompute them when generating the next one.**

Without a cache, generating token N would re-run attention over all earlier tokens' keys and values, which never change once computed. Size = `seq_len × n_layers × 2 × kv_dim` bytes; here (kv_dim 32, 5 layers, 128-token cap): 128 × 5 × 2 × 32 = 40,960 bytes, ~41 KB.

> [!tip] Why the cache matters for BRAM budgeting
> 41 KB is roughly 9 of your 50 BRAM blocks — before allocating a byte for weight tiles. Budget it first.

*Used in: [[phase-1-baseline-llm]], [[phase-4-memory]], [[board-specs]]*

### RMSNorm
**A normalization step that rescales a vector by its root-mean-square magnitude, without subtracting the mean or adding a bias.**

Rescales a vector so its magnitude stays stable regardless of the previous layer's output size, keeping a 5-layer stack well-behaved. Unlike LayerNorm, it skips mean subtraction and bias — just a learned per-dimension scale `g`.

```
RMSNorm(x)_i = x_i * g_i / sqrt(mean(x^2) + eps)
```

*Used in: [[phase-1-baseline-llm]], [[phase-5-vector-unit]]*

### Softmax
**Turns a vector of raw scores into a probability distribution that sums to 1.**

Exponentiates every score and divides by the sum of all exponentials, so bigger scores get proportionally bigger, positive weights. Subtracting the max score first changes nothing mathematically but prevents `exp()` from overflowing in fixed-point.

```
softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
```

*Used in: [[phase-1-baseline-llm]], [[phase-5-vector-unit]]*

### SiLU
**An activation function: x times its own sigmoid, x·σ(x).**

Blends smoothly between roughly zero for negative inputs and roughly linear for positive ones, unlike ReLU's sharp corner. `stories260K` uses it inside SwiGLU, and since it involves an exponential, it's a natural fit for a small lookup table in hardware.

```
SiLU(x) = x * sigmoid(x) = x / (1 + e^-x)
```

*Used in: [[phase-5-vector-unit]]*

### SwiGLU
**The feed-forward activation in `stories260K`: SiLU applied to a "gate" projection, multiplied element-wise by an "up" projection.**

Computes two projections, `gate` and `up`, applies SiLU to `gate`, and multiplies element-wise by `up` — gating that tends to outperform a plain ReLU feed-forward per parameter, at the cost of one extra matmul.

```
SwiGLU(x) = SiLU(x @ W_gate) ⊙ (x @ W_up)
```

*Used in: [[phase-5-vector-unit]]*

### RoPE
**Rotary position embedding: encodes a token's position by rotating pairs of dimensions in its query and key vectors.**

Each pair of dimensions in a query or key is rotated by an angle proportional to the token's position — token 5 rotates differently than token 12. The dot product between a rotated query and key then depends only on their relative position, exactly what attention needs.

*Used in: [[phase-1-baseline-llm]], [[phase-5-vector-unit]]*

### Lookup table
**Using a small precomputed table of outputs, indexed by a quantized input, instead of computing an expensive function directly.**

Sigmoid, exp, and 1/sqrt(x) show up throughout SiLU, softmax, and RMSNorm, none cheap in fixed-point. You precompute the function at a few points, store results in BRAM, and index at runtime. Don't confuse this with the FPGA LUT primitive — it's an algorithmic technique, not the hardware element it's built from.

*Used in: [[phase-5-vector-unit]]*

### Perplexity
**A single number measuring how well the model predicts held-out text; lower is better.**

The exponential of the average negative log-likelihood assigned to the correct next token — roughly, how many equally-likely tokens it was choosing among when it got it right. It's the standard check that quantizing to int8 hasn't broken the model: compare before and after, expect a small increase (under 1% here).

```
perplexity = exp( -mean(log P(correct token)) )
```

*Used in: [[phase-2-numerics]], [[phase-7-end-to-end]]*

## Still confused?

That's expected — a glossary entry is a definition, not a lesson. Each phase note walks through these ideas with numbers specific to `stories260K` and this board. Follow a term's *Used in* link and read it in context; watching a Q-format multiply happen in real Verilog, or a KV cache fill up during generation, tends to make it click.

<script src="/tutor.js" defer></script>
