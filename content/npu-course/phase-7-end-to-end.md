---
title: "Phase 7 — End to End"
description: "KV cache budgeting, a fully on-chip language model, and the DDR-streamed model that hits the memory wall."
tags:
  - npu
  - hardware
  - capstone
---

# Phase 7 — End to end

**5 weeks** · Part of [[index|Zero to NPU]] · Prev: [[phase-6-sequencer]] · Next: [[phase-8-optimization]]

> [!info] What this phase is for
> Every phase since Phase 2 has produced one isolated block: fixed-point math, a MAC array, a memory pipeline, a vector unit, a sequencer that runs without hand-holding. None of them, alone, generates a sentence. This phase wires everything into one loop that takes a token in and produces the next token out, over and over, until it's telling a story. Both capstones live here, and they're deliberately different lessons: Tier 1 proves your design works end to end on hardware small enough to hide entirely on-chip; Tier 2 proves that past a certain size, model performance on this board stops being a compute problem and becomes a memory problem, no matter how clever your RTL is. Expect a new kind of frustration here — not "does this block work" but "why does the *combination* produce garbage" — so budget real time for it. When it works, it's the best moment in the course: an FPGA you built from LUTs and DSPs, telling a coherent little story, powered by silicon you understand from the transistor level to the token level.

---

## Bringing it all together

Before you write a line of integration code, take stock of what you actually have. By the end of Phase 6 your design contains five working blocks, and this phase's entire job is routing one token through all five of them, in order, every step:

- **The PE array** ([[phase-3-mac-engine]]) — an 8×4 grid of DSP-backed MAC cells doing every matmul: Q/K/V projections, the attention-weighted sum, the FFN, the logits projection.
- **The memory pipeline** ([[phase-4-memory]]) — your AXI4 burst master and double-buffered BRAM tiles, keeping the array fed without stalling on DDR3's latency.
- **The vector unit** ([[phase-5-vector-unit]]) — dedicated hardware for RMSNorm, RoPE, softmax, SwiGLU. Small in operation count (Phase 1.2: only a few percent of total work), built for correctness, not speed.
- **The command sequencer** ([[phase-6-sequencer]]) — the FSM that walks a descriptor chain in DDR and drives a whole layer unattended, turning "the PS pokes registers all day" into "the PS writes one doorbell and sleeps."
- **The KV cache** (this phase, §7.1) — new here. It's what lets each new token's attention step see *every prior token's* Key and Value vectors without recomputing them.

Integrating them means one thing operationally: the PS builds a descriptor chain for one full transformer layer (all seven steps from [[phase-1-baseline-llm|Phase 1 - Baseline LLM on ARM]]), rings the doorbell, and the sequencer drives the array, the vector unit, and the KV cache through that chain unattended, five times (once per layer), for every single token you generate.

```
PS (Cortex-A9)                          PL (fabric)
   │
   │ 1. build descriptor chain for
   │    layer 0..4 of THIS token
   │    (writes to DDR)
   │
   │ 2. doorbell write ──────────────▶  Sequencer (Ph.6)
   │                                        │ walks descriptor chain
   │                                        ▼
   │                                   Burst master (Ph.4)
   │                                        │ fetches weight/activation
   │                                        │ tiles into double-buffered BRAM
   │                                        ▼
   │                                   PE array (Ph.3)
   │                                        │ matmul: Q/K/V, attn sum, FFN
   │                                        ▼
   │                                   Vector unit (Ph.5)
   │                                        │ RMSNorm → RoPE → softmax → SwiGLU
   │                                        ▼
   │                                   KV cache (§7.1)
   │                                        │ write this token's K,V;
   │                                        │ read all prior tokens' K,V
   │                                        ▼
   │                                  (repeat for layers 1,2,3,4)
   │                                        │
   │                                   logits out (512 scores)
   │
   │ ◀── 3. interrupt on completion ────────┘
   │
   │ 4. sample next token (cheap — PS does this)
   │ 5. append token, go to step 1
```

Nothing in this diagram is new hardware. It's a routing exercise: making sure descriptor opcodes actually reach the right block, that the sequencer's chain covers all seven per-layer steps in the right order, and that the KV cache is read and written at the right point in that chain. If you get lost integrating, come back to this diagram before you touch RTL — most Phase 7 bugs are "the sequencer skipped a step" or "the KV cache was read before it was written for this token," not new arithmetic bugs.

---

## 7.1 The KV cache

### What a KV cache actually is

Look back at step 4 of the per-layer loop in [[phase-1-baseline-llm]]: attention scores compare the *current* token's Query vector against *every previous token's* Key vector, then blend every previous token's Value vector using those scores. That's the whole mechanism — "attention" is just "look back at everything so far and weigh it by relevance."

The naive way to implement that is to recompute Key and Value for every previous token, every single time you generate a new one. Walk through it concretely, generating three tokens of a story:

- **Token 1** ("Once"). No history yet. Compute Q₁, K₁, V₁ from the embedding of "Once," attend only to itself, then cache K₁ and V₁ at "position 0" — you'll need them for every future token.
- **Token 2** ("upon"). Compute Q₂, K₂, V₂ fresh — unavoidable, it's a new token. But attention needs Key vectors for both tokens so far: K₁ (already cached, untouched) and K₂ (new). Compare Q₂ against both, softmax, blend V₁ (cached) and V₂ (new). Write K₂, V₂ to position 1. Cache now holds two entries.
- **Token 3** ("a"). Same pattern: fresh Q₃, K₃, V₃; attention compares Q₃ against K₁, K₂ (both cached, zero recomputation) and K₃ (new); write K₃, V₃ to position 2.

Notice the pattern: **generating token N always costs exactly one new Q/K/V computation, no matter how long the story has gotten.** Without the cache, generating token N would mean re-running the *entire prefix* through the Q/K/V projections from scratch just to reconstruct Keys and Values you'd already computed correctly the first time — work that grows every token, summing to roughly the *square* of the sequence length across a full generation (the "O(N²) waste" you'll see referenced in ML literature — you don't need the math, just the shape: quadratic cost for something that should be linear). The cache turns that back into linear cost, for the price of some on-chip memory, and changes nothing about *what* gets computed, only how much of it you redo.

One more thing worth being explicit about, because it trips people up: the cache stores **Key and Value only**, never Query. Query is used once, immediately, for the current token's attention step, and then discarded — you never need token 5's Query vector again once token 5's attention output has been computed. Key and Value are what future tokens need to look back at, which is exactly why they're the two things worth keeping around.

### The budget arithmetic — this decides your context length

Every cached entry costs real BRAM, and BRAM is the scarcest resource on the [[board-specs|XC7Z007S]] (225 KB total, 50 blocks of 36 Kb / 4.5 KB each). So before you write a line of KV-cache RTL, you need to know how big the cache gets as a function of how many tokens of story you want the model to remember — its **context length**.

The formula, in bytes:

```
KV cache size = seq_len × n_layers × 2 × kv_dim
```

Walk through where each factor comes from. `seq_len` is how many tokens of context you want to support. `n_layers = 5` because `stories260K` has 5 transformer layers, and — critically — **each layer keeps its own independent KV cache**. Layer 2's attention only ever looks at layer 2's own cached Keys and Values, never layer 3's. The `2` is for K *and* V — two separate vectors get cached per token per layer. And `kv_dim = 32` is the size of one K (or V) vector: `n_kv_heads=4` heads × `head_size=8` = 32. (If you're wondering why it's `n_kv_heads` and not `n_heads=8` — this checkpoint uses grouped-query attention, where multiple Query heads share the same Key/Value heads. You don't need to understand why for this course; you just need `kv_dim=32` as the number that goes in the formula.)

Work the arithmetic for a 128-token context, the row you'll actually build:

```
128 × 5 × 2 × 32 = 40,960 bytes ≈ 40 KB
```

(You'll see this rounded to 41 KB elsewhere in the course material — that's just block-granularity rounding, and it doesn't change the decision below.)

Now the full table, showing why 128 is the number you land on and not something more generous:

| Context | KV size | BRAM blocks | Weights (29) + KV | Left for buffers |
|---|---|---|---|---|
| 512 | 164 KB | 37 | 66 / 50 | **doesn't fit** |
| 256 | 82 KB | 18 | 47 / 50 | 3 — **too tight** |
| **128** | **41 KB** | **9** | **38 / 50** | **12 — works** ✅ |

512 tokens simply doesn't fit — 66 blocks needed against 50 available, full stop. 256 fits arithmetically, but 3 spare blocks isn't enough for the double-buffering ([[phase-4-memory|§4.2]]) and sequencer descriptor-queue storage both need. 128 tokens — roughly 100 words of story — leaves 12 free blocks, enough headroom for both.

> [!important] Decision: cap context at 128 tokens
> This is one of the few genuinely load-bearing engineering decisions in the whole course, and it's worth being able to defend in an interview: "why 128, and not 256 or 512?" The honest answer is arithmetic, not intuition — walk through the table above.

### Laying the cache out in BRAM

Once you know *how much* memory you need, you need an addressing scheme that tells you exactly *where* one specific (layer, K-or-V, token, element) entry lives. Get this wrong and you'll read someone else's cache entry — the single most common Tier-1 bug (see §7.2's troubleshooting).

Two things happen to the cache during generation, and the layout should favor whichever one needs to be simple:

- **Writing** happens once per token, per layer: one new K vector and one new V vector (32 elements each) at the end of that layer's attention step.
- **Reading** happens every attention step, and it's a *sweep*: every previously-cached token's K vector (to score against), then every V vector (to blend). This is the more frequent, more data-heavy operation, so it's the one worth making cheap — a plain incrementing address, not a scattered one.

That means: for a fixed layer and K-or-V selection, consecutive token indices should sit at consecutive addresses, so a full-history sweep is one contiguous run with an up-counter.

```
kv_addr = layer_idx × (2 × MAX_SEQ_LEN × KV_DIM)     ← which layer's region
        + kv_select × (MAX_SEQ_LEN × KV_DIM)          ← K block or V block within that layer
        + token_idx × KV_DIM                          ← which token's vector within that block
        + elem_idx                                    ← which of the 32 elements within that vector
```

Reading it fastest-to-slowest: `elem_idx` varies fastest (the 32 elements of one K or V vector are contiguous), then `token_idx` (so a full-history sweep at attention time is a simple counter), then `kv_select` (K block, then V block), then `layer_idx` as the outermost, coarsest dimension (since a whole layer's cache is only touched during that layer's own step in the descriptor chain).

```verilog
// KV cache address generator.
// Produces a flat BRAM address from (layer, K/V, token, element) indices,
// laid out so a fixed-layer, fixed-K/V sweep over all cached tokens is a
// simple sequential address run -- the access pattern attention actually uses.
module kv_addr_gen #(
    parameter KV_DIM      = 32,   // elements per K or V vector (n_kv_heads * head_size)
    parameter MAX_SEQ_LEN = 128,  // context cap decided above
    parameter N_LAYERS    = 5
)(
    input  [2:0]  layer_idx,   // 0..4, which transformer layer
    input         kv_select,   // 0 = K block, 1 = V block
    input  [6:0]  token_idx,   // 0..127, which cached token
    input  [4:0]  elem_idx,    // 0..31, which element of the K/V vector
    output [16:0] kv_addr      // flat BRAM address
);
    // One layer's full K+V region, in elements:
    localparam LAYER_STRIDE = 2 * MAX_SEQ_LEN * KV_DIM;
    // K block and V block are equal-sized halves of that region:
    localparam KV_STRIDE    = MAX_SEQ_LEN * KV_DIM;

    wire [16:0] layer_base = layer_idx * LAYER_STRIDE;   // jump to this layer's region
    wire [16:0] kv_base    = kv_select ? KV_STRIDE : 17'd0; // jump to K half or V half
    wire [16:0] token_base = token_idx * KV_DIM;          // jump to this token's vector

    assign kv_addr = layer_base + kv_base + token_base + elem_idx;
endmodule
```

> [!success] Done when
> You can write a synthetic K/V vector for token 0 of every layer, then token 1 of every layer, and so on up through token 20 or so, and read every one of them back from the exact address your formula predicts by hand — not just "it looks right," but a value-by-value check against addresses you computed on paper. Then verify against the golden model ([[phase-1-baseline-llm|§1.4]]) across one full 128-token generation: your hardware's cached K/V values at every step should match the reference bit-for-bit.

> [!question]- It's not working
> 1. **Off-by-one on `token_idx`** — are you writing this token's K/V *before* or *after* incrementing the position counter? The token you're currently generating hasn't been cached yet when its own attention step reads history; make sure you read positions 0..N-1, then write position N, in that order, not the reverse.
> 2. **K/V swapped** — check `kv_select` is wired the way you think; a K vector read out of the V half (or vice versa) produces attention scores that look plausible but are subtly wrong, which is a nastier bug than an obvious crash.
> 3. **Layer index bleeding into another layer's region** — verify `LAYER_STRIDE` is computed from the *full* per-layer region (`2 × MAX_SEQ_LEN × KV_DIM`), not just the K or V half; a factor-of-2 error here means layer 1 silently overwrites the back half of layer 0's cache.
> 4. **Stale cache across a fresh generation run** — if you don't reset or track a "valid length" per run, restarting generation may read leftover K/V values from the *previous* run's later tokens as if they were real history.
> 5. **Address width** — `MAX_SEQ_LEN × N_LAYERS × 2 × KV_DIM = 128 × 5 × 2 × 32 = 40,960` addressable elements; confirm your address bus is wide enough (17 bits comfortably covers this) and that you're not silently truncating.

---

## 7.2 Tier 1 capstone — fully on-chip

This is the headline result of the entire course: `stories260K`, quantized to 4-bit, resident entirely in on-chip BRAM, generating text with **zero DDR traffic during generation**. Not "mostly on-chip" — zero. Every weight the array touches for every matmul, for every layer, for every token, comes from BRAM you loaded once at startup.

### What "4-bit weights" means, concretely

Through [[phase-2-numerics]] and [[phase-3-mac-engine]] you've used int8 weights — one byte each, roughly -127 to 127. 4-bit pushes further: each weight is only 4 bits, 16 distinct values (typically -8 to 7). Coarser, yes — but it halves your BRAM footprint, which is the resource under pressure. Two 4-bit weights pack into one byte: high nibble holds the first weight, low nibble the next.

That packing is cheap to store but not free to *use* — a PE expects a full signed integer, not half a byte. Something has to unpack each nibble back into a signed value before it reaches the array. [[phase-8-optimization|Phase 8.1]] covers exactly where that logic lives and why; for this capstone, treat it as a black box and focus on getting the end-to-end system generating text at all.

### The block budget, and the embedding-table tradeoff

The whole `stories260K` model — the 5 transformer layers (227K params) plus the token embedding table (33K params), all of it, packed at 4-bit — comes to roughly 130 KB, which is **29 BRAM blocks**. Add the 9 blocks the KV cache needs at 128-token context (§7.1) and you're at 38 of 50 blocks, with 12 left over for weight/activation double-buffering and the sequencer's descriptor-queue storage. That's the budget this course's Tier-1 capstone is built around, and it's tight but it works, which is the whole point of doing the arithmetic in §7.1 before writing RTL.

Worth knowing as you build: that 29-block figure assumes the *entire* model, embedding table included, is resident. [[board-specs]] documents a stricter accounting once you add up what the vector unit, burst master, and sequencer actually consume for their own buffers — less slack than the simplified 29+9 arithmetic assumes. If you end up short a few blocks once everything else is placed, the embedding table is the natural thing to push to DDR: unlike every weight matrix, it's touched by a single-row gather per token (not a matmul sweep), so it's the cheapest thing to fetch off-chip. That trades the "zero DDR traffic" headline for "one small read per token" — a legitimate call, just document which version you built.

### A staged bring-up order — don't integrate everything at once

Wiring five blocks together and expecting correct English on the first try is not a realistic plan; if you do it that way, when it fails (and it will), you'll have no idea which of five subsystems is at fault. Bring it up in stages, and validate each one against the golden model ([[phase-1-baseline-llm|§1.4]]) before moving to the next:

1. **Single matmul.** Feed one known weight tile and activation tile through the array at 4-bit, requantize, and compare against the golden model's output bit-for-bit. Smallest unit that exercises unpacking, the array, and requantization together.
2. **One full layer, one token.** Chain all seven per-layer steps — RMSNorm, Q/K/V, RoPE, scores, softmax, weighted sum, FFN+residual — with a KV cache of exactly one entry (self-attention only). Compare the layer output against golden.
3. **All five layers, one token.** Chain layer 0's output into layer 1, through layer 4. Compare final logits against golden — this is the first stage that would catch a layer-*ordering* bug (wrong RMSNorm placement, a skipped residual) that single-layer testing can't.
4. **Full generation loop, 128 tokens.** Let the KV cache actually grow across tokens, sampling and feeding back each one, per the "Bringing it all together" diagram. Compare token-by-token against a golden run using greedy/argmax sampling, so both runs are deterministic and directly comparable.

Each stage should be strictly smaller and more debuggable than the next. Don't skip one because "it'll probably be fine" — stage 3 versus stage 4 is exactly what lets you tell "my per-layer math is wrong" apart from "my KV addressing across tokens is wrong."

> [!success] Done when
> Coherent English stories stream out the UART at high speed, matching the golden model's output when run in deterministic (argmax) sampling mode, with the DDR-traffic counter from [[phase-4-memory|§4.4]] reading zero for the entire generation.

> [!question]- The output is garbage — where to look, roughly in order of how often each one is actually the culprit
> 1. **Wrong requantization shift.** Every matmul's int32 accumulator has to be rescaled back to 4-bit range; get the shift off by even one bit and everything downstream is scaled wrong — uniformly saturated or uniformly near-zero output is the classic symptom. Re-derive the shift from your quantization scales ([[phase-1-baseline-llm|§1.3]], extended to 4-bit) rather than guessing.
> 2. **KV cache addressing off by one.** The single most common Tier-1 bug. Check: do you read history *before* writing the current token's own K/V, and does `token_idx` at step N actually equal N? Revisit §7.1's troubleshooting — this bug looks like grammatically-plausible garbage, not obvious noise, because attention is still running, just over the wrong tokens.
> 3. **RoPE applied to V by mistake.** RoPE rotates Q and K only, never V. A copy-pasted call site rotates V too, and the corruption is subtle — attention weights stay sane, but the content being blended is wrong.
> 4. **Softmax overflow.** If you exponentiate raw scores without subtracting the max first, large scores overflow silently and attention collapses onto one token regardless of relevance. Use the numerically-stable form (subtract max before exponentiating).
> 5. **Weight-unpacking nibble order swapped.** If unpack reads "first weight" from the opposite nibble your packing wrote it to, every weight is consistently but incorrectly wrong — check one known value before assuming the whole model is broken.
> 6. **Tokenizer mismatch.** If your on-chip token IDs use a different vocabulary ordering than the golden reference's tokenizer, everything downstream computes correctly on the *wrong* input. Print raw token IDs for a known string and compare against the Python-side tokenizer before chasing the math pipeline.

---

## 7.3 Tier 2 capstone — the memory wall

Tier 1 proved your design works end to end. Tier 2 proves what Tier 1 can't: what happens when the model doesn't fit on-chip anymore. Same NPU — same array, vector unit, sequencer, not one gate changed — running `stories15M`, roughly **60× larger** than `stories260K` by parameter count. At int8, `stories15M` is about 15 MB, nowhere close to fitting in 225 KB of BRAM. So instead of loading once at startup, every weight gets streamed from DDR3, tile by tile, every token — the same double-buffered pipeline from [[phase-4-memory]], just running continuously instead of once.

### The bandwidth arithmetic, worked explicitly

Generating one token means the array touches essentially the *entire* 15 MB of weights (unlike KV-cache reuse, a matmul over a full weight matrix has no shortcut — it reads every weight at least once). At your measured sustained DDR3 bandwidth — **~1.2 GB/s**, the same planning number from [[phase-4-memory|§4.4]] — moving 15 MB takes:

```
time per token = 15 MB / 1.2 GB/s = (15 × 10⁶ bytes) / (1.2 × 10⁹ bytes/sec) ≈ 0.0125 s = 12.5 ms
```

Invert that to get a throughput ceiling:

```
throughput ceiling = 1 / 0.0125 s ≈ 80 tokens/sec
```

That's not a target to hit — it's a hard ceiling set entirely by the DDR3 bus. No amount of PE array cleverness, no amount of pipelining, no faster clock on the fabric side changes this number, because the array spends most of every token cycle *waiting* for weight bytes to arrive, not computing. This is the whole lesson made numeric.

### What to measure, and how to present it

Instrument the same way you did for the roofline exercise: a hardware cycle counter gated on "burst master actively moving bytes," so you can separate genuine compute time from memory-wait time. Run a real generation and record:

- **Measured tok/s** for `stories15M` end to end.
- **Measured sustained DDR3 bandwidth** during that run, which should land close to your Phase 4 measurement — if it doesn't, investigate that before trusting anything else here.
- Plot both capstones on the same roofline from [[phase-4-memory|§4.4]]: Tier 1 sits to the right of the ridge point (high intensity, zero DDR traffic, compute-bound); Tier 2 sits near the memory-bound diagonal, well to the left of it. Seeing both on one plot is more convincing than either number alone.

> [!success] Done when
> `stories15M` generates coherent text, and your measured tok/s lands within roughly 2× of the ~80 tok/s bandwidth ceiling you computed above. If you're far under that (say, below 40 tok/s), suspect burst sizing or 4 KB-boundary splitting eating more of your DDR transfers than expected, the same way [[phase-4-memory|§4.4]]'s troubleshooting describes — not a "your NPU is bad" problem.

> [!warning] This is your headline lesson, not a disappointing result
> Same NPU, 60× the model, and throughput is now set **entirely** by a 16-bit DDR3 bus running at a sustained ~1.2 GB/s. ==No amount of added DSPs would help.== If you doubled your PE array to 64 PEs tomorrow, `stories15M`'s throughput would not move, because the array was never the bottleneck to begin with — it was starved, waiting on memory, the entire time. That sentence is the whole reason NPU design is fundamentally a memory-bandwidth problem wearing a compute problem's clothes, and the difference between Tier 1 and Tier 2 is the cleanest possible demonstration of it: identical silicon, and the only variable that changed the outcome was whether the model fit next to the compute or had to commute to it. You've now measured that yourself, on real hardware, instead of reading it as someone else's claim.

---

## Check your understanding

> [!question]- Why does a KV cache turn quadratic work into linear work, and what specifically gets cached?
> > [!success]- Answer
> > Without a cache, generating token N recomputes Key and Value for all N-1 previous tokens from scratch — work that grows every token, summing to roughly O(N²) across a generation. The cache stores each token's K and V the one time they're computed and just reads them back later. Query is never cached — it's used once, immediately, and discarded.

> [!question]- Work out the KV cache size, in bytes and BRAM blocks, for a hypothetical 64-token context. Use the same formula as §7.1.
> > [!success]- Answer
> > `64 × 5 × 2 × 32 = 20,480 bytes ≈ 20 KB`. At 4.5 KB per block, that's `20,480 / 4,608 ≈ 4.4`, rounding up to **5 blocks** — about half of the 128-token version's 9, since KV size scales linearly with context length.

> [!question]- The Tier 1 capstone reports "zero DDR traffic during generation." Why can't Tier 2 ever make that same claim, no matter how well it's optimized?
> > [!success]- Answer
> > Tier 1's entire 260K-parameter model fits inside 225 KB of BRAM at 4-bit, so once loaded at startup, generation never touches DDR again. Tier 2's model is 15 MB at int8 — 60× the entire BRAM budget — so it physically cannot be resident. Every weight has to be fetched from DDR3 every token, because there's nowhere else for 15 MB to live on a chip with 225 KB of block RAM. Not a solvable inefficiency — a direct consequence of size versus capacity.

> [!question]- Why validate against the golden model at each stage of §7.2's bring-up order, rather than only checking the final story for coherence?
> > [!success]- Answer
> > "The story looks roughly okay" can't localize a bug — several different bugs (wrong requant shift, RoPE on V, a KV addressing error) all produce plausible-but-wrong text, not obvious garbage. Checking matmul, then one layer, then all five, then the full loop against golden at each stage means a mismatch always points to the one piece of complexity you just added.

---

## What's next

You now have a complete, working NPU that generates real text — both the "everything fits" story and the "here's exactly why the bigger model doesn't" story, both measured on real silicon. [[phase-8-optimization]] takes this working system and makes it *good*: closing timing to push your clock higher, tightening the 4-bit path, measuring power so you can make the actual case for dedicated inference hardware, and writing all of it up as something you could put in a portfolio.
