---
title: "Module 9 — KV Quantization and Its Specific Danger"
description: "Proving the flash-attention dependency behind KV cache quantization from block layout first principles, and measuring why quantizing the KV cache is riskier for agentic tool-calling than quantizing the weights."
tags:
  - local-ai
  - kv-cache
  - quantization
  - flash-attention
  - agentic
---

# Module 9 — KV Quantization and Its Specific Danger

**6 hours** · Part of [[index|Two Pools]] · Prev: [[module-8-quantization]] · Next: [[module-10-serving]]

> [!info] What this module is for
> [[module-8-quantization|Module 8]] quantized the weights and found a plateau — pass rate held flat from BF16 down through Q4_K_M on the cited benchmarks, with the real cliff much further down the ladder. It would be reasonable to assume the [[concepts#KV cache|KV cache]] behaves the same way, since it’s "just another tensor to quantize." It doesn’t. This module has two jobs. First, prove — don’t just assert — a real mechanical constraint: quantizing the V half of the KV cache requires flash attention, for a reason rooted in how the V cache is physically laid out in memory, not an arbitrary software restriction. Second, and more important: establish that **KV-cache quantization is a genuinely different risk profile than weight quantization**, specifically for agentic tool-calling, where documented failures at long context are more severe and more model-dependent than anything Module 8’s weight ladder produced. If you take away one sentence from this module, it should be the one in the ground rules: if you must cut memory, quantize weights before you quantize the KV cache.

---

## 9.1 The flags: `-ctk`/`-ctv` and what they accept

> [!abstract] Goal
> Know the exact flag names, their default, and the full list of accepted cache types before touching either one.

**What’s going on:** `llama.cpp` exposes K-cache and V-cache quantization as two **independent** flags — `--cache-type-k` / `-ctk` and `--cache-type-v` / `-ctv` — both defaulting to `f16`. Accepted values for either flag: `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`. That K and V are independent flags, not a single combined "KV quant level," matters for the whole rest of this module — you can quantize K aggressively while leaving V untouched, and (as 9.3 proves) there’s a real mechanical reason you might be forced to do exactly that.

**Steps:**
1. Run `llama-server --help` and confirm `-ctk` and `-ctv` are present as separate flags, and check the exact accepted-value list your build reports — value lists can drift release to release, so confirm rather than trusting this module’s restated list.
2. Load the model once with both flags left at their `f16` default, and once with `-ctk q8_0 -ctv f16` (K quantized, V untouched), and confirm from the startup logs that only the K-cache tensors report a reduced size.

> [!success] Done when
> You can name both flags, their default, and the accepted-value list from memory, and you’ve confirmed from startup logs that K and V can be set independently.

## 9.2 Memory math from the actual block layouts

> [!abstract] Goal
> Derive the real bytes-per-element figure for `f16`, `q8_0`, and `q4_0` from each format’s block layout, not from a rule of thumb.

**What’s going on:** `f16` stores each value at a flat 2.0 bytes/element — no blocking, no per-block metadata. The quantized types are block-based, and the metadata overhead per block is what keeps the effective bytes/element above the naive "N bits ÷ 8" figure:

- **`q8_0`**: 32-weight blocks, 34 bytes/block (32 bytes of 8-bit values + a 2-byte fp16 scale) → `34 / 32 = 1.0625` bytes/element → a **46.9% reduction** versus `f16`’s 2.0 bytes/element.
- **`q4_0`**: 32-weight blocks, 18 bytes/block (16 bytes of packed 4-bit values + a 2-byte fp16 scale) → `18 / 32 = 0.5625` bytes/element → a **71.9% reduction** versus `f16`.

Applied to the model’s actual KV-cache figure from [[module-4-vram-budget|Module 4]] — 98,304 bytes/token at `f16` (4 KV heads × 128 head_dim × 48 layers × 2 (K and V) × 2 bytes) — quantizing **both** K and V to `q8_0` brings that down to:

```
98,304 × (1.0625 / 2.0) = 52,224 bytes/token   (53.1% of the f16 footprint)
```

And both K and V at `q4_0`:

```
98,304 × (0.5625 / 2.0) = 27,648 bytes/token   (28.1% of the f16 footprint)
```

**Steps:**
1. Recompute the `q8_0` and `q4_0` bytes-per-element figures yourself from the block byte counts above, rather than copying the 1.0625/0.5625 results.
2. Recompute the resulting bytes/token figures for both-K+V-at-`q8_0` and both-K+V-at-`q4_0`, and confirm the 53.1%/28.1%-of-`f16` figures independently.
3. Compute the **context-length multiplier** each quant level buys for a fixed VRAM budget devoted to KV cache alone: `1 / (1 − reduction)`. You should get roughly **1.88×** more context at `q8_0`/`q8_0` and roughly **3.56×** more context at `q4_0`/`q4_0`, for the same bytes spent on cache.

> [!success] Done when
> You have your own derivation of both bytes-per-element figures from block layout, the resulting KV-cache bytes/token at each quant level, and the predicted context-length multiplier each one buys.

> [!bug] Gotcha
> The context-length multiplier in step 3 is a **predicted** headroom figure from the memory math alone — it says nothing about whether the model’s *quality* holds up at that much longer context under a quantized cache. 9.4 and 9.6 are where you find out whether that headroom is actually usable.

## 9.3 Proving the flash-attention constraint

> [!abstract] Goal
> Don’t take "V-cache quantization requires flash attention" on faith — reproduce the failure yourself, then understand the actual mechanical reason for it.

**What’s going on:** This is the kind of constraint that’s easy to state and easy to skip verifying, so verify it directly instead.

**Steps:**
1. Launch the server with V-cache quantization forced on and flash attention forced off:
   ```bash
   ./build-hip/bin/llama-server -m qwen3-coder-30b-a3b-q4_k_m.gguf \
       -ctv q8_0 -fa off -c 8192
   ```
2. Expect this to fail at load time, with an error to the effect of "V cache quantization requires flash_attn" — the **exact wording is build-specific and may differ**, so treat the phrase above as a description of the failure mode, not a transcript to match character-for-character. Copy your own build’s actual error text into your notes.
3. Now retry with K-only quantization and V left at `f16`, still with flash attention off:
   ```bash
   ./build-hip/bin/llama-server -m qwen3-coder-30b-a3b-q4_k_m.gguf \
       -ctk q8_0 -ctv f16 -fa off -c 8192
   ```
   This should load successfully — confirming K-only quantization has no flash-attention dependency, while V-cache quantization does.
4. Now understand *why*, rather than just accepting the asymmetry: without flash attention, the V cache is stored **transposed** relative to how it would be laid out with FA enabled. Block-quantized formats like `q8_0`/`q4_0` need to dequantize **contiguous, block-aligned rows** to reconstruct values efficiently — a transposed layout breaks that contiguity for V, since the block boundary and the transpose stride no longer line up. K is **never transposed**, under either attention path, so K-only quantization has no equivalent layout conflict to hit, with or without flash attention.

> [!success] Done when
> You’ve reproduced the `-ctv` + `-fa off` failure on your own build, captured your build’s actual error text, confirmed K-only quantization loads fine without flash attention, and can explain the transposed-V-cache mechanism in your own words without looking it up.

> [!question]- It’s not working
> 1. The `-ctv q8_0 -fa off` combination *loads successfully* on your build instead of failing? Check your `llama.cpp` build date — this constraint’s enforcement (and its exact error message) has shifted across versions, and it’s possible your build silently forces flash attention on internally, or silently falls back to an unquantized V cache. Check the startup log carefully for either behavior before assuming the constraint doesn’t apply to your version.
> 2. K-only quantization *also* fails without flash attention on your build? That would be a genuine deviation from the mechanism described above, worth filing as its own finding — note your exact build commit before concluding the general rule is wrong.

> [!bug] Gotcha
> `-ctv` quantization without `-fa on` (or `-fa auto` resolving to on) is the single most common way to get a confusing load-time failure in this module. If you only remember one flag pairing rule from this section, make it: **quantizing V always means flash attention is in play, one way or another.**

## 9.4 Quality evidence: near-lossless at q8_0, model-dependent collapse at q4_0

> [!abstract] Goal
> Know the cited third-party accuracy data for KV-cache quantization, clearly separated from your own upcoming measurement in 9.6.

**What’s going on:** These are other researchers’ reported numbers, on their own models and hardware — cited here as the shape you’re checking your own tool-call measurement against, not as a number you’re allowed to write down as your result.

A precision-focused measurement reported **`-ctk q8_0 -ctv q4_0` at roughly 1.3% precision loss versus full `f16`** — a small, seemingly comfortable number on its own.

A broader ARC-Challenge evaluation, run across 500 questions on each of four different models, tells a sharper story:

- **`q8_0`/`q8_0`** changed only **0 to 4 of 500 answers** across all four models — consistent, near-lossless behavior regardless of which model was tested.
- **`q4_0`/`q4_0`** was **strongly model-dependent**: one model changed only 2 of 500 answers, while another collapsed from 92% accuracy down to 24.2% — **375 of 500 answers changed**, a near-total breakdown, on the exact same quant setting that left a different model essentially untouched.

The takeaway from this pairing of results is specific, not a blanket "q4 KV cache is risky": **`q8_0` is broadly safe to assume near-lossless across models. `q4_0` must be benchmarked per-model — it cannot be assumed safe just because it worked on a different model, even a similar one.**

**Steps:**
1. Write down, in your own words, why "1.3% average precision loss" and "375 of 500 answers changed on one specific model" can both be true statements about the same general quant setting — think about what an average across models hides.
2. State explicitly which of the two cited claims — the 1.3% figure or the ARC-Challenge model-dependence finding — you’d trust more as a planning input for your own model, and why.

> [!success] Done when
> You can restate both cited findings from memory, correctly attributed as third-party measurements, and explain why "model-dependent" is the load-bearing qualifier on the `q4_0` result specifically.

## 9.5 The specific danger: KV quantization is riskier than weight quantization, for tool calls

> [!abstract] Goal
> Understand why this module’s thesis singles out KV-cache quantization as a distinct risk from weight quantization — and why the risk shows up specifically in tool-calling, not in aggregate benchmark scores.

**What’s going on:** Module 8’s weight-quantization ladder held remarkably flat on cited benchmarks down through Q4_K_M. KV-cache quantization does not get the same pass, and the documented failure mode is specific enough to name directly: **agentic tool-calling breaks under KV-cache quantization in ways that weight quantization, at comparable precision, does not.**

Documented cases worth taking seriously, all cited as other researchers’ or practitioners’ findings on their own setups:

- **INT4 KV cache** has produced **reproducible tool-call failures at long context** — not occasional flakiness, but a repeatable breakdown as context grows.
- **NVFP4** KV-cache quantization caused **roughly 50% of tokens to flip by around 88K context** in a reported evaluation — a scale of degradation that would show up as broken tool arguments, malformed calls, or outright wrong tool selection well before it showed up as a noticeably worse aggregate score.
- **AWQ W4A16** — a *weight*-quantization scheme, included here deliberately as a contrast — was reported to **fail to close tool calls correctly**, showing that weight quantization isn’t universally safe either, but the failure modes cited for weight quantization schemes in this space are comparatively rarer and less severe than the KV-cache cases above.
- **FP8 and INT8 weight quantization**, by contrast, **stayed reliable** for tool-calling in the same body of reporting.

The pattern across these cases: **precision loss in the KV cache degrades the model’s ability to track and correctly close tool-call state as context accumulates, in a way that’s more severe, and more prone to hard collapse, than precision loss in the weights at a comparable bit width.** This is not a claim that weight quantization is risk-free — Module 8 already showed a real cliff exists past Q2_K_XL — it’s a claim that **the KV cache hits that kind of cliff at a shallower quant level, and specifically in tool-calling behavior, which aggregate pass-rate benchmarks can miss entirely** if they aren’t testing tool-call validity as its own metric.

**The rule this module exists to establish:** if you need to reclaim memory and have to choose which pool to quantize harder, **quantize weights first, and treat KV-cache quantization — beyond `q8_0`/`q8_0` — as the higher-risk lever**, to be reached for only after weight quantization headroom is exhausted, and only with your own tool-call-specific measurement in hand.

**Steps:**
1. Write down, in one sentence each, why INT4 KV cache and NVFP4 KV cache both surfaced as *tool-calling* failures specifically, rather than as generic accuracy drops on a knowledge benchmark.
2. State, in your own words, why "quantize weights before you quantize the KV cache" follows from the evidence in this section, rather than being an arbitrary ordering preference.

> [!success] Done when
> You can state the module’s thesis sentence from memory — KV-cache quantization is riskier than weight quantization for agentic tool-calling — and back it with at least two of the cited documented cases.

## 9.6 Measure: context gained vs. tool-call failure rate

> [!abstract] Goal
> Build a small tool-call validity test, run it at several KV-cache quant configurations and context lengths, and produce your own failure-rate table alongside the context headroom each configuration buys.

**What’s going on:** Module 10 builds out a full multi-protocol tool-call shootout. This module needs a much smaller version of the same idea now, because "context gained" without "tool-call failure rate at that context" is exactly the kind of incomplete result 9.5 warns against — a bigger context window that silently breaks tool calls isn’t a win.

**Steps:**
1. Assemble a small battery of tool-calling tasks — 20 to 50 is plenty for this module’s purposes — each requiring at least one correctly formed and correctly closed tool call, ideally including a few multi-turn tasks where an earlier tool result has to be referenced correctly later in the same context window.
2. Pick a small set of KV-cache configurations to test: `f16`/`f16` (baseline), `q8_0`/`q8_0`, `q4_0`/`q4_0`, and K-only `q8_0` with V left at `f16` (the configuration 9.3 showed doesn’t require flash attention at all).
3. For each configuration, run the battery at two context lengths — a short one (e.g. 8K) and one deliberately close to the point where your 9.2 context-multiplier prediction says the quantized configuration should be buying you meaningfully more usable headroom than `f16` (e.g., somewhere past 64K, scaled to whatever your `q8_0`/`q8_0` multiplier predicts as newly reachable).
4. Record, per configuration per context length: the fraction of tasks with a malformed tool call (invalid JSON or syntax), the fraction with a *valid* but *semantically wrong* tool call (right syntax, wrong tool or wrong arguments), and the fraction that failed to close a multi-turn tool reference correctly.
5. Tabulate context gained (your 9.2 multiplier, or the actual longest context you tested successfully) alongside the tool-call failure rate at each configuration — this pairing is the module’s real deliverable, not either number alone.

> [!success] Done when
> You have a table of tool-call failure rate (malformed, semantically-wrong, and multi-turn-reference-broken) at each of the four KV-cache configurations, at both a short and a long context length, presented alongside the context-length headroom each configuration bought you.

> [!question]- It’s not working
> 1. `q4_0`/`q4_0` shows a low failure rate on your model, contradicting the 9.4 collapse case? That’s a legitimate, model-dependent result, consistent with the "must be benchmarked per-model" framing in 9.4 — don’t force your data to match the cited collapse case if it doesn’t; report your own number as your own finding.
> 2. Failure rate is *high* even at `f16` baseline? Check your tool-calling harness and prompt template before blaming KV-cache quantization at all — a broken baseline invalidates every comparison built on top of it.
> 3. Can’t reach the longer context length you planned to test without running out of VRAM? Recompute your available context headroom using the Module 4 budget equation with your actual KV-cache byte figures from 9.2 — you may need a smaller `n_cpu_moe` or a more aggressively quantized weight set from Module 8 to free the room.

---

## Check your understanding

> [!question]- 1. Why does quantizing the V cache require flash attention, while quantizing the K cache doesn’t?
> > [!success]- Answer
> > Without flash attention, the V cache is stored transposed relative to how flash attention lays it out. Block-quantized formats need contiguous, block-aligned rows to dequantize efficiently, and a transposed layout breaks that contiguity for V — the block boundaries no longer line up with how the data needs to be read. K is never transposed under either attention path, so K-only quantization never runs into that layout conflict and works fine without flash attention.

> [!question]- 2. The cited ARC-Challenge results show `q4_0`/`q4_0` KV-cache quantization causing a 92%→24.2% accuracy collapse on one model but almost no change on another. What’s the correct takeaway?
> > [!success]- Answer
> > That `q4_0` KV-cache quantization is not safe to assume based on another model’s results — it must be benchmarked per-model. `q8_0` behaved consistently near-lossless across all four tested models, but `q4_0`’s effect was strongly model-dependent, ranging from negligible to near-total collapse on the same nominal setting. The correct rule is "benchmark q4_0 yourself on your own model," not "q4_0 is either safe or unsafe in general."

> [!question]- 3. If you need to free up VRAM and have to choose between quantizing weights further or quantizing the KV cache further, which should you reach for first, and why?
> > [!success]- Answer
> > Quantize the weights first. Documented cases show KV-cache quantization causing reproducible tool-call failures and severe token-flip rates at long context — failures specific to agentic tool-calling that weight quantization at a comparable precision level did not produce in the same body of reporting. FP8/INT8 weight quantization stayed reliable for tool-calling in cited cases, while INT4 KV-cache quantization and NVFP4 KV-cache quantization did not. KV-cache quantization beyond q8_0/q8_0 is the higher-risk lever and should be reached for only after weight-quantization headroom is exhausted.

## What’s next

You now have a proven mechanical explanation for the flash-attention/V-cache constraint, cited quality evidence separating near-lossless `q8_0` from model-dependent `q4_0`, and your own measured tool-call failure rate at several KV-cache configurations and context lengths. Every measurement so far — Module 8’s weight ladder and this module’s KV-cache configurations — has assumed a single model process serving one client at a time through a bare `llama-server`. [[module-10-serving]] moves to the actual serving layer: swapping models on demand, and the tool-call contract itself, tested across three different protocols and including your own reproduction (or refutation) of a specific claim about which model sizes can be trusted with agentic tool-calling at all.

<script src="/tutor.js" defer></script>
