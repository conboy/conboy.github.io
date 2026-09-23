---
title: "Concepts — Glossary"
description: "Plain-English definitions for every term in the Two Pools course, from memory-bandwidth-bound to effective context."
tags:
  - local-ai
  - reference
---

# Concepts

This is the glossary for [[index|Two Pools]]: every term used running Qwen3-Coder-30B-A3B-Instruct on a Ryzen 9 9950X + RX 9070 XT, assuming no prior local-inference infrastructure knowledge. Read it front to back or jump straight to a linked word. Each entry ends with a *Used in* line pointing at the module where it does real work.

## Memory & bandwidth

### Memory-bandwidth-bound

**A workload where throughput is capped by how fast bytes move, not by how fast the compute units run.**

Token generation reads every active weight from memory exactly once per token, and modern GPUs and CPUs do far more arithmetic per second than they can feed with data — so the multiply-accumulate units sit idle waiting on bytes almost the entire time. This is why the RX 9070 XT’s 640 GB/s and DDR5’s 57.6 GB/s, not TFLOPs, are the numbers that predict your tok/s.

*Used in: [[module-0-two-pools]], [[module-1-toolchain]]*

### The two pools

**VRAM and system DDR5, roughly 11x apart in bandwidth on this workstation.**

The RX 9070 XT’s 16GB of VRAM moves data at up to 640 GB/s; the 128GB of DDR5, populated as 4x32GB, sustains a theoretical 57.6 GB/s (DDR5-3600, dual-channel, 16 bytes/transfer). Every placement decision in this course — which layers’ experts live where — is a decision about how much of a token’s traffic crosses the fast pool versus the slow one.

*Used in: [[module-0-two-pools]], [[module-1-toolchain]], [[module-5-expert-placement]]*

### The 4-DIMM tax

**The DDR5 speed penalty for populating all four DIMM slots instead of two.**

The 9950X supports DDR5-5600 with two DIMMs installed, but drops to DDR5-3600 once all four slots carry a module — a signal-integrity cost of routing four ranks on a two-channel bus, not a misconfiguration. That’s 57.6 GB/s theoretical instead of 89.6 GB/s, before a single benchmark runs.

*Used in: [[module-0-two-pools]], [[workstation-specs]]*

### Roofline (placement-dependent)

**A performance ceiling that, for a MoE model split across two memory pools, is a function of *where bytes live*, not a fixed hardware number.**

For a dense model the roofline is one line: bandwidth divided by model size. For a MoE model split between VRAM and DDR5, the ceiling is $t_{\text{token}}(n) = V(n)/BW_{\text{vram}} + D(n)/BW_{\text{ddr}}$ — it moves every time you change $n$, the count of layers offloaded to CPU. [[module-5-expert-placement]] is the module that turns this from an equation into a measured curve.

*Used in: [[module-5-expert-placement]], [[capstone-1-predicted-machine]]*

## Mixture-of-Experts & model architecture

### Mixture-of-Experts (MoE)

**An architecture where each token is routed through a small subset of many expert sub-networks, instead of through one dense feed-forward block.**

Qwen3-Coder-30B-A3B-Instruct has 128 experts per layer and activates 8 per token. Only those 8 experts’ weights need to be read to produce that token — the other 120 sit untouched in memory. This is the entire reason a 30B-parameter model can be workable on a 16GB card: most of its bytes are read rarely, not on every token.

*Used in: [[module-4-vram-budget]], [[module-5-expert-placement]]*

### Expert

**One of the parallel feed-forward sub-networks a MoE layer routes tokens through.**

Each of the 48 layers in this model carries 128 experts. At Q4_K_M quantization the routed experts total 29.01B parameters (17.41GB) — nearly all the model’s weight mass — while everything that is *not* an expert (attention, embeddings, router, norms) is only 1.49B parameters (0.89GB). Placement decisions are really always about where *expert* weights live, because that’s where nearly all the bytes are.

*Used in: [[module-4-vram-budget]], [[module-5-expert-placement]]*

### Router

**The small learned network inside each MoE layer that picks which experts a given token activates.**

Counted inside the "non-expert" 1.49B parameters above — cheap in bytes, but it runs on every token and its output determines which 8-of-128 expert weights get read next. You never quantize or offload it separately; it stays with the always-resident weights.

*Used in: [[module-5-expert-placement]]*

### Active parameters

**The subset of a model’s total parameters actually read to produce one token.**

Qwen3-Coder-30B-A3B-Instruct has 30B total parameters but activates roughly 3B per token (8 of 128 experts per layer, plus the always-active non-expert weights) — hence the "A3B" in its name. At Q4_K_M, active *expert* bytes per token work out to $8/128 \times 17.41\text{GB} \approx 1.088\text{GB}$, which is the number that actually predicts VRAM-side traffic per token, not the 18.30GB total weight size.

*Used in: [[module-4-vram-budget]], [[module-5-expert-placement]]*

### Dense-equivalent baseline

**The hypothetical throughput if every weight, not just the active experts, had to be streamed for every token.**

Streaming the full 18.30GB Q4_K_M weight set from DDR5 at 57.6 GB/s gives 318ms/token — about 3.15 tok/s. Measured MoE+placement throughput is predicted at roughly 49–323 tok/s depending on placement (see [[module-5-expert-placement]]), a ~33x win purely from not reading experts you don’t need. This baseline is the number that makes the MoE win legible as a number, not just a feeling.

*Used in: [[module-5-expert-placement]], [[capstone-1-predicted-machine]]*

### Attention head, KV head, GQA

**The parallel "lanes" a transformer layer uses to compute attention, and the grouping trick that lets several query heads share one key/value head.**

This model uses 32 attention (query) heads but only 4 KV heads per layer — grouped-query attention, where groups of 8 query heads share a single key/value pair. Head dimension is 128. Fewer KV heads directly means a smaller KV cache, which is exactly why the KV-cache formula below scales with 4, not 32.

*Used in: [[module-4-vram-budget]], [[module-9-kv-quantization]]*

### Layer

**One repeated transformer block — attention plus a MoE feed-forward sub-layer — stacked to build the full model.**

This model has 48 layers. [[module-5-expert-placement|Expert placement]] is decided per layer: for a given layer, its experts either live in VRAM or get offloaded to system RAM, and $n$ in the roofline equation counts how many of the 48 layers you’ve pushed to the slow pool.

*Used in: [[module-5-expert-placement]]*

### Context window

**The maximum number of tokens (prompt plus generation) a model can attend over in one sequence, as configured.**

Qwen3-Coder-30B-A3B-Instruct’s `config.json` sets `max_position_embeddings` to 262,144 (256K) with `rope_scaling: null` — the widely advertised 1M-token context is a documented but *disabled-by-default* extension, not what you get out of the box. Don’t confuse this configured maximum with *effective* context — see NoLiMa below.

*Used in: [[module-4-vram-budget]], [[capstone-2-effective-context]]*

## KV cache

### KV cache

**Stored key and value vectors from every previous token in the sequence, kept so attention never recomputes them.**

Without a cache, generating token $N$ would re-run attention over every earlier token’s keys and values, which never change once computed. Its size scales with layers, KV heads, head dimension, and context length — all four are fixed by the model, so context length is the only lever you control.

*Used in: [[module-4-vram-budget]], [[module-9-kv-quantization]]*

### KV cache formula

**bytes/token = 2 × layers × kv_heads × head_dim × bytes_per_element.**

For this model at FP16: $2 \times 48 \times 4 \times 128 \times 2 = 98{,}304$ bytes per token. That works out to 3.22GB at 32K context, 12.88GB at 128K, and 25.77GB at 256K — a KV cache that alone exceeds the RX 9070 XT’s entire 16GB before a single model weight is loaded. At 1M context (were it enabled) the cache alone would need 103.08GB. This formula is why [[module-4-vram-budget|the VRAM budget]] module exists as its own module rather than a footnote.

*Used in: [[module-4-vram-budget]], [[module-9-kv-quantization]]*

### Effective context (NoLiMa)

**The longest context length at which a model still retains at least 85% of its short-context retrieval score — as distinct from its configured maximum.**

The NoLiMa benchmark (arXiv 2502.05167) found Llama 4 Scout’s effective length is **1,000 tokens** against a marketed 10,000,000 — base score 81.7, dropping to 72.3 at 1K, 61.8 at 2K, 50.8 at 4K, 35.5 at 8K, 26.9 at 16K, and 21.6 at 32K. No public NoLiMa-style number exists yet for Qwen3-Coder-30B-A3B-Instruct at the quant and context you actually run — that gap is exactly what [[capstone-2-effective-context|Capstone 2]] fills.

*Used in: [[capstone-2-effective-context]]*

## Quantization

### Quantization (weights)

**Storing each weight at fewer bits than its original float precision, trading some accuracy for a much smaller memory footprint.**

At Q4_K_M (~4.8 bits/weight), this model’s 18.30GB total collapses from a much larger FP16 footprint — the difference between not fitting on a 16GB card at any offload setting and fitting with room for a meaningful KV cache. [[module-8-quantization]] walks the full ladder and measures where quality actually falls off, rather than assuming lower is always fine.

*Used in: [[module-4-vram-budget]], [[module-8-quantization]]*

### Q4_K_M / GGUF quant naming

**A GGUF quantization scheme mixing 4-bit and 6-bit blocks depending on tensor sensitivity, at roughly 4.8 bits/weight overall.**

The "K_M" suffix in llama.cpp’s GGUF quant names denotes a "medium" k-quant mix — not every tensor is quantized equally aggressively. On Terminal-Bench-style coding evaluation, measured scores hold at roughly BF16’s ~75% through Q4_K_M with no measurable loss, drop to ~70% at Q2_K_XL, and become unusable at 1-bit. Q4_K_M is the practical floor for coding, not a comfortable middle ground.

*Used in: [[module-8-quantization]]*

### KV cache quantization

**Storing cached key and value vectors at reduced precision instead of FP16, to shrink the cache’s memory footprint.**

Controlled with `--cache-type-k`/`-ctk` and `--cache-type-v`/`-ctv` (values: `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`; default `f16`). This is riskier than weight quantization specifically for tool-calling — documented INT4-KV tool-call failures show up at long context, which is exactly the regime you need the cache savings most.

*Used in: [[module-9-kv-quantization]]*

### Flash attention’s V-cache constraint

**Why the V half of a quantized KV cache requires flash attention to be enabled, while K does not.**

Without flash attention, the V cache is stored transposed, and block-quantized types need contiguous, block-aligned rows — a transposed layout breaks that. K is never stored transposed, so K-only quantization works with or without flash attention; V-quantization requires `-fa` on. The `-fa` flag itself is tri-state (`on`/`off`/`auto`, default `auto` since llama.cpp PR #15434).

*Used in: [[module-9-kv-quantization]], [[module-7-backend-shootout]]*

## llama.cpp mechanics

### `--n-cpu-moe` / `-ncmoe`

**A llama.cpp flag that offloads the experts of the last $N$ MoE layers to CPU/DDR5, leaving everything else on GPU.**

Added in PR #15077 (August 2025), alongside the simpler `--cpu-moe`/`-cmoe` (offload *all* MoE layers). This is the single flag [[module-5-expert-placement]] sweeps from 0 to 48 to trace the measured roofline curve — it is, directly, the $n$ in $t_{\text{token}}(n)$.

*Used in: [[module-5-expert-placement]], [[module-4-vram-budget]]*

### `--override-tensor` / `-ot`

**The older, general-purpose regex mechanism for placing specific tensors on specific devices, predating `--n-cpu-moe`.**

Added in PR #11397, it matches tensor names against a regex and assigns them to a device — more flexible than `--n-cpu-moe`, and still the mechanism of choice once more than one GPU is involved, since `--n-cpu-moe` only reasons about a single CPU/GPU split.

*Used in: [[module-5-expert-placement]]*

### GBNF grammar

**A context-free grammar format llama.cpp uses to mask invalid tokens out of the logits before sampling.**

A GBNF grammar makes invalid JSON structurally unreachable — the sampler can only ever pick tokens that keep the output grammatically valid. That guarantees *syntax*, not *semantics*: a schema-valid tool call under GBNF constraint can still call the wrong tool or pass the wrong arguments. [[module-10-serving]] measures schema-valid rate, which is a different number from "correct" rate.

*Used in: [[module-10-serving]]*

### llama-bench

**llama.cpp’s built-in benchmarking binary for measuring prompt-processing and token-generation throughput.**

Used throughout [[module-6-prefill-decode]] and [[module-7-backend-shootout]] to produce comparable tok/s figures across backend, thread count, and flag combinations — always with the exact command line recorded, per the benchmark-honesty checklist in [[reference-material]].

*Used in: [[module-6-prefill-decode]], [[module-7-backend-shootout]]*

## Serving & coding harnesses

### Tool calling / function calling

**A protocol where the model emits a structured request to invoke a named function with arguments, instead of (or alongside) prose.**

How an agentic coding harness gets a model to actually edit a file or run a command, rather than just describe doing so. AMD’s own public testing with Cline found models smaller than Qwen3-Coder-30B "consistently fail" at this specifically — it is a harder capability than fluent chat, and it is the capability [[module-10-serving]] through [[module-13-claude-code]] are built around measuring.

*Used in: [[module-10-serving]], [[module-11-aider]], [[module-13-claude-code]]*

### Repo map

**Aider’s mechanism for giving the model a compressed view of a codebase’s structure without pasting every file into context.**

Built from tree-sitter parses of the repository plus a PageRank-style ranking over the file-dependency graph — it is a graph-ranking technique, *not* an embedding-based retrieval system, and confusing the two leads to the wrong mental model of what it will and won’t find relevant. [[module-11-aider]] measures how its hit rate changes as file count grows.

*Used in: [[module-11-aider]]*

### Diff / whole-file edit format

**How a coding harness tells the model to express a code change: as a targeted diff, or by re-emitting an entire file.**

Aider sidesteps native tool-calling entirely for edits, instead parsing diff or whole-file formats out of the model’s plain-text response. Cline and Roo take the opposite path — neither uses native JSON tool calling; both inline a custom XML-ish tool syntax directly in the prompt, which [[module-12-cline]] measures as a token-overhead cost against Aider’s format.

*Used in: [[module-11-aider]], [[module-12-cline]]*

### Prompt caching (and why switching models breaks it)

**Reusing a previously-processed prefix of a prompt so it doesn’t need to be re-run through the model on the next turn.**

Each model maintains its own cache; pointing a harness like Claude Code at a different backend model invalidates it, along with extended thinking and Plan Mode — all three depend on behavior specific to Anthropic’s own models, and don’t survive a swap to a local model even when that model speaks the same wire protocol. [[module-13-claude-code]] measures exactly what breaks when Claude Code is pointed at a local server via `ANTHROPIC_BASE_URL`/`ANTHROPIC_AUTH_TOKEN`.

*Used in: [[module-13-claude-code]]*

## Benchmarks

### Aider Polyglot

**A 225-exercise, Docker-based coding benchmark that runs against any OpenAI-compatible endpoint.**

The practical local benchmark used throughout this course — small enough to run repeatedly on a workstation, unlike SWE-bench Verified, which needs roughly 120GB of disk and 8 cores of heavy Docker orchestration per full run. [[module-3-eval-instrument]] sets this up as the course’s standing eval instrument; [[module-8-quantization]] and later modules reuse it.

*Used in: [[module-3-eval-instrument]], [[module-8-quantization]]*

## Still confused?

That’s expected — a glossary entry is a definition, not a lesson. Each module walks through these ideas with numbers specific to this model, this quant, and this GPU. Follow a term’s *Used in* link and read it in context; watching your own `n_cpu_moe` sweep produce a roofline curve, or your own NoLiMa sweep find where retrieval actually falls apart, tends to make it click in a way the definition alone won’t.

<script src="/tutor.js" defer></script>
