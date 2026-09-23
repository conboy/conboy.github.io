---
title: "Two Pools: Local AI Infrastructure for Coding on One Workstation"
description: "Running a 30B coding model on a 16GB GPU, and knowing exactly why it works — a measurement-driven course in memory placement, quantization, and agentic tool-calling on a single Ryzen/Radeon workstation."
tags:
  - local-ai
  - llm
  - amd
  - rocm
---

> [!info] Project status: learning roadmap
> These pages describe a proposed sequence of measurements, not a finished write-up. None of the module-specific tok/s numbers below have been produced yet on this exact workstation — the predictions in [[module-5-expert-placement]] are derived from a public config file and public bandwidth figures, and the whole point of the course is to go find out whether the prediction survives contact with real silicon. Where a number is marked "predicted," treat it as a hypothesis, not a result.

Can a single desktop run a serious coding model, locally, well enough to point an agentic harness at it and trust what comes back?

The model in question is **Qwen3-Coder-30B-A3B-Instruct** — 30 billion parameters, 8 active per token, a coding-focused Mixture-of-Experts model small enough to be interesting and large enough that AMD’s own public testing with Cline found smaller models "consistently fail" at agentic tool calling. The workstation is a **Ryzen 9 9950X paired with an RX 9070 XT (16GB)** — a real desktop, not a rented accelerator. The GPU alone cannot hold this model. That mismatch isn’t a problem to route around. It’s the entire subject of this course.

> [!abstract] The one insight that determines everything
> Your workstation has **two memory pools**, roughly **11x apart in bandwidth** — a 16GB VRAM pool at up to 640 GB/s, and a 128GB DDR5 pool at a theoretical 57.6 GB/s once you account for the 4-DIMM tax (see [[module-0-two-pools]]). Every configuration decision from here forward — quantization level, which experts go where, KV cache size, context length — is really a decision about **which bytes live in which pool, and how often you pay to move them.**
>
> $$t_{\text{token}}(n) = \frac{V(n)}{BW_{\text{vram}}} + \frac{D(n)}{BW_{\text{ddr}}}$$
>
> where $n$ is the number of transformer layers whose experts you’ve placed on the CPU side. Notice what that equation does *not* contain: a single "how fast is my GPU" number. **The roofline is not a property of your hardware — it is a function of placement.** Mixture-of-Experts architectures exist specifically so that most of a model’s bytes can sit in the slow pool without being read on every single token.

This is why the course ends with **two** capstones rather than one. The first checks whether your own measured curve matches the equation above. The second checks whether the context length you’re actually running can be trusted at all.

---

## Who this is for

You’ve run a local model before — Ollama, LM Studio, something — and it worked, in the sense that text came out the other end. You want to understand *why* the knobs (`--n-cpu-moe`, `-ctk`, `-fa`) do what they do, instead of copying a config off a forum post that never states its hardware. **You do not need prior ML infrastructure experience.** Every term is defined the first time it matters, and there’s a [[concepts|glossary]] for anything you lose track of.

Every module ends in a **measurement** — a number your own hardware produced — never a feeling. "It feels faster now" doesn’t count until you have a `tok/s` figure and know which variable moved it.

> [!tip] If you get lost
> Two places to look. [[concepts|Concepts]] defines every term — MoE, KV cache, effective context, GBNF grammar — with numbers specific to this model and this GPU. [[reference-material]] holds every source this course leans on, tagged by the module that needs it, plus a checklist for spotting the content-farm benchmark numbers that look precise and mean nothing.

---

## Start here

Read [[workstation-specs]] first — the Ryzen 9 9950X, the RX 9070 XT, and the 4x32GB DIMM population all force specific decisions later, and the numbers there are the ones this whole course is built on. Then begin at [[module-0-two-pools]], which gets you a real, measured DDR5 bandwidth number using STREAM Triad — no GPU or ROCm required yet, so it’s the one module you can run before you’ve even installed Linux. [[module-1-toolchain]] does the same honesty check for VRAM and gets ROCm 10 and llama.cpp built twice, once against HIP and once against Vulkan.

> [!tip] There’s a working local coding session by week 3
> [[module-2-calibration]] gets a dense model that actually fits entirely inside 16GB running end to end, so you have a clean, uncomplicated baseline *before* expert placement, quantization ladders, and KV-cache tuning start interacting with each other.

---

## The modules

| # | Module | Hours | What you measure |
|---|---|---|---|
| 0 | [[module-0-two-pools]] | 5 | Sustained DDR5 GB/s (STREAM Triad), the thread-count bandwidth elbow |
| 1 | [[module-1-toolchain]] | 6 | Sustained VRAM GB/s on the RX 9070 XT, HIP vs Vulkan builds side by side |
| 2 | [[module-2-calibration]] | 6 | tok/s on a model that fits entirely in VRAM — your uncomplicated baseline |
| 3 | [[module-3-eval-instrument]] | 8 | Aider Polyglot pass rate and run-to-run variance for that baseline |
| 4 | [[module-4-vram-budget]] | 6 | Non-expert weight size, KV cache bytes/token at 8K/32K/128K, remaining headroom |
| 5 | [[module-5-expert-placement]] | 12 | tok/s across `n_cpu_moe` = 0…48 — your own roofline curve |
| 6 | [[module-6-prefill-decode]] | 7 | Prefill tok/s vs decode tok/s, measured separately |
| 7 | [[module-7-backend-shootout]] | 9 | HIP vs Vulkan tok/s by model shape; your own `-fa` on/off A/B |
| 8 | [[module-8-quantization]] | 7 | Aider Polyglot pass rate across the Q4_K_M → Q2_K_XL → 1-bit ladder |
| 9 | [[module-9-kv-quantization]] | 6 | Tool-call failure rate, K-only vs K+V quant, at long context |
| 10 | [[module-10-serving]] | 9 | Schema-valid tool-call rate under GBNF-constrained decoding |
| 11 | [[module-11-aider]] | 7 | Tokens per turn, repo-map hit rate as file count grows |
| 12 | [[module-12-cline]] | 7 | Token overhead of an XML tool-syntax harness vs Aider’s diff format |
| 13 | [[module-13-claude-code]] | 6 | What breaks — extended thinking, Plan Mode, prompt caching — pointing Claude Code at a local model |

> [!success] Capstone 1 — The Predicted Machine (2 weeks)
> Run the full `n_cpu_moe` = 0…48 sweep, plot measured tok/s against the $t_{\text{token}}(n)$ prediction from the abstract above, and account for every point that disagrees. This is where "the roofline is a function of placement" either survives contact with your GPU or it doesn’t. See [[capstone-1-predicted-machine]].

> [!warning] Capstone 2 — Your Effective Context (2 weeks)
> Run a NoLiMa-style needle-in-haystack sweep against *your actual daily config* — your quant, your context length, your KV-cache settings — and find the point where retrieval quality drops below 85% of its short-context baseline. The marketed 256K or 1M context window on the box is not the number that matters. The number where your setup actually degrades is. See [[capstone-2-effective-context]].

**Total: ~14–16 weeks at 6–10 hrs/week.**

See [[concepts|Concepts]] for definitions and [[reference-material]] for sources.

---

## Three things nobody has published numbers for

> [!question]- Why call these out specifically?
> Because separating "I measured this" from "a forum post claims this" is the entire credibility model of this course, and these three gaps are exactly where the public record runs dry. Filling any one of them with a documented methodology is a genuinely useful contribution, not busywork.

1. **No public RX 9070 XT MoE-offload numbers exist.** Plenty of `--n-cpu-moe` benchmarks circulate for NVIDIA cards and for older AMD parts; none target gfx1201. [[module-5-expert-placement]] is where you generate the first one.
2. **No controlled AMD `-fa` on/off benchmark exists.** Flash attention’s effect on throughput is well characterized on CUDA; on ROCm/RDNA4 it’s asserted more often than measured. [[module-7-backend-shootout]] is where you run the A/B yourself, with a command line and a variance figure attached.
3. **No controlled repo-map vs. RAG vs. long-context comparison exists** for a 30B local coding model against a real repository. [[module-11-aider]] through [[module-13-claude-code]] build the pieces; nobody has published the head-to-head.

---

## Ground rules

> [!important] Four rules that keep this honest
>
> 1. **Every module ends in a measurement.** A `tok/s` figure, a pass rate, a GB/s number — printed by a tool you ran, on hardware you own. Nothing else counts as "done."
> 2. **Separate predicted from measured, always.** The equation in the abstract is a hypothesis until Capstone 1 checks it against your GPU. Never write a number down as measured when it was only predicted.
> 3. **Placement before quantization before serving.** Get [[module-5-expert-placement|expert placement]] right on a baseline you understand before compounding it with a quant ladder and a KV-cache scheme — three simultaneous unknowns is not a debuggable system.
> 4. **Q4_K_M is the quality floor, not a starting guess.** [[module-8-quantization]] shows measured scores holding at Q4_K_M and falling off a cliff by Q2_K_XL. Don’t quantize past the floor and then blame the harness when tool calls start failing.

---

## Why bother, when an API is better

A hosted frontier model is faster to set up, almost certainly more capable per token, and someone else pays for the electricity and the context window. This course doesn’t pretend otherwise. What building this instead gets you:

- a concrete, falsifiable model of *why* a 30B MoE model fits on a 16GB card at all, instead of a vague sense that "quantization helps"
- the ability to read a `llama-server` flag list and know which flags trade VRAM for tok/s, and which trade tok/s for correctness
- a working local fallback with a known, *measured* failure mode — not an unknown one — for the day the API is down, rate-limited, or the code genuinely cannot leave your machine
- the specific, measured difference between a marketed context window and an effective one, which transfers directly to judging *any* vendor’s context-length claims, hosted or not

That last point is worth sitting with. Every context-window number on every model card, hosted or local, is measured the vendor’s way, on the vendor’s chosen task. Capstone 2 teaches you to measure it yours.

<script src="/tutor.js" defer></script>
