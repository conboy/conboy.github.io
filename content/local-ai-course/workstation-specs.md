---
title: "Workstation Specs — The Two Pools"
description: "The bandwidth, capacity, and DIMM-population constraints that force every decision in this course."
tags:
  - local-ai
  - hardware
  - reference
---

# Workstation Specs — Know your memory pools before you place a single expert

Part of [[index|Two Pools]].

| Part | Model | The number that matters |
|---|---|---|
| CPU | Ryzen 9 9950X | 16 Zen 5 cores, **dual-channel** memory controller |
| GPU | Radeon RX 9070 XT | 16 GB GDDR6, **640 GB/s**, gfx1201 (RDNA4) |
| RAM | 128 GB DDR5 as **4 × 32 GB** | **DDR5-3600 → 57.6 GB/s** |

> [!abstract] The one insight that determines everything
> This workstation has two memory pools, roughly **11× apart** in bandwidth: 16 GB of VRAM at 640 GB/s, and 128 GB of system RAM at a theoretical 57.6 GB/s once the DIMM population is accounted for. Every configuration decision in this course — quantization level, which experts go where, KV cache size, context length — is really a decision about **which bytes live in which pool, and how often you pay to move them.**
>
> $$t_{\text{token}} = \frac{B_{\text{vram}}}{BW_{\text{vram}}} + \frac{B_{\text{ddr}}}{BW_{\text{ddr}}}$$
>
> where $B$ is bytes actually **read per token** from each pool — not bytes stored there. Mixture-of-Experts models exist specifically so that most of a model’s bytes can sit in the slow pool without being read on every single token.

---

## Pool 1: VRAM — fast, and too small

640 GB/s comes from a 256-bit bus at 20 Gbps. That part of the spec is generous. The capacity is the problem.

Here’s what Qwen3-Coder-30B-A3B actually costs at Q4_K_M (~4.8 bits/weight):

| Component | Params | Size at Q4_K_M |
|---|---|---|
| Non-expert (attention, embeddings, router, norms) | 1.49 B | 0.89 GB |
| Routed expert FFN | 29.01 B | 17.41 GB |
| **Total** | **30.5 B** | **18.30 GB** |

> [!warning] The model does not fit
> 18.30 GB into a 16 GB card — and that’s **before** a single byte of KV cache. At 32K context, KV cache adds another 3.22 GB on top. This is not a problem to route around. It **is** the course.

---

## Pool 2: System RAM — large, and slower than you bought it for

Dual-channel DDR5 means a 128-bit bus, or 16 bytes per transfer, so:

$$BW_{\text{ddr}} = 16 \times MT/s$$

The 9950X officially supports DDR5-5600, which works out to 89.6 GB/s. But AMD publishes supported memory speed **by DIMM population**, not by total capacity:

| Population | Official supported speed | Bandwidth |
|---|---|---|
| 2 × single-rank | DDR5-5600 | 89.6 GB/s |
| 2 × dual-rank | DDR5-5600 | 89.6 GB/s |
| **4 × single-rank** | **DDR5-3600** | **57.6 GB/s** |
| **4 × dual-rank** | **DDR5-3600** | **57.6 GB/s** |

> [!bug] The 4-DIMM tax
> Reaching 128 GB as 4 × 32 GB drops the officially supported speed from 5600 to 3600 — a **36% bandwidth cut**, purely from how the capacity was assembled. It’s invisible in every parts list and every spec sheet screenshot. It directly sets the token rate of every model that spills into system RAM.

> [!tip] If you’re still speccing a machine
> Buy 2 × 64 GB instead of 4 × 32 GB. Two DIMMs keep you at the higher officially supported speed for the same total capacity.

The JEDEC number in that table is a **floor**, not a ceiling — a board may post a higher sustained number with manual tuning. That’s exactly why [[module-0-two-pools]] measures actual DDR5 bandwidth with STREAM Triad instead of assuming the datasheet number. The course runs on what you measure, not on what AMD prints.

---

## The ratio

| | VRAM | System RAM | Ratio |
|---|---|---|---|
| Capacity | 16 GB | 128 GB | 1 : 8 |
| Bandwidth | 640 GB/s | 57.6 GB/s | 11.1 : 1 |

Eight times the capacity, at one eleventh the speed.

> [!important] These are theoretical peaks and you will not hit them
> Real sustained bandwidth is typically 60–90% of the peak numbers on this page, on both sides of the equation. **Do not build predictions on the numbers on this page.** [[module-0-two-pools]] and [[module-1-toolchain]] replace both of them with measured values before the course asks you to predict anything. The numbers here are the specification; the course runs on the measurement.

---

## What this machine can and cannot run

At Q4_K_M:

| Model | Total / active | Weights | Fits VRAM? | Viable? |
|---|---|---|---|---|
| Qwen2.5-Coder-7B | 7B dense | ~4.5 GB | Yes | Yes — calibration model, see [[module-2-calibration]] |
| Qwen2.5-Coder-14B | 14B dense | ~9 GB | Yes | Yes, with modest context |
| **Qwen3-Coder-30B-A3B** | **30.5B / 3.3B** | **18.3 GB** | **No** | **Yes, via expert offload — the main model of this course** |
| gpt-oss-120b | 117B / 5.1B | ~63 GB | No | Yes, mostly in system RAM — predicted in [[capstone-1-predicted-machine]] |
| GLM-4.5-Air | 106B / 12B | ~60 GB | No | Yes, but 12B active is a heavy per-token DDR5 bill |
| Any 70B dense | ~40 GB | No | **No** |

> [!warning] Why dense models above ~14B are off the table
> A dense model reads **every** weight for **every** token. A 70B dense model at Q4 is roughly 40 GB, of which ~25 GB sits in system RAM once VRAM is full: 25 ÷ 57.6 ≈ 434 ms/token ≈ **2.3 tok/s**. Run the same arithmetic on the 30B MoE model and only **1.088 GB** of expert weights get read per token, because just 8 of 128 experts activate. That difference — not raw parameter count — is why this course is built on a MoE model. [[module-5-expert-placement]] measures whether that theory survives contact with real silicon.

---

## Software baseline

| Component | Choice |
|---|---|
| OS | Linux, Ubuntu 24.04.x or 26.04, dual-boot |
| GPU stack | ROCm 10.0 — gfx1201 officially supported, no `HSA_OVERRIDE_GFX_VERSION` needed |
| Inference engine | llama.cpp, recent master, built **twice** — once against HIP, once against Vulkan ([[module-7-backend-shootout]] needs both) |

> [!info] Why Linux, and why this is not just preference
> AMD’s own Windows compatibility documentation states that PyTorch on Windows ships ROCm components, but "the entire ROCm stack is not yet supported on Windows." llama.cpp does work on Windows, via Vulkan or the HIP SDK — but the full toolchain, and testing both backends cleanly against each other, is a Linux story today.
>
> One more thing worth flagging: gfx1201 support is recent. It was **not** supported at this card’s launch — official listing arrived around ROCm 7.0. Any 2025 guide telling you to set `HSA_OVERRIDE_GFX_VERSION` on this card is stale.

---

## Sources

- 640 GB/s, 16 GB GDDR6, gfx1201 — AMD RX 9070 XT product page. The page is JavaScript-rendered, so these figures were confirmed against the launch press-kit specification rather than trusted from a raw fetch; unchanged since the February 2025 launch.
- Dual-channel support, the DDR5-5600 official figure, and the 4-DIMM DDR5-3600 table — AMD Ryzen 9 9950X processor specification page, same rendering caveat as above.
- Model architecture — the shipped `config.json` for Qwen3-Coder-30B-A3B-Instruct: 48 layers, 4 KV heads, head_dim 128, 128 experts with 8 active, `max_position_embeddings` 262144, `rope_scaling: null`.
- ROCm 10.0 gfx1201 support — the ROCm compatibility matrix.

See [[reference-material]] for full citations.

<script src="/tutor.js" defer></script>
