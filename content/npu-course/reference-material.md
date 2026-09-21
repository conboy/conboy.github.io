---
title: "Reference Material"
description: "Papers, source repos and prior art for building a small int8 NPU, tagged by the phase that needs them."
tags:
  - npu
  - reference
---

# Reference Material

Part of [[index|Zero to NPU]].

> [!tip] Pull these as you need them, not up front
> Reading all of this before you start is a procrastination strategy. Each entry is tagged with the phase where it becomes relevant.

---

## Core

| Topic | Source | Needed by |
|---|---|---|
| The model + golden C | [`karpathy/llama2.c`](https://github.com/karpathy/llama2.c) | [[phase-1-baseline-llm]] |
| Board details | [Blackboard User Manual](https://realdigital.org/doc/9c908d94497abb1eac41175d1ab05b88) · [Product page](https://www.realdigital.org/hardware/blackboard) | [[phase-0-toolchain]] |
| Silicon details | [Zynq-7000 DS190 Overview](https://www.mouser.com/datasheet/2/903/ds190-Zynq-7000-Overview-1595492.pdf) | [[board-specs]] |

## Theory

| Topic | Source | Needed by |
|---|---|---|
| Quantization math | Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* (the gemmlowp paper) | [[phase-2-numerics]] |
| Systolic arrays | Jouppi et al., *In-Datacenter Performance Analysis of a TPU* (§ on the MXU) | [[phase-3-mac-engine]] |
| Online softmax | Milakov & Gimelshein, *Online normalizer calculation for softmax* | [[phase-5-vector-unit]] |

## Prior art — read, don't copy

| Project | Why it's useful |
|---|---|
| [`Leonui/tiny-npu`](https://github.com/Leonui/tiny-npu) | 16×16 int8 array, real GPT-2/LLaMA weights, cycle-accurate golden models. Closest architectural match. Note: FPGA synthesis is still a *roadmap* item there — it's a simulation project. |
| [`Buck008/Transformer-Accelerator-Based-on-FPGA`](https://github.com/Buck008/Transformer-Accelerator-Based-on-FPGA) | Parameterized systolic array that actually runs on Zynq hardware today. Sized for a Z1/Z2 — you'll need to shrink it. |
| [`taoFPGA/accelerator`](https://github.com/taoFPGA/accelerator) | Fused matmul → softmax → GELU in int8 streaming RTL on Zynq-7000. |

## Calibration

| Source | What it tells you |
|---|---|
| [LlamaF (arXiv 2409.11424)](https://arxiv.org/html/2409.11424v1) | TinyLlama-1.1B at **1.5 tok/s** on a ZCU102 — a board vastly larger than yours. Useful for calibrating what's genuinely hard, and for understanding why your Tier 2 capstone is slow. |

> [!warning] On the Zynq-7020 data point
> One published 16×16 int8 array on a **7020** mapped only 64 of 256 multipliers to DSPs — the other 192 became fabric multipliers, consuming most of a 59% LUT occupancy.
>
> The 7020 has 220 DSPs. You have **60**. This is the clearest available evidence that 16×16 is not reachable on your part, and why [[phase-3-mac-engine|8×4]] is the right call.
