---
title: "Kernel Ladder: ML GPU Kernels on Radeon and Instinct"
description: "A twelve-week, measurement-driven course in writing ML inference kernels with HIP and ROCm, climbing from an RX 5700 XT to an RX 9070 XT to a rented Instinct MI300X."
tags:
  - gpu
  - rocm
  - hip
  - kernels
---

> [!info] Project status: learning roadmap
> These pages describe a proposed sequence of labs and the number each one should produce. None of the module results below have been produced yet. Where a target is given ("60% of hipBLASLt"), treat it as an acceptance criterion to test, not a claim about what the hardware does.

Can you write the kernels a modern inference stack actually runs — GEMM, attention, mixture-of-experts routing, FP8 and FP4 quantization — fast enough to stand next to the vendor libraries, and explain every instruction in the hot loop?

This course answers that on AMD hardware, climbing a three-rung ladder: an **RX 5700 XT** (RDNA1, no matrix cores) for the fundamentals, an **RX 9070 XT** (RDNA4, WMMA matrix cores with FP8) for matrix-core work at home, and a rented **Instinct MI300X** (CDNA3) — plus an MI355X (CDNA4) for MXFP4 — for the datacenter-class kernels.

> [!abstract] The one insight that determines everything
> **A kernel is either waiting on memory or waiting on math, and you should always know which.** Every optimization in this course moves a kernel along one axis of the roofline:
>
> $$\text{attainable FLOP/s} = \min\left(\text{peak FLOP/s},\ \text{arithmetic intensity} \times \text{peak bytes/s}\right)$$
>
> Softmax, normalization and decode attention live on the bandwidth slope, so their score is percent of peak bytes/s. GEMM and prefill attention can reach the compute roof, so their score is percent of peak matrix-core FLOP/s. A benchmark number without its position on the roofline is half a result.

---

## Who this is for

You already understand GPU architecture from somewhere — graphics, drivers, validation, or CUDA — and you want to move into ML kernel engineering on AMD GPUs. You can read C++ and you know roughly what a wavefront is. **You do not need prior ML or ROCm experience.** Every term is defined in [[rocm-kernels-course/concepts|Concepts]], and every claim about hardware links to a public source in [[rocm-kernels-course/reference-material|Reference material]].

Every module ends in a **measurement**: GB/s, TFLOP/s, a percent of a vendor library, a tokens/s figure, or a leaderboard time. "It feels faster" does not count.

> [!tip] If you get lost
> [[rocm-kernels-course/concepts|Concepts]] defines every term — MFMA, WMMA, LDS, AGPR, online softmax, FNUZ — with a link back to the module where it matters. [[hardware-specs]] explains which rung of the hardware ladder each lab needs. [[gpu-mode]] is where to get unstuck by asking people who have written these kernels.

---

## Start here

Read [[hardware-specs]] first: it explains why the 5700 XT is enough for the first five weeks, what the 9070 XT adds, and how to get Instinct time cheaply. Then begin at [[module-0-setup]], which gets a benchmark harness running on both Radeon cards and produces your first GB/s number.

---

## The modules

| # | Module | Weeks | GPU | What you measure |
|---|---|---|---|---|
| 0 | [[module-0-setup]] | 1 | 5700 XT + 9070 XT | Vector add GB/s vs peak on both cards |
| 1 | [[module-1-hip-model]] | 1–2 | 5700 XT | Transpose at 80%+ of peak bandwidth; occupancy vs throughput curve |
| 2 | [[module-2-memory-bound]] | 2–3 | 5700 XT + 9070 XT | Reduction within 10% of rocPRIM; softmax, RMSNorm, RoPE GB/s |
| 3 | [[module-3-profiling]] | 4 | 9070 XT (+ first MI300X hour) | Decode-step time breakdown; one kernel placed on the roofline |
| 4 | [[module-4-gemm]] | 5–7 | All three rungs | TFLOP/s per optimization step vs rocBLAS / hipBLASLt |
| 5 | [[module-5-attention]] | 8 | 9070 XT → MI300X | FlashAttention forward vs the library; decode attention GB/s |
| 6 | [[module-6-quantization-moe]] | 9 | MI300X (+ MI355X) | FP8 GEMM vs hipBLASLt; MoE layer stage breakdown |
| 7 | [[module-7-libraries]] | 10 | 9070 XT + MI300X | CK-Tile and Triton versions vs your hand-written HIP |
| 8 | [[module-8-shipping]] | 11 | MI300X (1× and 8×) | End-to-end tokens/s change from one kernel; RCCL overlap |

> [!success] Capstone — a fast embeddings encoder (weeks 11–12)
> Build the inference path of an LLM-based embedding model from your own kernels and benchmark it against Hugging Face Transformers and a serving engine on MI300X. See [[capstone]].

**Running alongside every module:** the [[gpu-mode|GPU MODE track]], which pairs each module with a public leaderboard problem. Week 12 adds [[interview-prep]].

**Total: about 130 hours over 12 weeks at 10–12 hours a week,** plus roughly US$100–200 of rented Instinct time after free credits. See [[hardware-specs]] for where that estimate comes from.

---

## Ground rules

> [!important] Four rules that keep this honest
>
> 1. **Every module ends in a measurement.** A number printed by a tool you ran, with the GPU, gfx target and ROCm version written next to it.
> 2. **Correctness before speed.** Every kernel is checked against a PyTorch or CPU reference with explicit per-dtype tolerances before it is timed.
> 3. **Always compare against the vendor library on the same card.** "2× faster than my naive version" means nothing; "71% of hipBLASLt at 4096³ on MI300X" means something.
> 4. **Public hardware and public sources only.** Use cards you own and cloud accounts you pay for. Every hardware fact cites a public datasheet, ISA guide, repo or paper.

---

## Why bother, when the libraries already exist

hipBLASLt, Composable Kernel and AITER will beat your first several attempts, and this course doesn't pretend otherwise. What writing the kernels yourself gets you:

- the ability to read a profiler report and know which of ten possible causes is the real one
- a concrete model of why matrix cores need specific data layouts, instead of a vague sense that "tensor cores are fast"
- the skill to write the fused, odd-shaped kernel no library ships yet — which is most of what kernel engineers are actually paid to do
- a portfolio of public, reproducible numbers on real hardware

<script src="/tutor.js" defer></script>
