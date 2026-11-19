---
title: "Hardware Ladder — Which GPU Each Lab Needs"
description: "Why an RX 5700 XT covers the fundamentals, what an RX 9070 XT adds, and how to rent Instinct MI300X and MI355X time cheaply."
tags:
  - gpu
  - rocm
  - reference
---

# Hardware ladder

Part of [[index|Kernel Ladder]].

The course climbs three rungs. Each rung is used only for the labs that genuinely need it, so most of the hours run on hardware you already own.

| Rung | GPU | Architecture / target | Matrix units | Used for |
|---|---|---|---|---|
| 1 — sandbox | RX 5700 XT, 8 GB GDDR6 | RDNA1 / gfx1010 | None | [[module-1-hip-model]], [[module-2-memory-bound]], GEMM steps 1–5 in [[module-4-gemm]] |
| 2 — matrix cores at home | RX 9070 XT, 16 GB GDDR6 | RDNA4 / gfx1201 | WMMA, including FP8 | GEMM steps 6–7, attention development, Triton, PyTorch custom ops, small-model profiling |
| 3 — datacenter | Instinct MI300X, 192 GB HBM3, 5.3 TB/s ([getdeploying](https://getdeploying.com/reference/cloud-gpu/amd-mi300x)) | CDNA3 / gfx942 | MFMA | MFMA GEMM, FP8, attention, MoE, multi-GPU, the [[capstone]] |
| 3+ — stretch | Instinct MI355X | CDNA4 / gfx950 | MFMA with FP6/FP4 | MXFP4 GEMM and MoE in [[module-6-quantization-moe]] |

## Rung 1: the RX 5700 XT

**What it is good for:** everything that does not need matrix cores — indexing, LDS, cross-lane operations, reductions, softmax, normalization, profiling habits, and the first five GEMM optimizations. It is wave32-native, so it also teaches the wave-size portability that [[module-1-hip-model]] insists on.

**The catch:** gfx1010 is not on ROCm's official support list. "Unsupported" here means AMD won't take your bug reports, not that nothing runs:

- Debian 13 and Ubuntu 23.10+ ship distribution-built ROCm math libraries with gfx1010 enabled, and their packager reports that the rocBLAS and hipBLAS test suites passed on an RX 5700 XT ([ROCm issue #2527](https://github.com/ROCm/ROCm/issues/2527)).
- Community build scripts add unofficial gfx1010 support to ROCm itself ([ROCm-RDNA1](https://github.com/TheTrustedComputer/ROCm-RDNA1)).
- Do not expect official PyTorch wheels for gfx1010. Labs that need PyTorch move to rung 2.

## Rung 2: the RX 9070 XT

RDNA4 is on ROCm's official Linux support list (see the system requirements page in the official ROCm documentation), so PyTorch, Triton and the profilers work without workarounds. Its WMMA matrix instructions support FP8 in the OCP format, which makes it the only rung where you can compare the two FP8 encodings in [[module-6-quantization-moe]] side by side with MI300X.

## Rung 3: renting Instinct time

**Free credits first.** The AMD AI Developer Program gives members about $100 of AMD Developer Cloud credit, roughly 50 hours on a single MI300X ([guide](https://lzwjava.github.io/how-to-create-amd-gpu-droplets-en)). One listing reports the credits [expire after thirty days](https://startupintros.com/resources/amd-startups), so claim them at the start of [[module-4-gemm]], not in week 1.

**Then pay by the hour.** The median on-demand MI300X price was US$2.87 per GPU-hour across five providers in early November 2026 ([getdeploying](https://getdeploying.com/reference/cloud-gpu/amd-mi300x)). A pricing review from October 2026 lists offers from about $1.71/hr up to $6.00/hr, the top end being 8-GPU nodes on the big clouds ([Thunder Compute](https://www.thundercompute.com/blog/amd-mi300x-pricing)). Prices move monthly; re-check before you rent.

**Free leaderboard runs.** Some GPU MODE leaderboards time submissions on hosted AMD GPUs at no cost. When one is open, use it for timed runs and save paid hours for interactive profiling. See [[gpu-mode]].

## Budget

| Item | Estimate |
|---|---|
| Rungs 1 and 2 | Hardware you own |
| MI300X, weeks 7–12 | ~50 hours on credits, then ~30–60 paid hours |
| 8× MI300X node for [[module-8-shipping]] | 2–3 hours |
| MI355X stretch labs | A few hours, if available |
| **Total paid** | **roughly US$100–200** |

The estimate assumes you develop and debug on rungs 1–2 and only move to rung 3 for timed runs and profiling. Leaving an instance idle while you think is the fastest way to double it.

<script src="/tutor.js" defer></script>
