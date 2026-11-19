---
title: "Module 7 — ROCm Kernel Libraries"
description: "Composable Kernel, Triton, AITER, hipBLASLt, hipCUB and rocWMMA: when to reach for a library, and how to bend one."
tags:
  - composable-kernel
  - triton
  - aiter
  - libraries
---

# Module 7 — ROCm kernel libraries

**1 week** · Part of [[index|Kernel Ladder]] · Prev: [[module-6-quantization-moe]] · Next: [[module-8-shipping]]

> [!info] What this module is for
> Production teams rarely hand-write every kernel. Knowing which library to reach for, and how to extend it with a custom epilogue or a new shape, is most of the day-to-day job. You'll rebuild earlier kernels in two libraries and compare them, with numbers, against your hand-written HIP.

| Library | What it is | When you reach for it |
|---|---|---|
| [Composable Kernel](https://github.com/ROCm/composable_kernel) (CK / CK-Tile) | C++ templates for GEMM, attention and fused operators | Production GEMM and attention variants with custom epilogues |
| [Triton](https://triton-lang.org/main/getting-started/tutorials/index.html) on ROCm | Block-level Python DSL compiled to AMDGPU | Fast iteration on fused kernels |
| [AITER](https://github.com/ROCm/aiter) | A library of tuned inference kernels (assembly, CK and Triton) used by serving engines | Reading state-of-the-art Instinct kernels; contribution target |
| hipBLASLt / rocBLAS | Vendor GEMM libraries | The baseline for every GEMM benchmark |
| hipCUB / rocPRIM | Block, warp and device-wide primitives | Reductions, scans and sorts inside your own kernels |
| rocWMMA | C++ wrapper over WMMA and MFMA | Matrix-core code portable across RDNA and CDNA |

---

## Labs

1. Rebuild your [[module-4-gemm]] kernel in CK-Tile for MI300X with a custom epilogue (bias + GELU + FP8 quantize), and compare it with your hand-written step 9.
2. Write fused RMSNorm and a GEMM in Triton on the 9070 XT, with autotuning. Compare lines of code and speed against your HIP versions.
3. Read one AITER kernel end to end — an MoE or attention kernel. Write one page explaining its tiling, pipelining and instruction scheduling.
4. Replace the reduction in your [[module-2-memory-bound|softmax]] with `hipcub::BlockReduce` and confirm no regression.

> [!success] Done when
> Your repo has a decision guide — hand-written HIP vs CK vs Triton vs rocWMMA — where every recommendation is backed by one of your own benchmark numbers.

## Check your understanding

- When would you choose Triton over CK for a new fused operator, and when would that choice hurt you?

<script src="/tutor.js" defer></script>
