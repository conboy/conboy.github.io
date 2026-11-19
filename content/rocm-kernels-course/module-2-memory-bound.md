---
title: "Module 2 — Memory-Bound Kernels"
description: "Reductions, softmax, RMSNorm and RoPE — the non-GEMM kernels of an inference stack — scored as a percentage of peak memory bandwidth."
tags:
  - hip
  - reductions
  - softmax
  - bandwidth
---

# Module 2 — Memory-bound kernels

**2 weeks** · Part of [[index|Kernel Ladder]] · Prev: [[module-1-hip-model]] · Next: [[module-3-profiling]]

> [!info] What this module is for
> Most of an inference step that is not a matrix multiply is memory-bound: it reads a tensor, does a little arithmetic and writes it back. These kernels need no matrix cores, so the 5700 XT handles almost all of this module. The score is always percent of peak bytes per second.

**Concepts:** coalescing and 128-bit vector loads; buffer vs global loads; cross-lane operations (`__shfl_xor`, DPP and `ds_swizzle` through compiler builtins) written to work at both wave32 and wave64; block reductions; [[rocm-kernels-course/concepts#Online softmax|online softmax]]; [[rocm-kernels-course/concepts#Kernel fusion|kernel fusion]]; why Infinity Cache flatters small-tensor benchmarks.

---

## 2.1 The reduction ladder

> [!abstract] Goal
> Sum 64M floats within 10% of rocPRIM on the 9070 XT.

Write six versions and record GB/s for each:

1. Naive: every thread atomically adds into one global value.
2. LDS tree reduction within a block.
3. Cross-lane (shuffle) reduction within a wave, LDS only between waves.
4. 128-bit vector loads.
5. Several elements per thread (grid-stride loop).
6. Compare against `hipcub::DeviceReduce` (rocPRIM underneath).

> [!success] Done when
> Version 5 is within 10% of rocPRIM on the 9070 XT, and the same source runs correctly on the 5700 XT.

## 2.2 Softmax

> [!abstract] Goal
> Row-wise softmax that is wave-size agnostic and competitive with PyTorch on at least one shape class.

Shapes: (4096, 32000) for long rows and (32, 128) for short rows. Write three-pass, two-pass and online (single-pass) variants. Use one wave per row for short rows and one block per row for long rows.

## 2.3 Fused RMSNorm + residual, then FP8 output

Inputs bf16, accumulate in fp32, output bf16. Then add an FP8 quantize step on the output with a per-token scale. Keep this kernel: [[module-6-quantization-moe]] reuses it, and the FP8 encoding differs between MI300X and the 9070 XT (see [[rocm-kernels-course/concepts#FP8 FNUZ vs OCP|FP8 FNUZ vs OCP]]).

## 2.4 RoPE

Apply rotary position embeddings to Q and K in place, in one kernel.

> [!success] Done when
> `RESULTS.md` has a table of GB/s, percent of peak and speedup over PyTorch eager for each kernel, per card, with the tolerance used for each correctness check.

## Check your understanding

- Why is two-pass softmax slower than online softmax for long rows but sometimes not for short ones?
- Your RMSNorm reaches 105% of "peak" bandwidth on a small tensor. What happened?

<script src="/tutor.js" defer></script>
