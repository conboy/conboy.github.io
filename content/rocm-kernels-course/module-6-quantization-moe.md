---
title: "Module 6 — Quantization and Mixture-of-Experts"
description: "FP8 in two encodings, int4 weights, MXFP4 on CDNA4, and a full mixture-of-experts layer profiled stage by stage."
tags:
  - fp8
  - mxfp4
  - quantization
  - moe
---

# Module 6 — Quantization and mixture-of-experts

**1 week** · Part of [[index|Kernel Ladder]] · Prev: [[module-5-attention]] · Next: [[module-7-libraries]]

> [!info] What this module is for
> Low-precision formats and mixture-of-experts layers are where inference cost is won or lost. They are also where numerics bugs hide: a kernel can be fast, pass a loose test and still quietly degrade a model. This module treats accuracy testing as part of the kernel, not an afterthought.

**Concepts:** FP8 E4M3 and E5M2; [[rocm-kernels-course/concepts#FP8 FNUZ vs OCP|FNUZ vs OCP FP8]] — MI300X uses the FNUZ variants, while the 9070 XT and MI355X use OCP FP8, so the same bytes decode to different values; per-tensor, per-token, per-channel and block (128×128) scaling; [[rocm-kernels-course/concepts#MXFP4|MXFP4/MXFP6]] with a shared scale per 32-element block; where dequantization happens; accuracy checks beyond max-absolute error (perplexity deltas, embedding cosine similarity). For MoE: top-k gating, token permutation, [[rocm-kernels-course/concepts#Grouped GEMM|grouped GEMM]] over uneven experts, unpermute and combine.

---

## 6.1 FP8 GEMM on MI300X

Per-token activation scales and per-channel weight scales, fused into your [[module-4-gemm]] epilogue. Compare against hipBLASLt FP8.

> [!success] Done when
> Your FP8 GEMM passes a cosine-similarity check against bf16 and you have its TFLOP/s as a percentage of hipBLASLt FP8.

## 6.2 The same kernel in OCP FP8 on the 9070 XT

Rebuild 6.1 with FP8 WMMA on RDNA4. Write up what changed in the code, and the test you used to prove both encodings give the same answers after dequantization.

## 6.3 W4A16 decode GEMM

M ≤ 16. Unpack int4 weights in registers, dequantize, feed WMMA or MFMA. Score it in GB/s of weight traffic, since it is memory-bound.

## 6.4 A mixture-of-experts layer

64 experts, top-8 routing:

1. Routing and permutation kernel.
2. Grouped GEMM over the experts.
3. Unpermute and weighted combine.

> [!success] Done when
> You have profiled the full layer on MI300X at batch 1 and batch 256 and can name the dominant stage at each.

## 6.5 Stretch: MXFP4 on MI355X

MXFP4 GEMM and MXFP4 MoE were both qualifier kernels in the [AMD × GPU MODE E2E Model Speedrun](https://luma.com/cqq4mojz). Problem definitions are in [gpu-mode/reference-kernels](https://github.com/gpu-mode/reference-kernels). See [[gpu-mode]].

## Check your understanding

- A model's perplexity rises 2% after switching an FP8 kernel from MI300X to an OCP-FP8 GPU, with no code change. What is the first thing you check?
- Why does the grouped GEMM dominate MoE at large batch but not at batch 1?

<script src="/tutor.js" defer></script>
