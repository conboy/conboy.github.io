---
title: "Module 5 — Attention Kernels"
description: "FlashAttention's forward pass with WMMA and then MFMA, plus the paged decode attention that dominates serving cost."
tags:
  - attention
  - flashattention
  - kv-cache
---

# Module 5 — Attention kernels

**1 week** · Part of [[index|Kernel Ladder]] · Prev: [[module-4-gemm]] · Next: [[module-6-quantization-moe]]

> [!info] What this module is for
> Attention is the second kernel every inference engine depends on. You'll write the FlashAttention forward pass yourself, first on the 9070 XT and then on MI300X, then the decode-time variant that reads a long KV cache for a single new token.

**Concepts:** why naive attention is memory-bound (the N×N score matrix); tiling Q, K and V with [[rocm-kernels-course/concepts#Online softmax|online softmax]] so scores never reach DRAM; FlashAttention-2 work partitioning; the different ROCm FlashAttention backends (Composable Kernel, Triton, AITER); causal masking; GQA, MQA and MLA; decode attention with [[rocm-kernels-course/concepts#Split-KV|split-KV]]; [[rocm-kernels-course/concepts#Paged KV cache|paged KV caches]].

---

## 5.1 FlashAttention-2 forward on the 9070 XT

> [!abstract] Goal
> A correct WMMA FlashAttention forward pass.

Head dimensions 64 and 128, causal and non-causal, bf16. Verify against PyTorch `scaled_dot_product_attention`.

## 5.2 Port to MFMA on MI300X

Benchmark against PyTorch SDPA and the ROCm build of [FlashAttention](https://github.com/Dao-AILab/flash-attention) at sequence lengths from 512 to 16K.

> [!success] Done when
> Your MI300X kernel reaches 50%+ of the library's throughput at sequence length 4K.

## 5.3 Paged decode attention

Single query token, a long KV cache stored in pages of 16 tokens, GQA ratio 8, split across the sequence and reduced at the end. This kernel is memory-bound, so score it in GB/s. Develop it on either Radeon card; time it on MI300X.

> [!success] Done when
> Decode attention reaches 70%+ of peak HBM bandwidth on MI300X.

## 5.4 Stretch: MLA decode

Multi-head latent attention decode was one of the kernels in the [AMD × GPU MODE E2E Model Speedrun](https://luma.com/cqq4mojz). Attempt it on MI300X and compare to the published problem's reference. See [[gpu-mode]].

## Check your understanding

- Why does online softmax let FlashAttention avoid ever writing the score matrix?
- Why does split-KV help decode but not prefill?

<script src="/tutor.js" defer></script>
