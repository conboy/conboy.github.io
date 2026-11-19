---
title: "Capstone — A Fast Embeddings Encoder on MI300X"
description: "Building an LLM-based embedding model's inference path from your own kernels, benchmarked against Hugging Face Transformers and a serving engine."
tags:
  - capstone
  - embeddings
  - mi300x
---

# Capstone — a fast embeddings encoder

**2 weeks** · Part of [[index|Kernel Ladder]] · Prev: [[module-8-shipping]] · Next: [[interview-prep]]

> [!info] What this capstone is for
> Embedding models are a clean, commercially important target: no autoregressive loop, lots of variable-length batches, and a quality metric (cosine similarity to a reference) that makes numerics errors visible. Production embedding runtimes compete on exactly this workload; for one published account of the engineering involved, see [How Baseten built BEI](https://www.baseten.co/blog/how-we-built-high-throughput-embedding-inference-with-tensorrt-llm/).

---

## The build

Take an LLM-based embedding model (for example Qwen3-Embedding-0.6B) and build its inference path from your own kernels:

- fused RMSNorm + residual ([[module-2-memory-bound]])
- RoPE ([[module-2-memory-bound]])
- FP8 GEMMs ([[module-4-gemm]], [[module-6-quantization-moe]])
- attention over variable-length packed batches, with no padding ([[module-5-attention]])
- fused pooling and L2-normalize

## The benchmark

On MI300X, compare against Hugging Face Transformers on ROCm and a serving engine's ROCm embedding path:

| Measure | Batch sizes | Notes |
|---|---|---|
| Throughput (sequences/s) | 1, 32, 512 | Mixed sequence lengths |
| p50 / p99 latency | 1, 32 | Client-side and server-side |
| Cosine similarity vs bf16 reference | All | Report the minimum, not the mean |

Bonus: run the same build, more slowly, on the 9070 XT, and explain the gap with a roofline.

> [!success] Done when
> You have published portfolio post #3 with all three measures, the exact commands to reproduce them, and a profile showing where the remaining time goes.

## Open-source contributions

Aim for two or three merged pull requests; small ones count.

- vLLM or SGLang ROCm backends, [AITER](https://github.com/ROCm/aiter) or [Composable Kernel](https://github.com/ROCm/composable_kernel): a tuned configuration for a new shape, a kernel fix or a missing dtype.
- llama.cpp's HIP backend on consumer Radeon cards: RDNA performance work is under-served, and the 5700 XT and 9070 XT are exactly the hardware many users have.
- A [[gpu-mode|GPU MODE]] leaderboard entry with a public write-up.

<script src="/tutor.js" defer></script>
