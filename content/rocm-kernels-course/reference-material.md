---
title: "Reference Material"
description: "The documentation, repositories, papers and pricing sources behind Kernel Ladder, tagged by the module that needs them."
tags:
  - gpu
  - reference
---

# Reference material

Part of [[index|Kernel Ladder]].

> [!tip] Pull these as you need them
> Each entry is tagged with the module where it becomes relevant. Reading everything first is a procrastination strategy.

---

## Documentation and tools

| Source | Needed by |
|---|---|
| The official ROCm documentation site: Linux system requirements, the HIP programming guide, and the GPU hardware specifications table | [[module-0-setup]], [[module-1-hip-model]] |
| [HIP on GitHub](https://github.com/ROCm/HIP) | [[module-1-hip-model]] |
| AMD's published ISA reference guides for RDNA4 and for CDNA3 / CDNA4 | [[module-4-gemm]] |
| [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator) | [[module-4-gemm]] |
| [rocprofiler-sdk](https://github.com/ROCm/rocprofiler-sdk) (`rocprofv3`), [rocprofiler-compute](https://github.com/ROCm/rocprofiler-compute), [rocprofiler-systems](https://github.com/ROCm/rocprofiler-systems) | [[module-3-profiling]] |
| [Composable Kernel](https://github.com/ROCm/composable_kernel), [AITER](https://github.com/ROCm/aiter) | [[module-7-libraries]], [[capstone]] |
| [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) | [[module-7-libraries]] |
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) and the FlashAttention-2 paper | [[module-5-attention]] |

## Hardware support and pricing

| Source | Claim it supports | Needed by |
|---|---|---|
| [ROCm issue #2527](https://github.com/ROCm/ROCm/issues/2527) | Distribution ROCm packages with gfx1010 enabled; test suites passed on an RX 5700 XT | [[hardware-specs]] |
| [ROCm-RDNA1](https://github.com/TheTrustedComputer/ROCm-RDNA1) | Community build scripts for RDNA1 | [[hardware-specs]] |
| [getdeploying: MI300X](https://getdeploying.com/reference/cloud-gpu/amd-mi300x) | MI300X specs; median on-demand price | [[hardware-specs]] |
| [Thunder Compute: MI300X pricing](https://www.thundercompute.com/blog/amd-mi300x-pricing) | Provider-by-provider MI300X prices | [[hardware-specs]] |
| [AMD Developer Cloud guide](https://lzwjava.github.io/how-to-create-amd-gpu-droplets-en), [StartupIntros](https://startupintros.com/resources/amd-startups) | Free credit amount and expiry | [[hardware-specs]] |

## GPU MODE

| Source | Needed by |
|---|---|
| [reference-kernels](https://github.com/gpu-mode/reference-kernels), [popcorn CLI](https://github.com/gpu-mode/popcorn-cli), [KernelBot docs](https://gpu-mode.github.io/kernelbot/docs/intro) | [[gpu-mode]] |
| [Lectures](https://github.com/gpu-mode/lectures), [kernelbot-data](https://huggingface.co/datasets/GPUMODE/kernelbot-data) | [[gpu-mode]] |
| [AMD × GPU MODE E2E Model Speedrun](https://luma.com/cqq4mojz) | [[module-5-attention]], [[module-6-quantization-moe]] |
| [Leaderboard evaluator audit](https://gist.github.com/josusanmartin/5cc5f696fe827cf5ba3e1da4355e1565) | [[gpu-mode]] |

## Books and worked examples

- _Programming Massively Parallel Processors_, 4th edition (Hwu, Kirk, El Hajj): the concepts behind [[module-1-hip-model]] and [[module-2-memory-bound]] carry over to HIP directly.
- [gau-nernst's AMD all-to-all worklog](https://scour.ing/p/https://gau-nernst.github.io/amd-a2a): a multi-GPU kernel on Instinct hardware, written up step by step ([[module-8-shipping]]).
- [How Baseten built BEI](https://www.baseten.co/blog/how-we-built-high-throughput-embedding-inference-with-tensorrt-llm/): a production embeddings runtime, background for the [[capstone]].

<script src="/tutor.js" defer></script>
