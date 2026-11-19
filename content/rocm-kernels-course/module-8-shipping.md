---
title: "Module 8 — Shipping Kernels, Multi-GPU, and a CUDA Bridge"
description: "Getting a kernel into PyTorch and a serving engine, overlapping RCCL collectives with compute on 8× MI300X, and porting two kernels to CUDA."
tags:
  - pytorch
  - vllm
  - rccl
  - multi-gpu
---

# Module 8 — Shipping kernels, multi-GPU, and a CUDA bridge

**1 week** · Part of [[index|Kernel Ladder]] · Prev: [[module-7-libraries]] · Next: [[capstone]]

> [!info] What this module is for
> A kernel only matters once it changes end-to-end latency or throughput in a real system. This module wires your kernels into PyTorch and a serving engine, scales one layer across eight GPUs, and ports two kernels to CUDA so the skills are visibly portable.

**Concepts:** PyTorch custom operators (`torch.library`) built against ROCm so they work with `torch.compile` and graphs; how vLLM and SGLang choose ROCm attention, GEMM and MoE backends; prefill vs decode and continuous batching; tensor and expert parallelism; RCCL all-reduce and all-to-all; [[rocm-kernels-course/concepts#Compute-communication overlap|overlapping communication with compute]].

---

## 8.1 Custom operators

Wrap your fused RMSNorm + FP8 quantize and your decode-attention kernel as PyTorch custom operators, with tests and a `torch.compile` smoke test, on the 9070 XT.

## 8.2 One kernel in a serving engine

Swap one of your kernels into vLLM or SGLang on MI300X for a small model. Report tokens/s and p50/p99 latency before and after, even if the change is small.

> [!success] Done when
> You have one end-to-end sentence you can defend: "my kernel moved model X from A to B tokens/s on MI300X."

## 8.3 Multi-GPU

1. If you have two GPUs in one machine, prototype tensor-parallel GEMM + RCCL all-reduce over PCIe at home.
2. Rent an 8× MI300X node for two to three hours. Measure the communication share in `rocprof-systems`, overlap it by chunking the GEMM, and show the timeline before and after.

Worked example of a multi-GPU kernel on Instinct hardware: [gau-nernst's all-to-all worklog](https://scour.ing/p/https://gau-nernst.github.io/amd-a2a) from a GPU MODE distributed challenge.

## 8.4 CUDA bridge (optional, no NVIDIA rental)

`hipify` only converts CUDA to HIP, so port by hand: take your softmax and your WMMA GEMM, rewrite them in CUDA, and compile with `nvcc`. Test them for an afternoon on a free notebook GPU.

> [!success] Done when
> Both kernels pass the same correctness tests in CUDA, and you have a half-page mapping table: MFMA/WMMA → `mma` / `wgmma`, LDS → shared memory, wave → warp, `rocprof-compute` → Nsight Compute.

## Check your understanding

- Your kernel is 30% faster in isolation but end-to-end throughput didn't move. Name three places to look.

<script src="/tutor.js" defer></script>
