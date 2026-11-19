---
title: "The GPU MODE Track"
description: "Pairing every module with a public GPU MODE leaderboard problem: free hosted timing, past solutions to study, and a community to learn from."
tags:
  - gpu-mode
  - leaderboards
  - community
---

# The GPU MODE track

Runs alongside every module · Part of [[index|Kernel Ladder]]

> [!info] What this track is for
> GPU MODE is a practice arena, an independent benchmark referee and a peer group in one place. Every module ends with one of its problems as well as your own lab. A leaderboard time is a public, independently measured number, which is harder to dismiss than a self-reported benchmark.

---

## How it works

You submit kernels to a Discord bot, or with the [popcorn CLI](https://github.com/gpu-mode/popcorn-cli), and it runs and times them on cloud GPUs, so you don't need your own hardware. Rankings are kept separately per algorithm and per device, and [submitting is free](https://gpu-mode.github.io/kernelbot/docs/intro). Past AMD rounds ran LLM inference kernels on [MI300s provided free through the Discord](https://kernelboarddev-8e4b32634dad.herokuapp.com/news).

Whenever an AMD leaderboard is open, use it for timed Instinct runs and keep your rented hours for interactive profiling, where you need a shell.

## The workflow for every problem

1. Develop and check correctness on the 9070 XT or 5700 XT.
2. Profile locally.
3. Submit for the official time.
4. After the deadline, read the top solutions and write down one technique you missed.

## Problem for each module

| Module | Problem | Folder in [reference-kernels](https://github.com/gpu-mode/reference-kernels) |
|---|---|---|
| [[module-0-setup]] | Vector add, to learn the submission flow | `pmpp_v2` |
| [[module-1-hip-model]], [[module-2-memory-bound]] | Grayscale, histogram, prefix sum, sort, vector add ([the standard practice set](https://developer.nvidia.com/blog/topping-the-gpu-mode-kernel-leaderboard-with-nvidia-cuda-compute)) | `pmpp_v2` |
| [[module-4-gemm]] | FP8 GEMM on MI300 | `amd` |
| [[module-5-attention]] | MLA decode | `amd_202602` |
| [[module-6-quantization-moe]] | MoE on MI300; MXFP4 GEMM and MXFP4 MoE on MI355X | `amd`, `amd_202602` |
| [[module-8-shipping]] | All-to-all and other multi-GPU kernels | `amd_distributed` |

The `amd_202602` problems are the qualifier kernels of the [AMD × GPU MODE E2E Model Speedrun](https://luma.com/cqq4mojz) (MXFP4 MoE, MLA decode, MXFP4 GEMM). Closed competitions may not accept new leaderboard entries. If so, run the problem's reference harness yourself on rented MI300X time and compare against the published winning times.

## Weekly habits

- [ ] Watch one GPU MODE lecture that matches the current module. The archive is in [gpu-mode/lectures](https://github.com/gpu-mode/lectures).
- [ ] Post your module write-up in the relevant Discord channel and ask for one round of feedback.
- [ ] Study one past solution. The [kernelbot-data dataset](https://huggingface.co/datasets/GPUMODE/kernelbot-data) holds real submissions, including AMD MI300 ones.
- [ ] In week 12, find two mock-interview partners and pitch a short community talk on your [[capstone]].

## Three rules

1. **Read each competition's terms before entering.** Sponsored rounds can exclude the sponsor's employees or require approval. Practising on closed problems is always fine.
2. **Win honestly.** A public audit found [evaluator bypasses accepted on several leaderboards](https://gist.github.com/josusanmartin/5cc5f696fe827cf5ba3e1da4355e1565), including MI355X ones. Anyone reviewing your work will read the code, so a gamed rank costs more than it earns.
3. **Use agents, but own the result.** Competitors now drive the popcorn CLI with AI agent loops; one placed [12th of 183 after more than 1,500 Codex-driven submissions](https://runtimewire.com/article/sankalp-codex-gpu-mode-qr-kernel). That is a legitimate tool, as long as you can explain every line you submit.

<script src="/tutor.js" defer></script>
