---
title: "Module 3 — Profiling on ROCm"
description: "rocprofv3, rocprof-compute, rocprof-systems and the PyTorch profiler: telling a bottleneck story with numbers."
tags:
  - rocm
  - profiling
  - roofline
---

# Module 3 — Profiling on ROCm

**1 week** · Part of [[index|Kernel Ladder]] · Prev: [[module-2-memory-bound]] · Next: [[module-4-gemm]]

> [!info] What this module is for
> A kernel engineer's real job is explaining why something is slow, with evidence. This module covers the four tools you need and ends with a public write-up: where the time goes in one decode step of a small language model.

| Tool | Question it answers | Works on |
|---|---|---|
| `rocprofv3` ([rocprofiler-sdk](https://github.com/ROCm/rocprofiler-sdk)) | Which kernels ran, for how long, with which hardware counters | Radeon and Instinct |
| `rocprof-compute`, formerly Omniperf ([repo](https://github.com/ROCm/rocprofiler-compute)) | Why one kernel is slow: roofline, memory chart, cache hit rates, stalls | Mainly Instinct; check Radeon support for your ROCm release |
| `rocprof-systems`, formerly Omnitrace ([repo](https://github.com/ROCm/rocprofiler-systems)) | The timeline: CPU–GPU gaps, launch overhead, stream overlap, collectives | Radeon and Instinct |
| PyTorch Profiler | Which operators dominate a model step | Any card with ROCm PyTorch |

**Concepts:** the [[rocm-kernels-course/concepts#Roofline|roofline]]; launch overhead and HIP Graphs; how counters map onto the memory hierarchy; reading ISA-level hotspots.

---

## 3.1 Where the time goes

> [!abstract] Goal
> A per-operator breakdown of prefill and decode for a ~0.6B–1B parameter model on the 9070 XT.

Generate 256 tokens with ROCm PyTorch and list the top 10 operators by GPU time, split into prefill and decode.

## 3.2 Launch gaps

Run the same generation under `rocprof-systems`. Measure the idle time between kernels during decode. Capture decode in a graph (`torch.cuda.graphs` works on ROCm builds) and measure tokens/s before and after.

## 3.3 One kernel on the roofline

Use your first hour of Instinct time to run `rocprof-compute` on your [[module-2-memory-bound|softmax]]. Place it on the roofline, name the top two limiters, change one thing, and show the profile move.

## 3.4 Your playbook

Write one page in the repo: which tool answers which question, and the ten counters you check first.

> [!success] Done when
> You have published a write-up titled "Where the time goes in a 1B-parameter decode step" with an annotated timeline and a roofline plot. This is portfolio post #1.

## Check your understanding

- Decode spends 40% of its wall time with the GPU idle. Which tool shows that, and what are the two most likely causes?
- Why can a kernel at 30% of peak bandwidth still be "done"?

<script src="/tutor.js" defer></script>
