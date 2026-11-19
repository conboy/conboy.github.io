---
title: "Module 1 — The HIP Compute Model Across RDNA and CDNA"
description: "Thinking as a compute-kernel author: waves, LDS, registers and occupancy, and what changes when a kernel moves from a Radeon card to an Instinct GPU."
tags:
  - hip
  - rdna
  - cdna
  - occupancy
---

# Module 1 — The HIP compute model across RDNA and CDNA

**1–2 weeks** · Part of [[index|Kernel Ladder]] · Prev: [[module-0-setup]] · Next: [[module-2-memory-bound]]

> [!info] What this module is for
> The same HIP source compiles for a 5700 XT and an MI300X, but the two run it very differently. This module builds the mental model you need to predict those differences before you measure them: wave size, LDS, the register file, and occupancy.

---

## What changes between the rungs

| Topic | RDNA1 (RX 5700 XT) | RDNA4 (RX 9070 XT) | CDNA3 / CDNA4 (MI300X / MI355X) |
|---|---|---|---|
| Wave size | wave32 native, wave64 supported | wave32 native, wave64 supported | wave64 only |
| Matrix units | None | [[rocm-kernels-course/concepts#WMMA|WMMA]], including FP8 | [[rocm-kernels-course/concepts#MFMA|MFMA]]; CDNA4 adds FP6/FP4 |
| Occupancy limited by | VGPRs per wave, LDS per workgroup | Same | VGPRs plus [[rocm-kernels-course/concepts#AGPR|AGPRs]], LDS per CU |
| Memory | GDDR6 | GDDR6 + Infinity Cache | HBM + Infinity Cache, chiplet layout |
| HIP target | gfx1010 | gfx1201 | gfx942 / gfx950 |

Exact per-card figures (CU counts, LDS size, cache sizes) are in the GPU hardware specifications table in the official ROCm documentation. Look them up rather than trusting a blog.

**Concepts to own:** grid, block and thread indexing; why `warpSize` must never be hard-coded to 32; [[rocm-kernels-course/concepts#LDS|LDS]] and bank conflicts; [[rocm-kernels-course/concepts#Occupancy|occupancy]]; `__syncthreads`, fences and atomics; reading the ISA the compiler emits (`s_waitcnt`, `buffer_load`, `ds_read`, `v_mfma`).

---

## 1.1 Three kernels from memory

> [!abstract] Goal
> Write vector add, SAXPY and 2D transpose in HIP without copying from anywhere.

Run each on the 5700 XT and the 9070 XT, log GB/s per card, and add MI300X numbers once you have Instinct time.

## 1.2 Transpose and bank conflicts

> [!abstract] Goal
> Make the LDS-tiled transpose reach 80%+ of peak bandwidth.

1. Write a naive transpose. Note that either the reads or the writes are uncoalesced.
2. Stage a tile through LDS so both global reads and writes coalesce.
3. Measure LDS bank conflicts with `rocprofv3` counters. Pad the tile by one element and measure again.

> [!success] Done when
> The padded LDS transpose reaches 80%+ of peak bandwidth on the 9070 XT, and you can explain every instruction in its ISA hot loop.

## 1.3 Find a race

ROCm has no race detector equivalent to CUDA's `racecheck`. So:

1. Write a histogram with a deliberate race (non-atomic increments to shared bins).
2. Write a deterministic test that exposes it: a skewed input where many threads hit one bin, compared against a CPU count.
3. Fix it twice: global atomics, then per-block LDS histograms merged at the end. Compare the speed of both.

## 1.4 Wave size

On the 9070 XT, compile one warp-level reduction for wave32 and again with `-mwavefrontsize64`, and explain the throughput difference. Then make the code correct for wave64-only CDNA without `#ifdef` per target.

## 1.5 Occupancy is not the goal

Sweep block size from 64 to 1024 and vary `__launch_bounds__` on one kernel. Record VGPRs per thread (from `-Rpass-analysis=kernel-resource-usage`), waves per SIMD and throughput.

> [!success] Done when
> You have a plot of occupancy vs throughput and can explain why the fastest point is not the highest-occupancy point.

## Check your understanding

- A kernel uses 128 VGPRs per thread. What happens to occupancy if you add 8 more, and why can that make it faster or slower?
- Why does a reduction written for 32 lanes give wrong answers on CDNA rather than just running slowly?

<script src="/tutor.js" defer></script>
