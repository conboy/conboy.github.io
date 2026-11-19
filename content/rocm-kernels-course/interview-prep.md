---
title: "Interview Prep for GPU Kernel Roles"
description: "Live-coding drills, architecture questions and a story bank for GPU kernel engineering interviews, built from the course's own labs."
tags:
  - interviews
  - kernels
---

# Interview prep

**Week 12** · Part of [[index|Kernel Ladder]] · Prev: [[capstone]]

> [!info] What this page is for
> Kernel interviews usually mix live coding, architecture questions, a performance-debugging case and a deep dive on your own work. Everything below comes from the course labs, so you are rehearsing work you have already done.

---

## Live-coding drills

Forty-five minutes each, plain HIP, no autocomplete. Repeat two of them in CUDA if you are interviewing at CUDA-first teams.

- [ ] Block reduction and row-wise softmax, correct at both wave32 and wave64 ([[module-2-memory-bound]])
- [ ] LDS-tiled GEMM, then explain how you would move it onto MFMA ([[module-4-gemm]])
- [ ] Transpose without LDS bank conflicts ([[module-1-hip-model]])
- [ ] Top-k over the last dimension, as in MoE gating ([[module-6-quantization-moe]])
- [ ] Prefix sum using cross-lane operations

## Questions to answer in under two minutes

1. Walk through an MI300X's memory hierarchy — registers, LDS, L1, L2, Infinity Cache, HBM — and what its chiplet layout means for kernels.
2. What limits occupancy on CDNA3, and when does higher occupancy hurt?
3. Wave32 vs wave64: what changes in your reduction and shuffle code?
4. MFMA vs WMMA: how do throughput, fragment layout and register use differ?
5. Why is decode memory-bound and prefill compute-bound, and what does that mean for kernel design?
6. Explain FlashAttention's online softmax and why it saves HBM traffic.
7. FP8 FNUZ vs OCP: what breaks if you mix them up? What does MXFP4 block scaling buy?
8. A `rocprof-compute` report shows low matrix-core utilization and high memory-wait time. What do you try?
9. A kernel is fast alone but slow inside the model. What do you check?
10. How would you overlap an all-reduce with a GEMM?

## Story bank

Prepare three stories, each as situation → action → numbers:

- a hard performance bug you found and fixed
- a time careful testing caught a numerics problem others missed
- one result from this course: the GEMM ladder or the capstone

## Mock interviews

Do at least two: one with a peer from [[gpu-mode|GPU MODE]] or a local meetup, and one by handing your capstone write-up to a reviewer and asking to be grilled on every number.

<script src="/tutor.js" defer></script>
