---
title: "Concepts — Glossary"
description: "Plain-English definitions for every term in the Kernel Ladder course, from LDS and AGPRs to online softmax and FP8 FNUZ."
tags:
  - gpu
  - reference
---

# Concepts

The glossary for [[index|Kernel Ladder]]. Each entry ends with a _Used in_ line pointing at the module where it does real work.

## Execution and memory

### Wave

**A group of threads that execute in lockstep: 32 or 64 on RDNA, always 64 on CDNA.**

Cross-lane operations, reductions and ballots all depend on the wave size, which is why `warpSize` must never be hard-coded.

_Used in: [[module-1-hip-model]], [[module-2-memory-bound]]_

### LDS

**Local Data Share: fast on-chip memory shared by the threads of one workgroup.**

LDS is where tiles are staged for reuse. It is split into banks, and when several lanes hit the same bank in one access they serialize — a bank conflict.

_Used in: [[module-1-hip-model]], [[module-4-gemm]]_

### VGPR

**Vector general-purpose register: per-lane storage, the most precious resource in a kernel.**

The more VGPRs each thread uses, the fewer waves fit on a SIMD at once.

_Used in: [[module-1-hip-model]], [[module-4-gemm]]_

### AGPR

**Accumulation register on CDNA, used to hold MFMA results.**

On CDNA3 the register budget is shared between VGPRs and AGPRs, so a large MFMA accumulator tile competes with everything else for occupancy.

_Used in: [[module-4-gemm]]_

### Occupancy

**How many waves are resident on a SIMD compared with the maximum.**

Higher occupancy hides memory latency, but a kernel that uses more registers to reuse data can be faster at lower occupancy. Occupancy is a means, not the goal.

_Used in: [[module-1-hip-model]]_

### Coalescing

**Adjacent lanes reading adjacent addresses, so the hardware can merge their accesses into a few wide transactions.**

_Used in: [[module-1-hip-model]], [[module-2-memory-bound]]_

### Roofline

**A plot of attainable FLOP/s against arithmetic intensity (FLOPs per byte moved).**

Below the ridge point a kernel is limited by memory bandwidth; above it, by compute. Placing a kernel on the roofline tells you which optimizations can possibly help.

_Used in: [[index]], [[module-3-profiling]]_

## Matrix cores

### WMMA

**Wave Matrix Multiply-Accumulate: the matrix instructions on RDNA3 and RDNA4.**

One instruction multiplies small matrix tiles held across a wave's registers. RDNA4 adds FP8 inputs. The fragment layout — which lane holds which element — is fixed per instruction.

_Used in: [[module-4-gemm]], [[module-5-attention]]_

### MFMA

**Matrix Fused Multiply-Add: the matrix instructions on CDNA (Instinct).**

Much higher throughput than WMMA, wave64 only, with results in AGPRs. CDNA4 adds scaled FP8/FP6/FP4 variants.

_Used in: [[module-4-gemm]], [[module-6-quantization-moe]]_

### GEMM

**General matrix multiply, C = αAB + βC: most of the arithmetic in a transformer.**

_Used in: [[module-4-gemm]]_

### Grouped GEMM

**Many independent GEMMs of different sizes in one launch, as in a mixture-of-experts layer where each expert gets a different number of tokens.**

_Used in: [[module-6-quantization-moe]]_

## Kernels and algorithms

### Kernel fusion

**Combining several operations into one kernel so intermediate results never round-trip through DRAM.**

For memory-bound operators this is often the largest available speedup.

_Used in: [[module-2-memory-bound]], [[module-7-libraries]]_

### Online softmax

**Computing softmax in one pass by keeping a running maximum and rescaling the running sum when the maximum changes.**

It is the trick that lets FlashAttention process attention in tiles without storing the full score matrix.

_Used in: [[module-2-memory-bound]], [[module-5-attention]]_

### Split-KV

**Splitting a long KV cache across several workgroups during decode and reducing their partial results at the end.**

With one query token there is too little parallelism otherwise.

_Used in: [[module-5-attention]]_

### Paged KV cache

**Storing the KV cache in fixed-size pages addressed through a table, so memory can be allocated per request without fragmentation.**

_Used in: [[module-5-attention]]_

### Compute-communication overlap

**Splitting work into chunks so one chunk's collective (for example an all-reduce) runs while the next chunk computes.**

_Used in: [[module-8-shipping]]_

## Numbers

### FP8 FNUZ vs OCP

**Two incompatible families of 8-bit float encodings.**

MI300X (CDNA3) uses the FNUZ variants (finite, no negative zero), which have a different exponent bias and NaN handling from the OCP FP8 formats used by RDNA4 and MI355X. The same byte decodes to a different value, so kernels and conversion code must know which one they are handling.

_Used in: [[module-2-memory-bound]], [[module-6-quantization-moe]]_

### MXFP4

**A micro-scaled 4-bit float format: blocks of 32 values share one scale factor.**

Block scaling keeps 4-bit values usable by letting each small block choose its own range.

_Used in: [[module-6-quantization-moe]]_

<script src="/tutor.js" defer></script>
