---
title: "Module 0 — Toolchain and Benchmark Harness"
description: "ROCm on two Radeon cards, a benchmark harness you can trust, and a first GB/s number measured against peak."
tags:
  - rocm
  - hip
  - toolchain
  - benchmarking
---

# Module 0 — Toolchain and benchmark harness

**1 week (~5 hours)** · Part of [[index|Kernel Ladder]] · Next: [[module-1-hip-model]]

> [!info] What this module is for
> Every later number in this course comes out of the harness you build here. A harness that forgets to warm up, times a cold cache, or skips the correctness check will produce confident, wrong numbers for twelve weeks. This module is short on purpose, but nothing after it is trustworthy until it is done.

---

## 0.1 Install ROCm on both cards

> [!abstract] Goal
> `rocminfo` lists gfx1010 (RX 5700 XT) and gfx1201 (RX 9070 XT).

1. Install a Linux release listed on the current ROCm system requirements page and install ROCm for the 9070 XT from the official instructions.
2. For the 5700 XT, use your distribution's ROCm packages (Debian 13 / Ubuntu 23.10+ build the math libraries for gfx1010) or the community build scripts. See [[hardware-specs#Rung 1: the RX 5700 XT|hardware-specs]] for both routes.
3. Add yourself to the `render` and `video` groups and reboot.
4. Run `rocminfo | grep gfx` and confirm both targets appear.

> [!success] Done when
> `rocminfo` lists both gfx targets, and a trivial HIP kernel compiled with `--offload-arch=gfx1010,gfx1201` runs on each card.

## 0.2 Create the repo

Create a public repository, for example `rocm-kernels-from-scratch`:

```text
bench/        harness, reference implementations, tolerance table
m1-hip/       one folder per module
...
RESULTS.md    every number, with GPU, gfx target, ROCm version, commit hash
```

Every row in `RESULTS.md` records the GPU, gfx target, ROCm version and commit. A number without those is not reproducible, even by you next month.

## 0.3 Build the harness

> [!abstract] Goal
> One Python entry point that checks correctness, then times a kernel properly.

The harness must:

1. Check the output against a PyTorch or NumPy reference with explicit `atol`/`rtol` per dtype (for example fp32 `1e-5`, bf16 `1e-2`, FP8 a cosine-similarity floor). Fail loudly; never time an incorrect kernel.
2. Warm up (at least 10 launches), then time 100 launches with `hipEvent` pairs and report the median and the spread.
3. Flush the L2 cache and Infinity Cache between timed runs by touching a buffer larger than both. Otherwise small tensors stay resident and the numbers flatter you.
4. Report achieved GB/s and TFLOP/s next to the card's peak, so every result is a percentage.

## 0.4 Build tooling

Write a `Makefile` (or CMake) target that:

- builds for several targets at once: `--offload-arch=gfx1010,gfx1201,gfx942,gfx950`
- keeps the intermediate ISA with `--save-temps` so you can read the instructions the compiler emitted
- prints register and LDS usage with `-Rpass-analysis=kernel-resource-usage`
- runs `rocprofv3` on a kernel and saves the trace next to the source

> [!success] Done when
> One command builds and runs vector add on both Radeon cards and prints achieved GB/s as a percentage of each card's peak memory bandwidth. Record both numbers in `RESULTS.md`.

> [!question]- It's not working
> 1. `hipErrorNoBinaryForGpu`: you did not build for that card's gfx target. Check `--offload-arch`.
> 2. Vector add reaches well under half of peak: check you are timing the kernel, not the host-to-device copy, and that you warmed up.
> 3. The 5700 XT disappears after a ROCm upgrade: an official package update can replace the gfx1010-enabled distribution libraries. Pin the packages you rely on.

## Check your understanding

- Why does flushing the cache matter more for a 1 MB tensor than a 1 GB one?
- Why report the median rather than the mean?

<script src="/tutor.js" defer></script>
