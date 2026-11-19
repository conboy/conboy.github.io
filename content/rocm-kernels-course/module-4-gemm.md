---
title: "Module 4 — GEMM, From Naive to Matrix Cores"
description: "One matrix multiply built in ten steps, climbing from an RX 5700 XT through RDNA4 WMMA to MI300X MFMA, scored against rocBLAS and hipBLASLt at every step."
tags:
  - gemm
  - wmma
  - mfma
  - matrix-cores
---

# Module 4 — GEMM, from naive to matrix cores

**3 weeks** · Part of [[index|Kernel Ladder]] · Prev: [[module-3-profiling]] · Next: [[module-5-attention]]

> [!info] What this module is for
> Matrix multiply is most of the arithmetic in a transformer, and it is the kernel interviewers most often ask you to walk through. You'll build one kernel in ten steps and keep every step, so you can say exactly what each optimization bought. The steps follow the hardware ladder: the first five need no matrix cores, so they run on the 5700 XT.

**Problem:** bf16 inputs, fp32 accumulation, M = N = K = 4096 (fp32 for steps 1–5 on the 5700 XT). Compare every step against rocBLAS or hipBLASLt on the same card.

---

## The ladder

1. **5700 XT — naive.** One thread per output element.
2. **5700 XT — coalescing.** Fix the memory access order.
3. **5700 XT — LDS tiling.** Load tiles of A and B into [[rocm-kernels-course/concepts#LDS|LDS]] once, reuse them many times.
4. **5700 XT — register blocking.** Each thread computes a micro-tile, first 1D then 2D.
5. **5700 XT — vector loads and double buffering.** 128-bit loads, an LDS layout without bank conflicts, and global→register→LDS prefetch of the next tile while computing the current one. RDNA1 has no matrix cores, so this is as far as rung 1 goes.
6. **9070 XT — WMMA.** Move the inner loop onto [[rocm-kernels-course/concepts#WMMA|WMMA]] with the `__builtin_amdgcn_wmma_*` intrinsics, using the gfx12 variants (RDNA3's gfx11 names and layouts differ). Learn the fragment layouts from the [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator), not by guessing.
7. **9070 XT — FP8 WMMA.** The same kernel with FP8 inputs.
8. **MI300X — MFMA.** Port to [[rocm-kernels-course/concepts#MFMA|MFMA]]: wave64, accumulators in [[rocm-kernels-course/concepts#AGPR|AGPRs]], and LDS layouts that feed MFMA operands without conflicts.
9. **MI300X — pipelining and scheduling.** Direct-to-LDS loads, a deeper pipeline, a persistent kernel with a tile scheduler, and split-K or stream-K for the skinny shapes decode produces (M = 1–64).
10. **Stretch, MI355X — scaled MFMA.** An FP8/FP4 kernel using CDNA4's scaled matrix instructions and larger LDS, even if it is slower than hipBLASLt.

**Concepts:** arithmetic intensity of a tile; choosing tile sizes against the VGPR, AGPR and LDS budgets; MFMA vs WMMA throughput and data layouts; `s_waitcnt` and instruction scheduling; epilogue fusion (bias, activation, scaling); confirming in the ISA that `v_mfma` / `v_wmma` instructions are actually issued.

---

## Labs

> [!abstract] Steps 1–5, on the 5700 XT
> **Done when** step 5 reaches 50%+ of rocBLAS fp32 SGEMM at 4096³ on the same card.

> [!abstract] Steps 6–7, on the 9070 XT
> **Done when** step 6 reaches 60%+ of hipBLASLt bf16 at 4096³, and step 7 passes an FP8 correctness check against a bf16 reference.

> [!abstract] Steps 8–9, on MI300X
> Use your free cloud credits here. **Done when** step 9 reaches 60%+ of hipBLASLt bf16 at 4096³. Correctness first, every time.

> [!abstract] Shape sweep, on MI300X
> Plot TFLOP/s for your best kernel and hipBLASLt at M ∈ {1, 16, 64, 256, 4096} with N = K = 4096. **Done when** you can explain, from a profile, where yours loses.

> [!success] Portfolio post #2
> "Building a GEMM from an RX 5700 XT to an MI300X in ten steps," with a chart of TFLOP/s per step and the percent of the vendor library each step reached.

> [!question]- It's not working
> 1. Wrong answers only on some tiles: check the fragment layout against the matrix instruction calculator. Layout bugs often pass small tests and fail large ones.
> 2. No speedup from WMMA/MFMA: confirm in the `--save-temps` ISA that the matrix instructions were emitted and are not starved by `s_waitcnt` on global loads.
> 3. Performance falls off a cliff at one tile size: check `-Rpass-analysis` for register spills.

## Check your understanding

- Why does register blocking raise arithmetic intensity, and which resource does it trade away?
- Why is M = 1 a memory-bound problem even though it is a "matrix multiply"?

<script src="/tutor.js" defer></script>
