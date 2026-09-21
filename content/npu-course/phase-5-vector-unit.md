---
title: "Phase 5 — The Vector Unit"
description: "RMSNorm, softmax, SiLU and RoPE in fixed point — and Amdahl's Law arriving on schedule."
tags:
  - npu
  - hardware
---

# Phase 5 — Everything that isn't matmul

**4 weeks** · Part of [[index|Zero to NPU]] · Prev: [[phase-4-memory]] · Next: [[phase-6-sequencer]]

> [!warning] Amdahl's ambush
> You made matmul 5× faster. The other 10% is now **35%** of runtime. This is the phase [[phase-1-baseline-llm|your week-4 profile]] warned you about.

---

## 5.1 The vector unit

Build a small shared datapath for element-wise work — ~12 DSPs and a few BRAMs for lookup tables.

| Op | Approach | Note |
|---|---|---|
| **RMSNorm** | Sum of squares → reciprocal-sqrt via LUT + one Newton–Raphson step | Needs int32 accumulate; watch overflow on the sum |
| **Softmax** | Subtract row max, exp via 256-entry LUT, sum, reciprocal | Two-pass, or implement *online softmax* (Milakov & Gimelshein) for one pass |
| **SiLU / SwiGLU** | 256-entry int8→int8 LUT | One BRAM. Free. |
| **RoPE** | Precompute sin/cos tables per position in BRAM | Compute once at init, **not** per token |

- [ ] RMSNorm
- [ ] Softmax
- [ ] SiLU / SwiGLU
- [ ] RoPE

**Done when:** each op matches the golden model bit-exactly, and end-to-end runtime drops by the amount your profiler predicted.

> [!danger] Gotcha — softmax overflow
> Softmax **without max-subtraction** overflows and gives you NaN-equivalents in fixed point.
>
> Do not skip that step because "the values are small." $e^x$ in int32 saturates faster than your intuition expects.
