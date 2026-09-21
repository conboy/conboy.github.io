---
title: "Phase 2 — Fixed-Point Numerics"
description: "Q-formats and the requantization pipeline: the place where FPGA ML projects silently produce garbage."
tags:
  - npu
  - hardware
---

# Phase 2 — Numerics

**2 weeks** · Part of [[index|Zero to NPU]] · Prev: [[phase-1-baseline-llm]] · Next: [[phase-3-mac-engine]]

The place where FPGA ML projects silently produce garbage.

---

## 2.1 Fixed-point fluency

- [ ] Build a Q-format calculator
- [ ] Given Q1.15 × Q1.15, what's the output format? Where does the binary point land after accumulation?

**Done when:** you can do this on a whiteboard without thinking.

---

## 2.2 The requantization pipeline

This is the operation between every matmul and the next layer:

```
int32 acc  →  ×M0 (fixed-point multiplier)  →  >>n (rounding shift)
           →  saturate to int8
```

The multiplier `M0` is a 32-bit fixed-point value derived from the float scale; `n` is a right shift.

- [ ] Implement it in C
- [ ] Implement it in Verilog
- [ ] Cross-validate

**Done when:** C and Verilog agree **bit-exactly** on 1,000,000 random int32 inputs across 100 different `(M0, n)` pairs.

> [!danger] The bug that will make your model output mush
> **Rounding.** Use round-half-away-from-zero and make sure your Verilog does the same thing as your C for ==negative numbers==.
>
> Arithmetic right shift of a negative number rounds toward $-\infty$, **not** toward zero. This one bug produces garbage output with no obvious cause and no crash to debug.
