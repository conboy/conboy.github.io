---
title: "Phase 3 — The MAC Engine"
description: "Building an int8 systolic array from a single DSP48E1 up to a 32-PE 8x4 array at 100 MHz."
tags:
  - npu
  - hardware
---

# Phase 3 — The MAC engine

**5 weeks** · Part of [[index|Zero to NPU]] · Prev: [[phase-2-numerics]] · Next: [[phase-4-memory]]

The heart of the NPU.

> [!quote] As a graphics person
> Think of this as a **fixed-function unit, not a shader core** — it does one thing with zero flexibility, extremely fast.

---

## 3.1 One processing element

- [ ] Build a single PE: int8 × int8 → int32 accumulate
- [ ] Weight register holds its value across many activations (*weight-stationary*)

**Done when:** the Vivado synthesis report shows ==DSP48E1 = 1, LUT ≈ 0== for this module.

> [!bug] Gotcha
> If LUTs > 0, the tool didn't infer the DSP. Check that your multiply is **registered on both input and output** — DSP inference needs the pipeline registers to map into the slice.

---

## 3.2 A 4×4 output-stationary systolic array

- [ ] 16 PEs in a grid
- [ ] Activations flow left→right, weights flow top→bottom
- [ ] Each PE accumulates one output element in place

**Done when:** a Verilator testbench multiplies random 4×4 matrices and matches the golden model exactly.

> [!bug] Gotcha — data skewing
> Row $i$ of the activation matrix must enter $i$ cycles later than row 0, so operands meet at the right PE on the right cycle.
>
> **Draw the timing diagram by hand for a 2×2 array before you write the 4×4.**

---

## 3.3 Scale to 8×4 (32 PEs)

- [ ] Widen to 8 columns × 4 rows = 32 PEs = 32 DSPs
- [ ] Close timing at 100 MHz

**Done when:** timing closes and the testbench still passes.

> [!info] Why 8×4 and not 8×8
> 64 PEs needs 64 DSPs and you have 60 — and the vector unit in [[phase-5-vector-unit]] needs a dozen.
>
> **32 MACs @ 100 MHz = 6.4 GOPS**, roughly 5× what the A9 will do on int8. A real win, and it fits. See [[board-specs]].

---

## 3.4 Two int8 MACs per DSP

> [!warning] Stretch goal — budget for this not working
> Pack two weights into the 25-bit A port as `w0 + (w1 << 18)`, multiply by a shared activation, extract two products from the 48-bit result.
>
> **Done when:** you get 64 PEs from 32 DSPs.
>
> **The honest warning:** the well-known Xilinx app note (WP486) targets **DSP48E2** on UltraScale, which has a 27×18 multiplier. Your 7-series **DSP48E1** is 25×18 — the packing is tighter and you may not get clean guard bits for signed int8. Plan to fall back to 1 MAC/DSP. This is a stretch goal, **not a dependency**.
