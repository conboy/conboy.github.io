---
title: "Phase 8 — Optimization and Write-Up"
description: "4-bit weights, timing closure, power measurement, and publishing the roofline."
tags:
  - npu
  - hardware
---

# Phase 8 — Make it good

**4 weeks** · Part of [[index|Zero to NPU]] · Prev: [[phase-7-end-to-end]]

---

## 8.1 4-bit weights everywhere

- [ ] Unpack in the **weight-load path**, not in the PEs

> [!tip] Why not in the PEs
> Unpacking per-PE replicates the shift/mask logic 32×. Doing it once in the load path costs a few dozen LUTs instead of a few hundred.

---

## 8.2 Timing closure

- [ ] Push 100 → 150 MHz
- [ ] Learn `report_timing_summary`
- [ ] Add pipeline stages on the critical path
- [ ] Try a Pblock floorplan constraint

> [!info] Speed grade -1
> You have the slowest silicon in the family. 150 MHz is an ambitious but reachable target; don't be discouraged if the array tops out around 125.

---

## 8.3 Power — the actual argument for NPUs

- [ ] Measure at the wall
- [ ] Compare **tok/s/W** against a CPU and a GPU running the same model

> [!success] Your number will be very good
> This is the real argument for dedicated inference hardware, and now you'll have measured it yourself instead of citing someone else's benchmark.

---

## 8.4 Write it up

- [ ] Roofline plot (from [[phase-4-memory|§4.4]])
- [ ] Resource utilization table (vs. the budget in [[board-specs]])
- [ ] Quantization accuracy delta (from [[phase-1-baseline-llm|§1.3]])
- [ ] tok/s/W comparison
- [ ] The two capstone results side by side

> [!important] Publish the measurements, not just the design
> "An int8 NPU that runs a transformer language model on a $149 FPGA with 14,400 LUTs, plus the roofline analysis showing exactly why the larger model is slow" is a complete, self-contained result.
>
> The numbers are what make it worth reading. Plenty of people have built a systolic array. Far fewer have published the bandwidth measurements explaining what their array was actually limited by.
