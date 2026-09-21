---
title: "Phase 4 — Feeding the Beast"
description: "Tiling, double-buffering, a hand-written AXI4 burst master, and measuring your actual DDR roofline."
tags:
  - npu
  - hardware
---

# Phase 4 — Feeding the beast

**5 weeks** · Part of [[index|Zero to NPU]] · Prev: [[phase-3-mac-engine]] · Next: [[phase-5-vector-unit]]

A perfect MAC array that's starved of data is a perfect waste of DSPs.

---

## 4.1 Tiling

- [ ] Work out **on paper** how a 64×64 matmul decomposes onto an 8×4 array — how many tiles? what's the loop nest?
- [ ] Implement the address generators

**Done when:** your array computes a 64×64 matmul correctly from BRAM.

---

## 4.2 Double buffering

- [ ] Ping-pong weight buffers — load tile N+1 while computing tile N

**Done when:** measured PE utilization goes above **80%** (count busy cycles in hardware with a counter you read over AXI-Lite).

---

## 4.3 Write your own AXI4 burst master

- [ ] A minimal AXI4 master issuing read bursts from DDR through an HP port into BRAM
- [ ] Support INCR bursts up to 256 beats

**Done when:** you can DMA 1 MB from DDR into BRAM and the contents match.

> [!important] Why not Xilinx AXI DMA IP
> It's ~2–3K LUTs — **20% of your entire fabric** — and 90% of it is features you don't need. Yours will be ~700 LUTs. On a 7007S this is not optional.

> [!danger] Gotcha #1 — cache coherency
> The A9 writes weights to DDR, but they're sitting in L1/L2 cache. Your PL reads **stale DDR**.
>
> Fix: `Xil_DCacheFlushRange()` before kicking the PL, `Xil_DCacheInvalidateRange()` after it finishes. Or mark the buffer non-cacheable via the MMU.
>
> ==This bug looks exactly like an RTL bug and it is not one.== Budget a day for it anyway.

> [!bug] Gotcha #2 — the 4 KB rule
> AXI bursts **cannot cross 4 KB address boundaries**. Your address generator must split them. The interconnect will not do this for you.

---

## 4.4 Measure your roofline

- [ ] Pure-read benchmark — DMA a large buffer, count cycles, compute GB/s

**Done when:** you know your actual sustained DDR bandwidth (expect **0.8–1.5 GB/s**).

> [!tip] This is the centerpiece of your write-up
> Plot compute intensity vs. achieved GOPS. This number tells you exactly which model sizes are feasible — and it's what makes [[phase-7-end-to-end|the Tier 2 capstone]] interpretable instead of just slow.
