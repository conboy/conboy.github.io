---
title: "Board Specs — Zynq XC7Z007S"
description: "The 14,400 LUTs, 60 DSPs and 225 KB of BRAM that force every design decision in this course."
tags:
  - npu
  - fpga
  - reference
---

# Board Specs — Know your silicon before you design for it

Part of [[index|Zero to NPU]].

The Blackboard carries a **Xilinx/AMD Zynq XC7Z007S-1CLG400C**. This is the *smallest* Zynq-7000 part.

> [!danger] Do not trust Pynq-Z2 tutorials
> Almost every FPGA-ML tutorial online assumes a **XC7Z020** (53K LUTs, 220 DSPs, 630 KB BRAM). You have roughly **one quarter** of that. Designs that fit a Pynq-Z2 will not fit here. Check every resource claim against the table below.

---

## The numbers that constrain you

| Resource | Amount | What it means for you |
|---|---|---|
| LUTs | **14,400** | Very tight. A stock Xilinx AXI DMA IP eats ~2–3K of these. You will write your own. |
| Flip-flops | 28,800 | Fine. Pipeline aggressively; FFs are the cheap resource. |
| DSP48E1 slices | **60** (datasheet says 66; plan for 60) | 1 int8 MAC each. Caps compute at ~32–64 MACs. |
| Block RAM | 50 × 36 Kb = **225 KB** | Your entire on-chip working set. A 260K-param int8 model *barely* doesn't fit. |
| PS | 1× Cortex-A9 @ 666 MHz | **Single core.** No SMP. NEON is available and matters. |
| DDR3 | 512 MB, 16-bit @ 533 MHz | Peak ~2.1 GB/s, realistic **~1.2 GB/s**. This is your real bottleneck. |
| PS↔PL | 4× AXI-HP (64-bit), 2× AXI-GP, 1× ACP | HP for bulk weight streaming, GP for control, ACP for cache-coherent access. |

Speed grade is **-1** (slowest). Plan PL designs at ==100 MHz==, push to 150 MHz only after you've closed timing once.

---

## What fits, and what doesn't

| Model | int8 size | Throughput ceiling | Where weights live |
|---|---|---|---|
| `stories260K` | 260 KB | *thousands* of tok/s | On-chip BRAM (at 4-bit) — **no DDR traffic** |
| `stories15M` | 15 MB | ~80 tok/s | Streamed from DDR every token |
| TinyLlama 1.1B | 1.1 GB | ~1 tok/s | Not happening on this board |

This is why the capstone is two-tier — see [[phase-7-end-to-end]].

---

## Resource budget

The design does close. I worked this out against the real part:

| Block | LUTs | DSPs | BRAM |
|---|---|---|---|
| 8×4 systolic array (32 PEs) | ~1,500 | 32 | — |
| Weight + activation buffers | ~800 | — | 12 |
| Vector unit (norm/softmax/act/RoPE) | ~2,000 | 12 | 4 |
| Custom AXI4 burst master | ~700 | — | 2 |
| AXI-Lite register slave | ~300 | — | — |
| Command sequencer | ~1,000 | — | 2 |
| Xilinx AXI interconnect | ~1,200 | — | — |
| KV cache (128-token context) | — | — | 9 |
| **Total** | **~7,500 / 14,400** | **44 / 60** | **31 / 50** |

> [!check] ~48% LUT utilization
> That leaves real room for timing closure and debug logic. Good.

The remaining **19 BRAM blocks** hold model weights for the Tier-1 capstone. 4-bit `stories260K` needs 29, so you'll spill the embedding table to DDR and keep the 5 transformer layers on-chip. See [[phase-7-end-to-end|§7.1]] for the full arithmetic — that budget is what forces the 128-token context limit.

---

## Why 8×4 and not 8×8

64 PEs needs 64 DSPs and you have 60 — and the vector unit needs a dozen.

**32 MACs @ 100 MHz = 6.4 GOPS**, roughly 5× what the A9 will do on int8. That's a real win and it fits.
