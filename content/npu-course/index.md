---
title: "Zero to NPU: Running a Tiny LLM on a $149 FPGA"
description: "A build-it-yourself course for designing an int8 neural processing unit on a Zynq-7007S and running a transformer language model on it end to end."
tags:
  - fpga
  - npu
  - hardware
  - machine-learning
---

Can you run a language model on the smallest Zynq FPGA you can buy?

The board in question is a [Real Digital Blackboard](https://www.realdigital.org/hardware/blackboard) — $149, built around a **Zynq XC7Z007S**: 14,400 LUTs, 60 DSP slices, 225 KB of block RAM. That's roughly a quarter of the Pynq-Z2 that most FPGA machine-learning tutorials assume. It is a genuinely small part.

The answer is yes, and this is the course that gets you there: a custom **int8 neural processing unit** in the programmable logic, running a transformer end to end and printing generated stories out the UART.

> [!abstract] The one insight that determines everything
> **Token generation is memory-bandwidth-bound, not compute-bound.** Generating a single token requires reading every weight exactly once. Every architectural decision here follows from that.
>
> $$\text{tokens/sec}_{\text{ceiling}} \approx \frac{\text{DDR bandwidth}}{\text{model size}} \approx \frac{1.2\ \text{GB/s}}{\text{model size}}$$

This is why the course ends with **two** capstones rather than one. The first shows off the compute. The second shows you the wall.

---

## Who this is for

You've written some Verilog — a class project, a toy CPU, something — and you want to understand accelerator design by building one rather than reading about one. You do not need prior machine-learning hardware experience.

Every module ends in a **measurement**, not a feeling. "It works" means a self-checking testbench passed or a number came out of the UART.

---

## Start here

Read [[board-specs]] first. Every decision in this course is forced by that silicon, and the numbers are *not* the ones in most tutorials.

Then begin at [[phase-0-toolchain]].

> [!tip] There's a working language model by week 4
> [[phase-1-baseline-llm]] gets a real LLM generating stories on the board **before you write a single line of accelerator RTL** — running on the ARM core, slowly. It doubles as the golden reference that validates every hardware block after it.

---

## The phases

| # | Phase | Weeks | What you build |
|---|---|---|---|
| 0 | [[phase-0-toolchain]] | 2 | Vivado, UART, AXI-Lite, ILA |
| 1 | [[phase-1-baseline-llm]] | 3 | `llama2.c` bare-metal + int8 golden model |
| 2 | [[phase-2-numerics]] | 2 | The requantization pipeline |
| 3 | [[phase-3-mac-engine]] | 5 | 8×4 int8 systolic array |
| 4 | [[phase-4-memory]] | 5 | Tiling, double-buffering, custom AXI4 DMA |
| 5 | [[phase-5-vector-unit]] | 4 | RMSNorm, softmax, SiLU, RoPE |
| 6 | [[phase-6-sequencer]] | 4 | A command ring buffer |
| 7 | [[phase-7-end-to-end]] | 5 | KV cache + both capstones |
| 8 | [[phase-8-optimization]] | 4 | 4-bit weights, timing, power, write-up |

**Total: ~34 weeks at 6–10 hrs/week**, or about eight months. That's an honest estimate for someone rusty on RTL — the first working matmul lands around week 12.

See [[architecture.canvas|the architecture diagram]] for how the blocks connect, and [[reference-material]] for sources.

---

## Ground rules

> [!important] Four rules that save months
> 1. **Every module ends in a measurement.** A self-checking testbench passed, or a number came out of the UART. Nothing else counts.
> 2. **Simulate before you synthesize.** Use [Verilator](https://www.veripool.org/verilator/), not Vivado `xsim` — Verilator links your C golden model directly into the C++ testbench for bit-exact per-cycle comparison, and it's 10–100× faster. Iteration speed is the whole game.
> 3. **The C golden model is law.** Written in [[phase-1-baseline-llm]], then never changed. Every RTL block validates against it.
> 4. **Bare-metal, not PetaLinux.** You want direct physical addresses and trivial cache reasoning. Linux on a single-core A9 with 512 MB is a month of yak-shaving that teaches you nothing about accelerators.

---

## The two capstones

> [!success] Tier 1 — fully on-chip
> `stories260K` at 4-bit, weights living **entirely in block RAM**. Coherent English stories out the UART, with the DDR bus completely idle.
>
> A whole language model — weights and all — inside the FPGA fabric. See [[phase-7-end-to-end]].

> [!warning] Tier 2 — the memory wall
> `stories15M` at int8, streamed from DDR every token. Same NPU, 60× the model, and throughput is now set **entirely** by a 16-bit DDR3 bus.
>
> No amount of added DSPs would help. This is the more valuable of the two results, and measuring it yourself lands differently than reading it.

---

## Why bother, when a laptop is faster

It isn't about beating a CPU on wall-clock. It's that after this you will know, concretely rather than abstractly:

- why inference accelerators are **memory systems** with multipliers bolted on
- what a systolic array actually does, cycle by cycle
- why quantization is a hardware decision before it's an accuracy decision
- how a command ring buffer decouples a host from an accelerator
- how to read a roofline plot of your own silicon

That transfers to every accelerator you'll ever touch, including ones far larger than this one.

---

## Escape hatch

If you hit a wall on capacity — and 14,400 LUTs is genuinely restrictive — a **Pynq-Z2** (XC7Z020: 53K LUTs, 220 DSPs, 630 KB BRAM) runs about $150 and gives 3–4× the headroom. Everything here ports over unchanged.

But don't start there. Constraint is the best teacher in hardware, and fitting an LLM onto a 7007S is a materially more interesting thing to have done.
