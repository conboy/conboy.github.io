---
title: Conrad Fernandez
description: "Computer engineer working on GPU verification at AMD. I build accelerators, processors, and the occasional radio network."
---

I'm a computer engineer at **AMD**, where I work on GPU verification. Before that I
studied computer engineering at **Queen's University**.

Most of what I enjoy sits at the boundary where software stops being an abstraction
and starts being a circuit — processors, accelerators, and the memory systems that
decide how fast either one actually goes.

This site is my notebook. It's written in Obsidian and published as a digital garden,
so pages here grow and get revised rather than being frozen the day they're posted.

---

## Zero to NPU

> [!abstract] [Running a tiny LLM on a $149 FPGA](/npu-course/)
> A complete build-it-yourself course: design an **int8 neural processing unit** in
> programmable logic on a Zynq XC7Z007S — 14,400 LUTs, 60 DSP slices, 225 KB of block
> RAM — and run a transformer end to end, printing generated stories out the UART.
>
> Nine phases, from toolchain setup through the MAC engine, memory hierarchy, vector
> unit, and sequencer, to two capstones. Every module ends in a **measurement**, not a
> feeling.
>
> The insight everything else follows from: *token generation is memory-bandwidth-bound,
> not compute-bound.*

If you want the idea in thirty seconds instead of nine phases, the
[**systolic array animation**](https://conboy.dev/npu-anim/) shows the matrix engine
running cycle by cycle — weights stationary, activations streaming, partial sums
falling out the bottom as finished dot products.

---

## Things I've built

| | |
|---|---|
| [**NPU animation**](https://conboy.dev/npu-anim/) | Interactive, cycle-accurate visualization of a weight-stationary systolic array. The timing is verified against a reference matmul, not eyeballed. |
| [**LoRa Wildfire Detection**](https://github.com/conboy/LoRa-Wildfire-Detection-System) | Distributed ESP32 sensor network over long-range, low-power radio. Temperature and humidity nodes reporting across kilometres without cell coverage. |
| [**RV32I**](https://github.com/conboy/RV32I) | A 32-bit integer RISC-V processor, built from the ISA spec up. |
| [**risc-cpu**](https://github.com/conboy/risc-cpu) | An earlier RISC CPU in Verilog — where the processor habit started. |
| [**FPGA projects**](https://github.com/conboy/fpga-projects) | Digital logic work on the Real Digital Blackboard, the same board the NPU course targets. |

---

## Elsewhere

[GitHub](https://github.com/conboy) · [Email](mailto:conjamalex@gmail.com)
