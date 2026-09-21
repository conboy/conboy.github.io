---
title: "Phase 0 — Toolchain and Verilog Refresh"
description: "Getting Vivado, the UART, a first AXI-Lite peripheral, and the ILA working on a Zynq-7007S."
tags:
  - npu
  - hardware
---

# Phase 0 — Recommission the toolchain and your Verilog

**2 weeks** · Part of [[index|Zero to NPU]] · Next: [[phase-1-baseline-llm]]

You did a CPU in Verilog once. This phase is about getting the muscle back and proving the board talks to you.

---

## 0.1 Toolchain up

- [ ] Install Vivado (WebPACK is free and supports 7007S) + Vitis
- [ ] Create a block design with the Zynq PS
- [ ] Run "Hello World" out the UART at 115200

**Done when:** text appears in your terminal.

> [!bug] Gotcha
> The Blackboard needs its **board definition files** installed, or you'll hand-configure the DDR3 controller and MIO pinout. Get them from Real Digital *first*.

---

## 0.2 Verilog limbering

- [ ] Build a pipelined 16×16→32 MAC unit — registered inputs, registered output, `valid`/`ready` handshake
- [ ] Write a Verilator testbench against a C reference

**Done when:** 100,000 random vectors run with zero mismatches.

> [!bug] Gotcha
> Get the handshake right *now*. AXI-Stream semantics — **`valid` cannot depend combinationally on `ready`** — will bite you for the next 8 months otherwise.

---

## 0.3 Your first AXI-Lite peripheral

The most important plumbing you'll learn.

- [ ] Build a 16-register AXI-Lite slave
- [ ] PS writes registers → PL drives the LEDs
- [ ] PL writes a counter into a register → PS reads it back

**Done when:** `Xil_Out32(BASE+0x04, 0xDEADBEEF)` from the A9 changes hardware state, and `Xil_In32` reads back a value the PL produced.

> [!bug] Gotcha
> AXI-Lite addresses are **byte** addresses; your register index is `addr[5:2]`. Off-by-four here costs people entire evenings.

---

## 0.4 Visibility

- [ ] Drop an ILA (Integrated Logic Analyzer) onto your AXI-Lite bus
- [ ] Trigger on a write to a specific address

**Done when:** you can see the AWVALID/WVALID handshake in the waveform viewer.

> [!tip] Why this matters
> When the NPU hangs at 3 a.m. in month 6, ILA is how you find out which side dropped the handshake.
