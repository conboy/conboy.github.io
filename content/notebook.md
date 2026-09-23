---
title: Technical notebook
description: "Notes, experiments, and learning roadmaps in processor and accelerator design."
---

Working notes. Some describe a design I have not built yet, others an experiment in progress. Where something is finished, the artifact or result is linked.

## Accelerator architecture

- [Zero to NPU](/npu-course/): a nine-phase learning roadmap for an int8 accelerator on a Zynq FPGA.
- [Inside a systolic array](/projects/npu-animation): an interactive matrix-engine visualization and its verification approach.
- <a href="/npu-anim/" data-router-ignore data-no-popover>Run the systolic array animation ↗</a>

## Local AI infrastructure

- [Two Pools](/local-ai-course/): a fourteen-module roadmap for running a 30B mixture-of-experts coding model on a 16 GB GPU, built around the bandwidth gap between VRAM and system memory.

## Digital design

- [RISC CPU](/projects/risc-cpu): Verilog datapath and instruction-level simulation artifacts.
- [FPGA exercises](https://github.com/conboy/fpga-projects): digital logic exercises targeting the Real Digital Blackboard.
- [RV32I design plan](https://github.com/conboy/RV32I): a planning document for a RISC-V processor. The public repository does not yet include an RTL implementation.

[Back to selected work](/)
