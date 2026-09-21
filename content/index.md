---
title: Conrad Fernandez
---

Computer engineering, GPUs, and digital logic. I work at AMD on graphics verification.

This site is written in [Obsidian](https://obsidian.md) and published with
[Quartz](https://quartz.jzhao.xyz/). Wikilinks, callouts, and backlinks all work — write
a note, push it, and it appears here.

## Writing

Notes go in `content/`. The filename becomes the URL, so `content/npu-notes.md` is
served at `/npu-notes`. This page is `content/index.md`.

> [!tip] Local preview
> Run `npx quartz build --serve` and open `localhost:8080`. The site rebuilds as you
> save, so you can keep Obsidian and the browser side by side.

## Notes

- [[npu-course/index|Zero to NPU: Running a Tiny LLM on a $149 FPGA]] — a full course on
  designing an int8 neural processing unit for a Zynq-7007S, from a rusty Verilog refresher
  to a transformer generating text on-chip. Eight phases, ~34 weeks.

## Projects

- [NPU systolic array animation](https://conboy.github.io/npu-anim/) — interactive,
  cycle-accurate visualization of the matrix engine inside a neural processing unit
- [risc-cpu](https://github.com/conboy/risc-cpu) — a RISC CPU in Verilog
- [fpga-projects](https://github.com/conboy/fpga-projects) — digital logic on the
  Blackboard FPGA

## Elsewhere

[GitHub](https://github.com/conboy)
