---
title: Inside a systolic array
description: "A matrix-engine visualization checked against a cycle-accurate register model and reference matrix multiply."
---

<p class="eyebrow">ACCELERATOR ARCHITECTURE / INTERACTIVE DEMO</p>

A weight-stationary systolic array is easier to understand when you can follow one activation and its partial sum through the hardware. This project makes that movement visible, cycle by cycle.

<p><a class="button primary" href="/npu-anim/" data-router-ignore data-no-popover>Run the animation ↗</a> <a class="button" href="https://github.com/conboy/npu-anim">Source code ↗</a></p>

![Activations move right through a grid of stationary weights; partial sums move down.](/assets/systolic.svg)

## What I built

A self-contained JavaScript and Canvas visualization of an int8 matrix engine. It exposes array dimensions, playback speed, single-cycle stepping, randomized inputs, and a comparison with a reference matrix multiply.

The scope is the matrix engine: a full NPU also needs memory movement, scheduling, and vector operations. Those are explored in the [NPU learning roadmap](/npu-course/).

## The design constraint

The animation must communicate the timing correctly. Activations are skewed across rows, partial sums move between registered stages, and useful utilization changes during fill and drain. A visually plausible animation can still show an impossible dataflow.

## How it is checked

The repository includes `verify.js`, which compares a cycle-accurate register model with a reference matrix multiplication. It also checks the renderer's closed-form timing against that model: active cells, activation positions, partial-sum values, and output arrival cycles.

```sh
git clone https://github.com/conboy/npu-anim.git
cd npu-anim
node verify.js
```

The suite covers seven array configurations. The README records two bugs caught during development: starting at cycle 1 dropped the first activation, and too few tokens prevented the array from reaching full utilization.

## What to inspect

1. Pause and step through the first activation entering the array.
2. Follow the partial sum down a column.
3. Compare fill and drain behavior across array sizes.
4. Generate new data and check the reference comparison.

The displayed utilization describes this matrix-engine model. It is not a measurement of FPGA timing, power, DDR bandwidth, or end-to-end model throughput.

[Verification source](https://github.com/conboy/npu-anim/blob/main/verify.js) · [Project README](https://github.com/conboy/npu-anim#readme) · [Back to portfolio](/)
