---
title: "Phase 1 — A Baseline LLM on the ARM Core"
description: "Porting llama2.c to bare-metal ARM, profiling it, quantizing to int8, and freezing the golden model."
tags:
  - npu
  - hardware
---

# Phase 1 — Build the thing you're going to accelerate

**3 weeks** · Part of [[index|Zero to NPU]] · Prev: [[phase-0-toolchain]] · Next: [[phase-2-numerics]]

You cannot accelerate what you haven't profiled. This phase produces your baseline *and* your golden model.

---

## 1.1 Get a real LLM running on the ARM core

- [ ] Port **`llama2.c`** (Karpathy — a single ~700-line C file, no dependencies) to bare-metal on the A9
- [ ] Load `stories260K.bin` and the tokenizer from the SD card
- [ ] Generate text

**Done when:** the UART prints a coherent short story.

> [!success] This is your first real milestone
> A language model running on your board, before you've written a line of accelerator RTL.

> [!info] Why this model
> `dim=64`, 5 layers, 8 heads (4 KV heads), `vocab=512`, ~260K params. Small enough to fit your board, large enough to produce real English. It is the single best target that exists for this project.

---

## 1.2 Profile it

- [ ] Wrap each operation with the A9's global timer
- [ ] Produce a breakdown: matmul vs. softmax vs. RMSNorm vs. RoPE vs. sampling

**Done when:** you have a table of percentages.

> [!warning] Expect matmul at 85–95%
> Everything else is noise — *until you accelerate the matmul*, at which point it becomes your bottleneck. Remember this number. It's Amdahl's Law and it will ambush you in [[phase-5-vector-unit]].

---

## 1.3 Quantize to int8, in C

- [ ] Per-channel symmetric int8 for weights, per-tensor for activations
- [ ] int32 accumulators
- [ ] Requantize back to int8 between operations

**Done when:** the int8 model still produces coherent stories, and you've logged the perplexity delta vs. fp32 (expect **<1%**).

> [!bug] Gotcha
> Use **symmetric** (zero-point = 0) quantization. Asymmetric adds a cross-term to every matmul and roughly doubles your hardware complexity for accuracy you don't need here.

---

## 1.4 Freeze the golden model

- [ ] Refactor so every quantized op is a standalone C function
- [ ] Each can dump its inputs and outputs to a file

**Done when:** you can generate a test vector file for any single operation.

> [!important] Why this matters
> Every RTL module from here on is validated against these dumps.
>
> This is standard design-verification practice: you don't check hardware by looking at it, you check it against a reference implementation you trust. Hobby FPGA projects usually skip this step, then spend weeks debugging a system where *any* of a dozen blocks could be the culprit. Build the oracle first.
