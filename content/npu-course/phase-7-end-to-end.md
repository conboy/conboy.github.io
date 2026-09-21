---
title: "Phase 7 — End to End"
description: "KV cache budgeting, a fully on-chip language model, and the DDR-streamed model that hits the memory wall."
tags:
  - npu
  - hardware
  - capstone
---

# Phase 7 — End to end

**5 weeks** · Part of [[index|Zero to NPU]] · Prev: [[phase-6-sequencer]] · Next: [[phase-8-optimization]]

Both capstones live here.

---

## 7.1 KV cache — and the on-chip budget that decides your context length

Do this arithmetic carefully, because it sets a parameter you can't change later.

`stories260K` is `dim=64`, `hidden=172`, 5 layers, 8 heads / 4 KV heads, `vocab=512` → `head_size = 8`, **`kv_dim = 32`**.

Weights: 5 × (12,288 attn + 33,024 FFN) = 227K, plus 32,768 embedding ≈ **260K params** = **130 KB at 4-bit** ≈ 29 BRAM blocks.

KV cache = `seq_len × n_layers × 2 × kv_dim` bytes:

| Context | KV size | BRAM blocks | Weights + KV | Left for buffers |
|---|---|---|---|---|
| 512 | 164 KB | 37 | 66 / 50 | **doesn't fit** |
| 256 | 82 KB | 18 | 47 / 50 | 3 — **too tight** |
| **128** | **41 KB** | **9** | **38 / 50** | **12 — works** ✅ |

> [!important] Decision: cap context at 128 tokens
> ~100 words of story, which is plenty for this model, and it leaves 12 BRAM blocks for weight/activation double-buffering and the sequencer's descriptor queue.
>
> Document the tradeoff. **"Why 128?"** has a precise arithmetic answer, and being able to give it is what separates a design from a guess.

- [ ] Implement KV storage and addressing
- [ ] Verify against the golden model across a full 128-token generation

---

## 7.2 Tier 1 capstone — fully on-chip

- [ ] `stories260K` with **4-bit weights** (130 KB) resident in BRAM
- [ ] Zero DDR traffic during generation

**Done when:** coherent English stories stream out the UART at high speed.

> [!success] This is your headline result
> A complete language model — weights and all — living inside the FPGA fabric with the DDR bus idle.

---

## 7.3 Tier 2 capstone — the memory wall

- [ ] `stories15M`, int8, streamed from DDR every token

**Done when:** it generates text, and your measured tok/s lands within ~2× of the bandwidth ceiling you computed in [[phase-4-memory|§4.4]].

> [!warning] This is your headline lesson
> Same NPU, 60× the model, and throughput is now set **entirely** by a 16-bit DDR3 bus.
>
> ==No amount of added DSPs would help.== That sentence is the whole reason NPU design is a memory problem wearing a compute problem's clothes — and now you'll have measured it yourself rather than read it.
