---
title: "Phase 6 — The Command Sequencer"
description: "A descriptor ring buffer and doorbell, so the accelerator runs a whole transformer layer without host involvement."
tags:
  - npu
  - hardware
---

# Phase 6 — The sequencer

**4 weeks** · Part of [[index|Zero to NPU]] · Prev: [[phase-5-vector-unit]] · Next: [[phase-7-end-to-end]]

Right now the PS pokes registers for every operation, and interrupt latency plus AXI-Lite round-trips dominate your runtime. Time to let the PL run itself.

---

## 6.1 A command ring

Descriptor format in DDR:

```c
struct descriptor {
    uint32_t opcode;
    uint32_t src_addr;
    uint32_t dst_addr;
    uint32_t M, N, K;
    uint32_t requant_M0;
    uint32_t requant_shift;
};
```

- [ ] Define the descriptor format
- [ ] PS builds a chain of descriptors for an entire transformer layer
- [ ] PS writes a doorbell register
- [ ] PL executes the whole chain without further PS involvement
- [ ] Interrupt on completion

**Done when:** one doorbell write executes a full layer and the PS sleeps through it.

---

> [!quote] This pattern is how real GPUs work
> What you've built is a **command ring buffer**, and it's the same structure every modern GPU uses to accept work from the driver:
>
> | Your NPU | A GPU command processor |
> |---|---|
> | Descriptor | Command packet |
> | Doorbell register | Write-pointer doorbell |
> | Sequencer FSM | Command processor front-end |
> | Descriptor chain | Indirect buffer |
>
> The driver writes packets into a ring in memory, bumps a write pointer through a doorbell aperture, and dedicated hardware fetches and executes them without further CPU involvement — exactly what you just built, minus about four orders of magnitude of complexity.
>
> If you want to see a production version, the open-source [`amdgpu`](https://gitlab.freedesktop.org/agd5f/linux) and [`Mesa`](https://gitlab.freedesktop.org/mesa/mesa) trees implement ring setup, doorbells, and indirect buffers in the open. Building a small one from scratch first makes that code *much* easier to read.

> [!bug] Gotcha
> Make the descriptor size a **power of two** and align it. Ring wraparound logic with non-power-of-two entries is a bug farm.
