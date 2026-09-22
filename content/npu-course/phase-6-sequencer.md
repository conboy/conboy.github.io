---
title: "Phase 6 — The Command Sequencer"
description: "A descriptor ring buffer and doorbell, so the accelerator runs a whole transformer layer without host involvement."
tags:
  - npu
  - hardware
---

# Phase 6 — The sequencer

**4 weeks** · Part of [[index|Zero to NPU]] · Prev: [[phase-5-vector-unit]] · Next: [[phase-7-end-to-end]]

> [!info] What this phase is for
> Right now, every single operation in a transformer layer requires the ARM core (the PS) to reach across the PS↔PL boundary, write a handful of [[concepts#AXI-Lite|AXI-Lite]] registers, and then sit there waiting for that one operation to finish before it can even think about starting the next one. A single transformer layer in this model is dozens of these operations — matmuls, RMSNorm, softmax, SiLU, RoPE — and you have 5 layers, run fresh for every token you generate. If the PS has to babysit each operation individually, your accelerator spends most of its life idle, waiting on round-trip latency instead of computing anything. This phase fixes that by building a small hardware [[concepts#FSM|FSM]] — the "sequencer" — that can read a whole to-do list for an entire layer out of memory and execute it start to finish with no PS involvement at all, raising one [[concepts#Interrupt|interrupt]] only when the whole list is done. You're making the accelerator self-driving.

---

## The problem: who is in charge?

Think about what actually has to happen for the PS to run one matmul on your PL accelerator, using the AXI-Lite register file you built back in [[phase-0-toolchain|Phase 0]]. The PS writes the source address register, the destination address register, and the M, N, K dimension registers — that's already 5 separate AXI-Lite writes. Then it writes `requant_M0` and `requant_shift` so the output gets requantized correctly (2 more writes). Then it writes a "go" bit to a control register (1 more write). Then it either polls a status register in a tight loop, burning ARM cycles doing nothing useful, or it sleeps and waits for an interrupt — and even the interrupt path costs you an interrupt controller round-trip and a context switch. Then, once the operation is done, the PS reads a status register to confirm it actually completed before it's safe to touch the result. That's roughly 8 register writes plus a wait plus a status read, for **one** operation.

Each of those AXI-Lite transactions isn't free. A single register write or read over AXI-Lite costs somewhere on the order of 10–20 bus cycles of latency, and on top of that there's real software overhead: the C code has to compute the register address, do the memory-mapped write, and — if you're waiting for completion — go through the ARM's interrupt or polling path, which is dozens to hundreds of cycles by itself depending on how it's implemented. None of that time does any useful math. It's pure coordination overhead.

Now multiply. A single transformer layer in this model has a matmul for Q, a matmul for K, a matmul for V, RoPE on Q and K, an attention-score matmul, softmax, a weighted-sum matmul, an RMSNorm before all of that, another RMSNorm before the feed-forward block, two more matmuls for the FFN's gate and up projections, a SiLU, an elementwise multiply, and a final down-projection matmul. Call it roughly a dozen matmuls and a handful of vector-unit operations per layer — and this model has **5 layers**, and you run this entire sequence again for **every single token** the model generates. If each operation costs even a modest few hundred cycles of pure PS-to-PL coordination overhead on top of its actual compute time, and you have on the order of 15-20 operations per layer times 5 layers times every token you generate, that overhead adds up to a real, measurable fraction of your total inference time — and unlike the compute itself, it's overhead you get zero benefit from. You built a fast matmul array in [[phase-3-mac-engine|Phase 3]] and a fast vector unit in [[phase-5-vector-unit|Phase 5]]; if the PS has to hold both units' hands through every individual step, you're leaving exactly the kind of performance on the table that Phases 3 and 5 were supposed to win back.

The fix is the same idea that shows up everywhere in real hardware whenever software and hardware need to hand off a lot of work without a lot of chatter: instead of the PS issuing commands one at a time and waiting after each one, have the PS **batch up an entire list of commands in memory** — one whole transformer layer's worth — and then tell the hardware "here's a list, go execute all of it, tell me only when you're completely done." The PS writes that list once, taps the hardware once, and then it's free to do something else (like start preparing the *next* token) while the PL works through the whole layer unattended. That's the entire idea behind this phase, and everything below is really just working out the mechanical details of "how do you hand a hardware unit a to-do list."

---

## What a descriptor is

A **[[concepts#Descriptor|descriptor]]** is nothing exotic — it's just a plain struct sitting in DDR that describes one unit of work, the same way a single line in a shopping list describes one item to buy. Instead of the PS pushing 8 individual register writes to kick off one matmul, it writes one descriptor struct to memory, and the hardware reads that whole struct back over the AXI bus in one burst instead of 8 separate slow AXI-Lite transactions.

Here's the descriptor format for this NPU, using the fields you were given back at the start of this phase:

```c
struct descriptor {
    uint32_t opcode;          // what operation to perform
    uint32_t src_addr;        // where to read input data from (DDR address)
    uint32_t dst_addr;        // where to write output data to (DDR address)
    uint32_t M, N, K;         // matrix dimensions this op operates on
    uint32_t requant_M0;      // requantization multiplier (Phase 2)
    uint32_t requant_shift;   // requantization shift amount (Phase 2)
};
```

Walk through why each field earns its place. `opcode` tells the sequencer *which* hardware unit should handle this descriptor and in what mode — is this a matmul that should go to the PE array from [[phase-3-mac-engine|Phase 3]], or an RMSNorm/softmax/SiLU/RoPE operation that should go to the shared vector unit from [[phase-5-vector-unit|Phase 5]]? `src_addr` and `dst_addr` are exactly what they sound like — DDR addresses, because your operands and results live in DRAM, not registers, once you're chaining more than one operation together. `M`, `N`, `K` are the matrix dimensions the operation needs (for a vector-unit op that isn't really a matmul, these get reused — e.g. `M` might just mean "vector length"). `requant_M0` and `requant_shift` are the fixed-point requantization parameters from [[phase-2-numerics|Phase 2]] — every operation in this pipeline produces a wider intermediate result that has to be requantized back down to int8, and that requantization needs per-operation parameters, so they travel with the descriptor instead of being hardcoded in hardware.

Define the opcode as an enum, one value per operation this NPU knows how to do:

```c
enum opcode {
    OP_MATMUL   = 0,
    OP_RMSNORM  = 1,
    OP_SOFTMAX  = 2,
    OP_SILU     = 3,
    OP_ROPE     = 4,
    OP_REQUANT  = 5,
    // add more as you need them
};
```

Here's the part worth sitting with: a **chain** of these descriptors, one after another in memory, is a *program* for your accelerator. Each descriptor is one instruction; the opcode field is the instruction's operation code, exactly like a real CPU's instruction encoding; the chain as a whole is a short program that says "do this matmul, then this RMSNorm, then this matmul, then this RoPE..." — the entire sequence of operations for one transformer layer, laid out once in DDR before the hardware ever looks at it. You have, without necessarily meaning to, invented a tiny special-purpose instruction set. That reframing matters for what comes next: once you think of a descriptor chain as a program, the sequencer FSM you'll build in 6.3 is really just an instruction fetch-decode-execute loop, the same shape as any CPU you've ever studied, just with "opcode" mapped onto "route this to the matmul array or the vector unit" instead of "add these two registers."

---

## Ring buffers and doorbells

This is the actual mechanism that makes the whole scheme work, so take it slowly.

A **[[concepts#Ring buffer|ring buffer]]** is a fixed-size array in memory that you treat as if it wraps around in a circle — when you reach the last slot, the next write goes back to slot 0. Two pointers matter: a **read pointer** (sometimes called the tail), which the *consumer* — your sequencer hardware — owns and advances as it finishes each descriptor, and a **write pointer** (the head), which the *producer* — your PS software — owns and advances as it adds new descriptors. Each side only ever touches its own pointer; that separation is what lets software keep filling the ring while hardware is still draining it, without either side stepping on the other's bookkeeping.

```
                     write_ptr (owned by PS,
                      advances as software adds work)
                            │
                            ▼
   ┌───┬───┬───┬───┬───┬───┬───┬───┐
   │ D │ D │ D │ D │   │   │   │   │   ring buffer (8 slots shown)
   └───┴───┴───┴───┴───┴───┴───┴───┘
       ▲
       │
   read_ptr (owned by PL, advances as
    hardware finishes each descriptor)
```

In the picture above, slots between `read_ptr` and `write_ptr` are descriptors the hardware hasn't gotten to yet; everything else is free space the PS can reuse. That raises the classic ring-buffer headache: **how do you tell "empty" apart from "full"?** If `read_ptr == write_ptr`, does that mean there's nothing left to do, or does it mean the ring is completely full and just wrapped all the way around back to where it started? Both situations produce the exact same pointer equality, so you need an extra bit of information to break the tie. Two standard fixes exist. One is to deliberately **leave one slot always empty** — you declare the ring full when `write_ptr` is one slot behind `read_ptr` (mod the ring size), which sacrifices one descriptor's worth of capacity but keeps the comparison logic dead simple. The other is to **widen each pointer by one extra "wrap" bit** that flips every time the pointer wraps around the end of the array; then `read_ptr == write_ptr` with matching wrap bits means empty, and matching indices with *different* wrap bits means full, using the full capacity of the ring. For this NPU, go with the wrap-bit approach — you have very few BRAM blocks to spare, and giving up one whole descriptor slot out of a small ring is a bigger relative cost here than it would be in a large ring buffer.

Now, the **[[concepts#Doorbell|doorbell]]**. Once the PS has appended one or more new descriptors to the ring, how does hardware find out there's new work? The naive answer — have the sequencer continuously poll the write-pointer value in DDR every cycle — burns power and, worse, adds memory-bus traffic competing with your actual data movement, for a value that changes only occasionally. The doorbell flips this around: the PS writes the *new* write-pointer value into a dedicated AXI-Lite register (the "doorbell register") after it's done adding descriptors, and that register write is itself the wake-up signal. The sequencer sits idle, watching only that one register, and the instant the doorbell value changes, it knows there's new work and starts fetching from wherever its read pointer currently sits. One cheap register write replaces continuous, wasteful polling of DDR.

Last piece: **why does the ring size need to be a power of two?** Because wraparound arithmetic on a power-of-two-sized ring collapses into a single bitwise AND. If your ring has, say, 16 entries, then advancing a pointer and wrapping it is just `ptr = (ptr + 1) & 0xF` — one cheap gate-level operation. With a non-power-of-two size, say 12 entries, you'd need an actual comparison (`if (ptr == 12) ptr = 0; else ptr++`) or a modulo operation, which is more logic, another comparator, and another place for an off-by-one bug to hide. In a design with a ~1,000-LUT budget for the whole sequencer, that difference between "one AND gate" and "a comparator plus a conditional reset" is not trivial — and correctness-wise, the comparator version is exactly the kind of thing that looks right in simulation and then breaks the one time the ring actually wraps during a long-running test.

---

## This is how real GPUs work

> [!quote] You already know this pattern
> This is a **baby PM4-style ring buffer** — the same fundamental mechanism that real GPU command processors use to accept work from the CPU, just scaled down to fit ~1,000 LUTs and two transformer-layer opcodes instead of a full graphics/compute instruction set.
>
> | Your NPU | How a real GPU does it |
> |---|---|
> | Descriptor | Command packet (e.g. a PM4-style packet) |
> | Doorbell register | Doorbell register |
> | Sequencer FSM | Command processor (CP) front-end |
> | Descriptor chain in DDR | Indirect buffer / command ring |
>
> The parallels aren't a loose analogy — they're structurally the same problem solved the same way: a CPU that doesn't want to babysit a device writes a batch of work into memory as a list of self-describing packets, then rings a doorbell so the device can pull work at its own pace and interrupt only when it's actually done. This is exactly why open-source GPU stacks are worth reading once you've built this phase yourself. The Linux `amdgpu` kernel driver (public on [gitlab.freedesktop.org/drm/amd](https://gitlab.freedesktop.org/drm/amd)) builds and submits command buffers into ring buffers and rings real doorbells to wake the hardware command processor, and Mesa's open-source GPU drivers (also on [gitlab.freedesktop.org/mesa/mesa](https://gitlab.freedesktop.org/mesa/mesa)) are what actually constructs those command-buffer packets user-space code ends up submitting. None of that code will look unfamiliar to you anymore — you'll recognize the ring, the head/tail pointers, and the doorbell write immediately, because you built the same shape of machine yourself, just sized for a 5-layer transformer instead of a modern GPU's full instruction set.

---

## 6.1 Define the descriptor format

> [!abstract] Goal
> Nail down the exact byte layout of a descriptor so the C code that builds it and the Verilog that reads it agree on every single bit, with no ambiguity.

**What's going on:** This exercise is pure bookkeeping, and it matters more than it looks like it should. The PS writes a descriptor as a C struct; the PL reads that same descriptor as a raw stream of bytes off an AXI burst and pulls fields out of fixed bit ranges. If the C compiler pads your struct differently than the Verilog expects — say, by inserting invisible padding bytes to align a field, which compilers do by default — the two sides silently disagree about where each field starts, and you get garbage src/dst addresses with no error message anywhere. You're going to make the layout airtight before you build anything that depends on it.

**Steps:**
1. Write the descriptor struct with `__attribute__((packed))` so the compiler cannot insert padding between fields.
2. Add a `_Static_assert` that checks the struct's size at compile time, so a future edit that accidentally changes the layout fails the build instead of failing silently on hardware.
3. Write the matching Verilog field-extraction logic that pulls each field out of a 32-byte descriptor fetched over AXI, using the exact same byte offsets.
4. Cross-check the two by hand once: list every field, its byte offset, and its width, in both the C struct and the Verilog extraction code, side by side.

```c
// descriptor.h — the exact wire format, shared by PS software and PL hardware
#include <stdint.h>

typedef enum {
    OP_MATMUL   = 0,
    OP_RMSNORM  = 1,
    OP_SOFTMAX  = 2,
    OP_SILU     = 3,
    OP_ROPE     = 4,
    OP_REQUANT  = 5,
} opcode_t;

typedef struct __attribute__((packed)) {
    uint32_t opcode;          // offset  0
    uint32_t src_addr;        // offset  4
    uint32_t dst_addr;        // offset  8
    uint32_t M;                // offset 12
    uint32_t N;                // offset 16
    uint32_t K;                // offset 20
    uint32_t requant_M0;       // offset 24
    uint32_t requant_shift;    // offset 28
} descriptor_t;                // total: 32 bytes — a power of two, on purpose

// Fails the build at compile time if anyone changes a field and breaks the
// 32-byte layout the Verilog side depends on.
_Static_assert(sizeof(descriptor_t) == 32,
               "descriptor_t must stay exactly 32 bytes — check PL side too!");
```

```verilog
// descriptor_decode.v — pulls the 8 fields out of one 32-byte (256-bit)
// descriptor fetched over an AXI burst. Byte offsets must match descriptor_t
// in descriptor.h EXACTLY, field for field.
module descriptor_decode (
    input  wire [255:0] desc_bytes,     // 32 bytes, little-endian byte order
    output wire [31:0]  opcode,
    output wire [31:0]  src_addr,
    output wire [31:0]  dst_addr,
    output wire [31:0]  M, N, K,
    output wire [31:0]  requant_M0,
    output wire [31:0]  requant_shift
);
    // Each field is 4 bytes = 32 bits. Offsets in BITS, matching the byte
    // offsets in the C struct (offset_bytes * 8).
    assign opcode         = desc_bytes[ 31:  0];  // byte offset  0
    assign src_addr       = desc_bytes[ 63: 32];  // byte offset  4
    assign dst_addr       = desc_bytes[ 95: 64];  // byte offset  8
    assign M              = desc_bytes[127: 96];  // byte offset 12
    assign N              = desc_bytes[159:128];  // byte offset 16
    assign K              = desc_bytes[191:160];  // byte offset 20
    assign requant_M0     = desc_bytes[223:192];  // byte offset 24
    assign requant_shift  = desc_bytes[255:224];  // byte offset 28
endmodule
```

> [!success] Done when
> You can generate a descriptor in C, dump its raw bytes to a hex file, feed those exact bytes into your Verilog `descriptor_decode` module in simulation, and every one of the 8 output fields matches what you put into the struct — no shifted fields, no swapped bytes.

> [!question]- It's not working
> 1. Did you add `__attribute__((packed))` to the struct? Without it, the compiler is free to insert padding for alignment, and your byte offsets silently stop matching.
> 2. Does `sizeof(descriptor_t)` actually equal 32? Let the `_Static_assert` catch this for you instead of discovering it in simulation.
> 3. Are you matching byte order correctly — is the C code running little-endian (ARM Cortex-A9 is little-endian by default), and does your Verilog bit-slicing assume the same byte order?
> 4. Did you double check the bit range math? Byte offset 4 is bits `[63:32]`, not `[39:32]` — that's `(byte_offset * 8)` to `(byte_offset * 8 + 31)`.
> 5. Are you reading all 32 bytes (256 bits) in one AXI burst, or accidentally reading a truncated burst that clips the last field or two?
> 6. If a field looks "close but shifted," check for an accidental off-by-one in which byte offset you started counting fields from.

> [!bug] Gotcha
> Make the descriptor size a **power of two** and align it. Ring wraparound logic with non-power-of-two entries is a bug farm — it turns a one-instruction AND-mask into a comparator-and-branch, and it's exactly the kind of thing that passes every test until the ring wraps for the first time in a long-running run.

---

## 6.2 Build the ring and doorbell

> [!abstract] Goal
> Get a full chain of descriptors from PS software into PL hardware's hands, correctly, using an aligned ring buffer and a doorbell register.

**What's going on:** This exercise wires together everything from the last section into something that actually runs. On the C side, you allocate the ring, fill it with real descriptors for one transformer layer, make sure the cache doesn't lie to the hardware about what's in DDR, and ring the doorbell. On the Verilog side, you build the doorbell register itself and the comparison logic that tells the sequencer whether there's unfetched work waiting.

**Steps:**
1. Allocate a ring buffer in DDR, sized as a power of two, aligned to a cache-line boundary.
2. Fill it with descriptors for a whole transformer layer — one descriptor per operation, in execution order.
3. **Flush the cache** for the range you just wrote, before you touch the doorbell register. The Cortex-A9's data cache means your writes may still be sitting in L1/L2, not yet visible in DRAM, when the PL goes to read them — the same [[concepts#Cache coherency|cache coherency]] issue you already hit with bulk weight transfers in [[phase-4-memory|Phase 4]]. `Xil_DCacheFlushRange()` before the doorbell write is not optional.
4. Write the new write-pointer value to the doorbell register — this is the signal that tells hardware "go."
5. On the Verilog side, implement the doorbell register itself and the pointer-compare logic a fetch state machine will use to decide whether the ring is empty.

```c
// build_ring.c — allocate an aligned ring, fill it with one layer's worth of
// descriptors, flush the cache, then ring the doorbell.
#include "descriptor.h"
#include "xil_cache.h"

#define RING_SIZE   16                 // power of two — required
#define RING_MASK   (RING_SIZE - 1)

// Cache-line aligned so a flush doesn't accidentally touch neighboring data.
static descriptor_t ring[RING_SIZE] __attribute__((aligned(32)));
static uint32_t write_ptr = 0;         // includes a wrap bit — see note below

void push_descriptor(descriptor_t d) {
    uint32_t idx = write_ptr & RING_MASK;
    ring[idx] = d;
    write_ptr++;                       // wraps naturally past RING_SIZE-1 bits;
                                        // the extra high bit acts as the wrap flag
}

void build_layer_chain(void) {
    // One real transformer layer's sequence — not exhaustive, but the shape:
    push_descriptor((descriptor_t){ .opcode = OP_RMSNORM, /* ... */ });
    push_descriptor((descriptor_t){ .opcode = OP_MATMUL,  /* Q proj  */ });
    push_descriptor((descriptor_t){ .opcode = OP_MATMUL,  /* K proj  */ });
    push_descriptor((descriptor_t){ .opcode = OP_MATMUL,  /* V proj  */ });
    push_descriptor((descriptor_t){ .opcode = OP_ROPE,    /* Q       */ });
    push_descriptor((descriptor_t){ .opcode = OP_ROPE,    /* K       */ });
    push_descriptor((descriptor_t){ .opcode = OP_MATMUL,  /* scores  */ });
    push_descriptor((descriptor_t){ .opcode = OP_SOFTMAX, /* ...     */ });
    push_descriptor((descriptor_t){ .opcode = OP_MATMUL,  /* weighted sum */ });
    push_descriptor((descriptor_t){ .opcode = OP_RMSNORM, /* pre-FFN */ });
    push_descriptor((descriptor_t){ .opcode = OP_MATMUL,  /* FFN gate */ });
    push_descriptor((descriptor_t){ .opcode = OP_MATMUL,  /* FFN up   */ });
    push_descriptor((descriptor_t){ .opcode = OP_SILU,    /* ...      */ });
    push_descriptor((descriptor_t){ .opcode = OP_MATMUL,  /* FFN down */ });

    // CRITICAL: flush before the doorbell write. Without this, the PL may
    // read stale (pre-write) bytes straight out of DDR.
    Xil_DCacheFlushRange((INTPTR)ring, sizeof(ring));

    // The doorbell write IS the "go" signal — one AXI-Lite register write.
    *(volatile uint32_t *)DOORBELL_REG_ADDR = write_ptr;
}
```

```verilog
// doorbell_and_ring_state.v — doorbell register plus empty/full comparison,
// using an extra wrap bit on each pointer to disambiguate empty from full.
module ring_pointers #(
    parameter RING_ADDR_BITS = 4          // 2^4 = 16 entries
) (
    input  wire        clk, rst_n,
    input  wire [RING_ADDR_BITS:0] doorbell_write_ptr, // one extra bit = wrap flag
    output reg  [RING_ADDR_BITS:0] read_ptr,           // owned by hardware
    output wire        ring_has_work
);
    // "Empty" means read_ptr == write_ptr, including the wrap bit.
    // "Full" would mean same index, DIFFERENT wrap bit (not needed here,
    // since only the PL advances read_ptr and only the PS advances write_ptr).
    assign ring_has_work = (read_ptr != doorbell_write_ptr);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            read_ptr <= 0;
        // read_ptr is advanced elsewhere, by the fetch FSM in 6.3,
        // once each descriptor finishes executing.
    end
endmodule
```

> [!success] Done when
> The PS builds a full layer's descriptor chain, flushes the cache, and rings the doorbell — and in simulation (or with an ILA on real hardware), you can see the PL's `ring_has_work` signal go high the instant the doorbell register updates, with the correct number of pending descriptors reflected in the pointer difference.

> [!question]- It's not working
> 1. Did you call `Xil_DCacheFlushRange()` on the ring *before* writing the doorbell register, not after? Order matters — the flush has to happen first so DDR reflects your writes before hardware is told to look.
> 2. Is your ring size actually a power of two, and does `RING_MASK` correctly compute `RING_SIZE - 1`?
> 3. Does the descriptor struct's packed layout in C match the field extraction in Verilog byte-for-byte (revisit 6.1 if you changed anything)?
> 4. Did the read pointer wrap correctly the first time the ring filled past its last slot — test this explicitly with more descriptors than fit in one pass through the ring, don't just test with a chain shorter than `RING_SIZE`?
> 5. Is the interrupt (for when the whole chain finishes) actually connected and enabled in your Zynq block design — check both the PL-side interrupt line and the GIC/interrupt controller configuration on the PS side?
> 6. Is the doorbell register wide enough to hold the extra wrap bit, on both the C side (`write_ptr` type) and the Verilog side (`RING_ADDR_BITS+1` wide)?
> 7. If the pointer difference looks wrong by exactly `RING_SIZE`, that's the classic sign the wrap bit isn't being compared correctly.

> [!bug] Gotcha
> Make the descriptor size a **power of two** and align it. Ring wraparound logic with non-power-of-two entries is a bug farm.

---

## 6.3 The sequencer FSM

> [!abstract] Goal
> Build the state machine that turns a descriptor chain sitting in DDR into actual work happening on the matmul array and vector unit, unattended, with zero PS involvement until the whole chain is done.

**What's going on:** This is the heart of the phase — the actual hardware that makes everything above mean something. The sequencer is a small [[concepts#FSM|FSM]] that loops: fetch the next descriptor over AXI, decode its opcode, hand it off to whichever unit should execute it (the [[phase-3-mac-engine|matmul array]] or the [[phase-5-vector-unit|vector unit]]), wait for that unit to report done, advance the read pointer, and check whether more work is waiting. When the ring finally empties, it raises one interrupt and goes back to idle. Everything before this exercise was about getting the *data* (the descriptor chain) into a shape hardware can consume; this exercise is about the *control logic* that consumes it.

**Steps:**
1. Define the FSM's states as a Verilog enum — one state per phase of the fetch/decode/dispatch/wait/advance loop.
2. Write the fetch state: issue an AXI read for the next 32-byte descriptor at `read_ptr`.
3. Write the decode state: run the fetched bytes through the `descriptor_decode` module from 6.1.
4. Write the dispatch state: based on `opcode`, start the matmul array or the vector unit, handing it the relevant fields.
5. Write the wait state: sit until the dispatched unit asserts its own `done` signal.
6. Advance `read_ptr`, and either loop back to fetch (if `ring_has_work` is still true) or raise the interrupt and return to idle.

```
        ┌────────┐
        │  IDLE  │◄───────────────────────────────────┐
        └───┬────┘                                    │
            │ ring_has_work                            │
            ▼                                          │
        ┌────────┐                                     │
        │ FETCH  │  (AXI read of 32B descriptor)        │
        └───┬────┘                                     │
            ▼                                          │
        ┌────────┐                                     │
        │ DECODE │  (descriptor_decode)                 │
        └───┬────┘                                     │
            ▼                                          │
        ┌──────────┐                                   │
        │ DISPATCH │  (route by opcode)                 │
        └───┬──────┘                                   │
            ▼                                          │
        ┌────────┐                                     │
        │  WAIT  │  (poll unit's done signal)           │
        └───┬────┘                                     │
            ▼                                          │
        ┌─────────┐   ring_has_work?  ──── yes ─────────┘
        │ ADVANCE │──── no ───────────┐
        └─────────┘                  ▼
                                 ┌───────────┐
                                 │ INTERRUPT │──► back to IDLE
                                 └───────────┘
```

```verilog
// sequencer_fsm.v — fetch/decode/dispatch/wait loop over a descriptor ring.
// ~commented reference; wire the real AXI read/write and unit handshakes
// for your board.

typedef enum logic [2:0] {
    S_IDLE      = 3'd0,   // nothing to do, waiting on doorbell
    S_FETCH     = 3'd1,   // AXI read of the next 32-byte descriptor
    S_DECODE    = 3'd2,   // split the fetched bytes into fields
    S_DISPATCH  = 3'd3,   // route to matmul array or vector unit by opcode
    S_WAIT      = 3'd4,   // wait for the dispatched unit's done pulse
    S_ADVANCE   = 3'd5,   // bump read_ptr, decide whether to loop or finish
    S_INTERRUPT = 3'd6    // ring is drained — raise interrupt, go idle
} seq_state_t;

module sequencer_fsm (
    input  wire        clk, rst_n,
    input  wire         ring_has_work,     // from ring_pointers (6.2)
    input  wire [31:0]  desc_fetch_data,   // one word of the AXI read burst
    input  wire         desc_fetch_done,   // full 32B descriptor now valid
    output reg  [31:0]  read_ptr,
    // Decoded descriptor fields (from descriptor_decode, 6.1):
    input  wire [31:0]  opcode, src_addr, dst_addr, M, N, K,
    input  wire [31:0]  requant_M0, requant_shift,
    // Handshakes to the two execution units:
    output reg           matmul_start,
    input  wire          matmul_done,
    output reg           vecunit_start,
    output reg  [2:0]    vecunit_mode,      // RMSNORM/SOFTMAX/SILU/ROPE select
    input  wire          vecunit_done,
    output reg           irq_out
);
    seq_state_t state, next_state;

    always @(posedge clk or negedge rst_n)
        if (!rst_n) state <= S_IDLE;
        else        state <= next_state;

    always @(*) begin
        // Defaults every cycle — avoids accidental latches.
        next_state    = state;
        matmul_start  = 1'b0;
        vecunit_start = 1'b0;
        irq_out       = 1'b0;

        case (state)
            S_IDLE: begin
                if (ring_has_work) next_state = S_FETCH;
            end

            S_FETCH: begin
                // AXI burst read of the descriptor at read_ptr is issued
                // outside this snippet; wait here until it lands.
                if (desc_fetch_done) next_state = S_DECODE;
            end

            S_DECODE: begin
                // descriptor_decode (6.1) is combinational on desc_fetch_data;
                // one cycle here to let opcode/src/dst/etc. settle.
                next_state = S_DISPATCH;
            end

            S_DISPATCH: begin
                case (opcode)
                    32'd0: begin matmul_start  = 1'b1; next_state = S_WAIT; end // OP_MATMUL
                    32'd1: begin vecunit_start = 1'b1; vecunit_mode = 3'd0;
                                 next_state = S_WAIT; end                       // OP_RMSNORM
                    32'd2: begin vecunit_start = 1'b1; vecunit_mode = 3'd1;
                                 next_state = S_WAIT; end                       // OP_SOFTMAX
                    32'd3: begin vecunit_start = 1'b1; vecunit_mode = 3'd2;
                                 next_state = S_WAIT; end                       // OP_SILU
                    32'd4: begin vecunit_start = 1'b1; vecunit_mode = 3'd3;
                                 next_state = S_WAIT; end                       // OP_ROPE
                    default: next_state = S_ADVANCE;  // unknown opcode: skip, don't hang
                endcase
            end

            S_WAIT: begin
                // Whichever unit we dispatched to, wait for its own done pulse.
                if (matmul_done || vecunit_done) next_state = S_ADVANCE;
            end

            S_ADVANCE: begin
                // read_ptr++ happens in the sequential block below;
                // decide here whether there's more work waiting.
                next_state = ring_has_work ? S_FETCH : S_INTERRUPT;
            end

            S_INTERRUPT: begin
                irq_out    = 1'b1;   // pulse the interrupt line
                next_state = S_IDLE;
            end

            default: next_state = S_IDLE;
        endcase
    end

    // read_ptr advance — separate sequential block, bumps once per completed
    // descriptor, in the ADVANCE state.
    always @(posedge clk or negedge rst_n)
        if (!rst_n) read_ptr <= 0;
        else if (state == S_ADVANCE) read_ptr <= read_ptr + 1'b1; // wraps via the
                                                                    // extra high bit,
                                                                    // power-of-two ring
endmodule
```

> [!success] Done when
> A full descriptor chain for one transformer layer executes start to finish with a single doorbell write, the PS observes exactly one interrupt at the end (not one per operation), and the sequencer correctly routes each opcode to the matmul array or the vector unit, waits for its `done`, and drains the entire ring before raising that interrupt.

> [!question]- It's not working
> 1. Did you flush the cache (`Xil_DCacheFlushRange`) before ringing the doorbell? If the PL reads stale descriptor bytes, you'll see wrong or garbage opcodes with no obvious cause.
> 2. Is the descriptor struct packed and aligned the same way in C and in the Verilog `descriptor_decode` module — did anything about the layout drift since 6.1?
> 3. Is the ring size still a power of two, and does the read pointer's wrap bit toggle correctly on every full pass through the ring (test with a chain longer than one pass)?
> 4. Did the read pointer actually advance in `S_ADVANCE`, or is it stuck — a classic symptom is the FSM fetching the same descriptor forever?
> 5. Is the interrupt line actually connected and enabled in the Zynq block design — check the PL-to-PS interrupt wiring *and* that the corresponding interrupt is unmasked in the GIC on the PS side, not just tied off in simulation?
> 6. Does `S_DISPATCH` have a `default` case that safely skips unknown opcodes, or will a corrupted/garbage opcode hang the FSM forever in `S_WAIT`?
> 7. If only the *first* descriptor in a chain ever executes, check whether `ring_has_work` is being recomputed against the updated `read_ptr`, not a stale cached value from before `S_ADVANCE`.

> [!bug] Gotcha
> Make the descriptor size a **power of two** and align it. Ring wraparound logic with non-power-of-two entries is a bug farm.

---

## Check your understanding

> [!question]- Why does batching descriptors into a chain actually save time, when the PL still has to execute the exact same number of operations either way?
> > [!success]- Answer
> > The compute work doesn't shrink — the same matmuls, RMSNorms, softmaxes, and so on still have to run. What disappears is the *coordination overhead*: the ~8 AXI-Lite register writes, the wait, and the status read the PS used to do around every single operation. With a descriptor chain, the PS pays that overhead once per whole layer instead of once per operation — one doorbell write instead of dozens of register writes — and the PS is free to do other work (like preparing the next token) while the PL works through the chain unattended. The savings come entirely from eliminating repeated round-trips, not from making any individual operation faster.

> > [!question]- Why does the ring buffer need a wrap bit (or a sacrificed empty slot) instead of just comparing `read_ptr == write_ptr`?
> > [!success]- Answer
> > Because `read_ptr == write_ptr` is ambiguous on its own — it's exactly what you'd see both when the ring is completely empty (hardware has consumed everything software produced) and when the ring is completely full and has wrapped exactly once more than it's been drained (software has produced a full ring's worth since hardware last caught up). Both states produce identical pointer values, so the comparison alone can't tell them apart. Adding an extra "wrap" bit to each pointer (or reserving one slot as permanently unusable) gives you the missing bit of information needed to disambiguate the two cases.

> [!question]- Why is the cache flush before the doorbell write not optional, and why doesn't it show up as an obvious crash when it's missing?
> > [!success]- Answer
> > The Cortex-A9 has a data cache, and a normal `struct` write from C code lands in that cache first — it isn't guaranteed to be visible in DDR until the cache line is evicted or explicitly flushed. If the doorbell is rung before `Xil_DCacheFlushRange()` runs, the PL can read stale, pre-write bytes straight out of DDR while your actual descriptor data is still sitting in the ARM core's cache. Nothing crashes — the sequencer just fetches whatever old bytes happened to be there and either dispatches garbage opcodes or silently re-runs a stale, previous descriptor chain. It looks exactly like a hardware bug in the sequencer FSM, and that's what makes it dangerous: people spend hours debugging Verilog for what is actually a one-line missing cache flush on the software side.

> [!question]- Why is a power-of-two descriptor size and ring size worth enforcing as a hard rule rather than a nice-to-have?
> > [!success]- Answer
> > With a power-of-two size, both the descriptor's byte offset within the ring and the ring's own wraparound reduce to a bitwise AND mask — one gate-level operation, cheap in both LUTs and timing, and there's no way to get it subtly wrong. With any other size, you need an actual comparator and a conditional reset (or a real modulo), which costs more logic in an already-tight ~1,000-LUT budget and introduces an extra opportunity for an off-by-one bug that will pass every short test and then fail the first time the ring genuinely wraps in a long-running session — exactly the kind of bug that's expensive to track down after the fact.

---

## What's next

You now have a sequencer that can take a whole transformer layer's worth of descriptors, execute them unattended across the matmul array and vector unit, and interrupt the PS exactly once when the chain drains. [[phase-7-end-to-end]] is where you wire this sequencer together with everything from Phases 1 through 6 into one complete, working inference pipeline.

<script src="/tutor.js" defer></script>
