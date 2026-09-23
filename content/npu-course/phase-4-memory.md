---
title: "Phase 4 — Feeding the Beast"
description: "Tiling, double-buffering, a hand-written AXI4 burst master, and measuring your actual DDR roofline."
tags:
  - npu
  - hardware
---

# Phase 4 — Feeding the beast

**5 weeks** · Part of [[index|Zero to NPU]] · Prev: [[phase-3-mac-engine]] · Next: [[phase-5-vector-unit]]

> [!info] What this phase is for
> Phase 3 built an engine — a real 8×4 grid of PEs that can multiply and accumulate every cycle. This phase builds the fuel line. An engine is only as fast as the fuel reaching its cylinders, and right now your array has no fuel line at all: it's sitting next to 512 MB of DDR3 with no way to get data in or out faster than one word at a time. A perfect MAC array that's starved of data is a perfect waste of DSPs — it will report beautiful utilization numbers in simulation and then sit idle 90% of the time on real silicon, waiting on memory. From here on, most of the performance work in this course is memory work, not math. You already built the math. Now you build the pipes, and you learn to reason about *why* the pipes are the bottleneck, not the multipliers.

---

## The memory hierarchy you're working with

Every system that moves data to compute engines has layers, and each layer trades capacity for speed. On the XC7Z007S you have exactly three layers that matter, and you need real numbers for all of them or every decision in this phase is just vibes.

**Registers / PE-internal storage.** Inside each PE from [[phase-3-mac-engine]] you have a handful of registers holding the current weight, activation, and accumulator. Access is instant — same clock cycle, no contention, no wiring delay worth talking about. Capacity is a few words per PE. This is the innermost, fastest, smallest tier, and it's basically free.

**[[concepts#BRAM|BRAM]].** The 7007S gives you about 225 KB of on-chip block RAM, organized as dual-port blocks of roughly 36 Kb (4.5 KB) each. BRAM is fast — one cycle to read or write, no bus arbitration if you design your address decode sensibly — and it's *on-chip*, meaning no pins, no off-chip electrical delay. But it's scarce. 225 KB has to hold your weight tiles, your activation tiles, your double-buffering copies (more on that below), and anything else you need resident. You will spend real design effort deciding what earns a spot in BRAM.

**DDR3.** This is where the bulk of your model actually lives: 512 MB, 16-bit wide, running at 533 MHz, with a *theoretical* peak around 2.1 GB/s. But peak bandwidth is a marketing number — it assumes back-to-back bursts with no refresh cycles, no row-buffer misses, no read/write turnaround penalty, and no other AXI master competing for the same DDR controller. In practice you should plan around **0.8–1.5 GB/s sustained**, and use **~1.2 GB/s** as your working number for every capacity calculation in this course. DDR3 is also *slow to start*: a single access has a latency of tens of cycles before the first word even shows up, even though once the pipe is flowing it can move a lot of data per cycle.

That last sentence is the single most important distinction in this phase: **[[concepts#Bandwidth|bandwidth]] and latency are not the same axis**, and beginners conflate them constantly. Latency is "how long until this specific piece of data arrives." Bandwidth is "how many bytes per second once data is flowing." DDR3 has *high latency* (tens of cycles before the first word) but can have *decent bandwidth* (a steady stream of words afterward) — provided you ask for data in big enough chunks that the fixed latency cost gets amortized over many bytes. Ask DDR3 for one word at a time and you pay the full tens-of-cycles latency penalty on every single word — you'll get a small fraction of the theoretical bandwidth. Ask for 256 words in one request (a burst) and you pay that latency penalty once, then the words stream out fast. This is the entire reason [[concepts#Burst|bursts]] exist, and it's why the rest of this phase is built around them.

So the whole game, top to bottom: keep the PE array fed from BRAM every cycle, and keep BRAM refilled from DDR3 in the background, in big bursts, while the array keeps computing from whatever's already in BRAM. That's it. That's the phase.

---

## Tiling: why you can't just multiply the whole matrix

Say you want to compute a 64×64 matrix multiplied by another 64×64 matrix. That's the kind of size you'll hit almost immediately once you leave toy examples. Your array is 8 rows × 4 columns of PEs — 32 PEs total. You cannot map a 64×64 × 64×64 problem onto 32 PEs in one shot, for two independent reasons, and it's worth separating them:

1. **The array is too small.** A 64×64 output has 4,096 elements. Your array computes 32 partial products per cycle (one per PE, in the systolic/parallel scheme from Phase 3). There is no physical wiring that lets 32 PEs "become" a 4,096-wide array for one instant.
2. **BRAM is too small.** Even if you could compute the whole thing at once, a 64×64 matrix of, say, INT8 weights is 4,096 bytes — that's fine actually — but once you're doing meaningful model sizes (which is the whole point of this course), your weight and activation matrices will not fit in 225 KB of BRAM simultaneously alongside everything else you need resident.

The fix is **[[concepts#Tiling|tiling]]**: cut both matrices into small rectangular chunks ("tiles") sized to match what the array and BRAM can actually hold at once, and loop over the tiles, accumulating partial results.

For an 8×4 array, the natural tile is **8×4**: 8 rows feed the 8 PE-rows, 4 columns feed the 4 PE-columns, and every element of that tile maps to exactly one PE for one cycle of useful work. A 64×64 matrix tiled at 8×4 gives you (64/8) × (64/4) = 8 × 16 = **128 tiles** just for one of the two input matrices. Multiplying two 64×64 matrices means iterating over an additional "K" dimension (the shared 64-length dimension being contracted), which itself tiles into 64/4 = 16 steps if you also tile K by 4 (matching the PE column depth). So the full computation becomes a **triple-nested loop**: outer over output row-tiles, middle over output column-tiles, inner over K-tiles, with a MAC-accumulate happening in the array on every inner-loop iteration.

Here's the loop nest with real numbers filled in, in C-like pseudocode:

```c
// C[64][64] = A[64][64] x B[64][64], INT8 elements
// Array is 8 rows x 4 cols -> tile M=8, N=4, K=4
#define M 64, N 64, K 64
#define TM 8, TN 4, TK 4

for (int mt = 0; mt < M/TM; mt++) {       // 8 row-tiles of output
  for (int nt = 0; nt < N/TN; nt++) {     // 16 col-tiles of output
    // acc[TM][TN] lives in the PE array's own accumulator registers
    zero_accumulators();
    for (int kt = 0; kt < K/TK; kt++) {   // 16 K-tiles (the reduction dimension)
      load_tile_A(mt, kt);   // 8x4 tile of A into BRAM buffer
      load_tile_B(kt, nt);   // 4x4 tile of B into BRAM buffer
      mac_array_step();      // 32 PEs each do one MAC, accumulate in place
    }
    store_tile_C(mt, nt);    // write the finished 8x4 output tile back to DDR
  }
}
```

Total tile-load-and-compute steps: 8 (mt) × 16 (nt) × 16 (kt) = **2,048 inner iterations**, each one a burst-load of a small A tile and a small B tile followed by one array cycle of MAC work. Notice which operand is a good candidate to keep resident across iterations: for a fixed `mt`, the A tile only depends on `kt`, and B only depends on `kt, nt` — a smarter loop order (or blocking scheme) lets you reuse a loaded A row-block across multiple `nt` iterations instead of re-fetching it from DDR every time. That reuse is exactly the kind of thing that turns "DDR-bandwidth-bound" into "compute-bound," and it's worth sketching your own loop order on paper before you write RTL — this is 4.1's real job.

---

## AXI4, explained for someone who has only used AXI-Lite

In [[phase-0-toolchain|Phase 0]] you used [[concepts#AXI|AXI]]-Lite: one address, one data word, done — simple, but strictly one word per transaction, with a full address-and-handshake round trip paid *every single word*. That's fine for configuration registers. It is disastrously slow for moving megabytes of weights, because it pays DDR3's tens-of-cycles latency penalty on every word instead of once per many words.

AXI4 (the full protocol, not AXI-Lite) fixes this by splitting a transaction into **five independent channels**, each with its own valid/ready handshake:

- **AW — write address channel.** Carries the target address for a write, once.
- **W — write data channel.** Carries the actual data beats for that write, one or many.
- **B — write response channel.** The slave tells you the write completed (or failed).
- **AR — read address channel.** Carries the target address for a read, once.
- **R — read data channel.** Carries the data beats coming back, one or many.

Every channel uses the same two-signal handshake pattern: a `VALID` signal from the sender saying "this data is ready," and a `READY` signal from the receiver saying "I can accept it this cycle." A transfer happens on any clock edge where **both** VALID and READY are high, on that channel, simultaneously. Critically: **VALID must never wait for READY**. The sender is allowed to assert VALID and hold data stable, but it cannot look at READY first and decide whether to assert VALID — that creates a combinational dependency loop that the AXI spec explicitly forbids, and it's the single most common beginner bug in a first AXI master.

The idea that makes this fast is the **[[concepts#Burst|burst]]**: instead of one address-handshake-per-word, you send **one address handshake, then many data beats**. You pay the address-phase overhead (and the DDR latency behind it) exactly once, then the data streams. This is precisely the fix for the latency-vs-bandwidth problem from the memory hierarchy section above.

The AR channel signals you actually need for 4.3:

- **ARADDR** — the starting byte address of the burst.
- **ARLEN** — number of beats **minus one**. If you want 16 beats, ARLEN = 15. This trips everyone up at least once; write it down.
- **ARSIZE** — bytes per beat, as a power-of-two exponent. `3'b011` = 8 bytes/beat (a 64-bit HP port running at full width).
- **ARBURST** — burst type. You want `2'b01` = **INCR** (incrementing address per beat) for sequential DDR reads. (WRAP and FIXED exist but you won't need them here.)
- **ARVALID / ARREADY** — the handshake pair for this channel.

Then on the R channel, per beat: **RDATA** (the payload), **RLAST** (asserted on the final beat of the burst — this is how the master knows to stop counting), and **RVALID / RREADY** (the handshake pair).

The **4 KB rule**: an AXI4 burst is not allowed to cross a 4 KB address boundary, ever, for any burst type. This exists because many AXI interconnects and memory controllers internally page-map addresses in 4 KB units — allowing a burst to straddle a page would mean the interconnect might have to route the same burst to two different destinations mid-transfer, which the protocol simply disallows to keep the interconnect logic tractable. The consequence for you: **the interconnect will not split an oversized or misaligned burst for you.** If your address generator asks for a burst that would cross a 4 KB line, you get undefined behavior, not an automatic split. Your address generator (built in 4.1, exercised again in 4.3) has to detect this case and issue two shorter bursts instead of one.

Here's a 4-beat read burst as an ASCII timing diagram — one address phase, four data beats, RLAST on the last one:

```
clk      : _/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_
ARVALID  : ‾‾\________________________________
ARREADY  : ‾‾‾‾\____________________________
ARADDR   : <0x1000>--------------------------
           (address handshake completes here)

RVALID   : ________/‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\____
RREADY   : ________/‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\____
RDATA    : ________<D0>< D1>< D2>< D3>______
RLAST    : ______________________/‾‾‾\________
                                 (only on D3)
```

Notice the gap between the address handshake completing and the first data beat arriving — that gap *is* the DDR latency you learned about above. Once R starts flowing, the beats come back to back if the DDR controller can keep up, and you only paid that latency once for all four beats instead of four times.

---

## Double buffering

Tiling and AXI bursts solve *how* to move data efficiently. They don't solve *when*. If your array sits idle every time you fetch a tile, you've built a fast fuel line that still stalls the engine at every fill-up.

The fix is **[[concepts#Double buffering|double buffering]]** (also called ping-pong buffering): allocate **two** copies of your BRAM tile buffer, A and B. While the array is computing from buffer A, your AXI4 burst master is already fetching the *next* tile into buffer B in the background. When the array finishes with A, it instantly swaps to B (a one-cycle pointer flip, not a copy), and the burst master starts refilling A with the tile after that. The array, ideally, never sees a stall — it just keeps consuming whichever buffer is "ready," while the other one is always being refilled behind the scenes.

Before double buffering, your timeline looks like this — compute and fetch strictly alternate, and the array is idle for the entire fetch:

```
time -->
Array:  [compute A]......[compute B]......[compute C]......
Fetch:  ..........[fetch B]......[fetch C]......[fetch D]..
        ^stall^            ^stall^            ^stall^
```

After double buffering, fetch overlaps with compute, and the array only stalls if a fetch takes *longer* than the compute it's overlapping with:

```
time -->
Array:  [compute A][compute B][compute C][compute D]
Fetch:  ..[fetch B][fetch C ][fetch D  ]............
```

This is exactly the mechanism that gets you from "correct but slow" (4.1's done-when) to ">80% PE utilization" (4.2's done-when). It costs you double the BRAM footprint for whatever's being double-buffered, which is one more reason BRAM budgeting matters.

---

## 4.1 Tiling and the address generator

> [!abstract] Goal
> Turn tile indices `(mt, nt, kt)` into real DDR byte addresses, on paper and in code, before you touch AXI.

**What's going on:** Your array consumes 8×4 tiles, but DDR3 stores your matrices as flat byte arrays, row-major. Every tile load needs an address generator that converts "which tile" into "which bytes." Get this arithmetic right and unambiguous now — 4.3's burst master will call this same logic to decide where each burst starts and how long it can run before hitting the 4 KB rule.

**Steps:**
1. On paper, work through the 64×64 × 64×64 example from the tiling section above: confirm 8 row-tiles, 16 column-tiles, 16 K-tiles, 2,048 total inner iterations.
2. Write the address arithmetic for a row-major INT8 matrix: given tile indices and matrix width, compute the byte offset of the tile's top-left element.
3. Implement it in Verilog as combinational logic (or a small function) so it can be reused by the burst master later.

```verilog
// Address generator: tile (mt, kt) of matrix A, row-major INT8, width = 64
// A_BASE   = base byte address of matrix A in DDR
// TILE_ROWS = 8, TILE_COLS = 4 (matches PE array), MAT_WIDTH = 64 (elements)
module tile_addr_gen #(
    parameter MAT_WIDTH  = 64,   // elements per row of the source matrix
    parameter TILE_ROWS  = 8,
    parameter TILE_COLS  = 4,
    parameter ELEM_BYTES = 1     // INT8
)(
    input  [31:0] base_addr,    // A_BASE
    input  [7:0]  mt,           // row-tile index
    input  [7:0]  kt,           // col-tile index (K dimension for A)
    output [31:0] tile_byte_addr
);
    // Row of the top-left element of this tile, in elements:
    wire [31:0] row_elem = mt * TILE_ROWS;
    // Column of the top-left element of this tile, in elements:
    wire [31:0] col_elem = kt * TILE_COLS;
    // Row-major offset: row * width + col, then convert elements -> bytes
    assign tile_byte_addr = base_addr
        + (row_elem * MAT_WIDTH + col_elem) * ELEM_BYTES;
endmodule
```

> [!success] Done when
> Given `(mt, kt)` for every tile in the 64×64 example, your generator's output addresses land exactly on the tile boundaries you computed by hand — verify at least the first tile, the last tile, and one tile in the middle of a row wrap.

> [!question]- It's not working
> 1. Did you convert element offsets to byte offsets (multiply by `ELEM_BYTES`), or are you off by a factor of the element size?
> 2. Is `MAT_WIDTH` the width of the *whole* matrix, not the tile — a common copy-paste mistake?
> 3. Does the address wrap correctly across a full row of tiles (i.e., does incrementing `kt` past the last column tile move to the next `mt` correctly in your loop, even if the address generator itself doesn't need to know about that)?
> 4. Are `mt`/`kt` wide enough to not overflow for your matrix size?
> 5. Did you check tile (0,0) lands exactly on `base_addr`?

> [!danger] Gotcha — this feeds 4.3 directly
> Whatever this module computes becomes the `ARADDR` your AXI4 master issues. If it's wrong here, 4.3 will look like an AXI bug when it's actually an addressing bug. Verify this module standalone, in simulation, against hand-computed addresses, before wiring it to anything AXI-shaped.

---

## 4.2 Double buffering

> [!abstract] Goal
> Build the ping-pong control logic that lets the array compute from one BRAM buffer while the next tile is being fetched into the other.

**What's going on:** You need one bit of state (which buffer is "active" for compute) and two "done" signals (compute-done, fetch-done) that together decide when to flip that bit. Get the flip condition wrong and you'll either read a buffer that's still being written (data corruption) or stall waiting for a swap that already happened (needless idle cycles).

**Steps:**
1. Declare two BRAM buffers, A and B, each sized for one tile.
2. Add a 1-bit `active_buf` register: 0 = array reads A / fetcher writes B, 1 = the reverse.
3. Flip `active_buf` only when **both** the current compute is done **and** the *other* buffer's fetch is done — flipping early is the corruption bug.

```verilog
// Ping-pong buffer controller
module double_buffer_ctrl (
    input  wire clk,
    input  wire rst_n,
    input  wire compute_done,   // array finished consuming active_buf
    input  wire fetch_done,     // burst master finished filling the OTHER buf
    output reg  active_buf,     // 0 = compute reads A / fetch writes B
    output reg  swap_pulse      // one-cycle pulse on the cycle we swap
);
    reg fetch_done_latched;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active_buf         <= 1'b0;
            fetch_done_latched <= 1'b0;
            swap_pulse         <= 1'b0;
        end else begin
            swap_pulse <= 1'b0;

            // Latch fetch completion so a fast fetch doesn't get missed
            // while we're still waiting on a slower compute.
            if (fetch_done)
                fetch_done_latched <= 1'b1;

            // Only swap when THIS cycle's compute is done AND the other
            // buffer already finished filling. This is the guard that
            // prevents reading a half-written buffer.
            if (compute_done && fetch_done_latched) begin
                active_buf         <= ~active_buf;
                fetch_done_latched <= 1'b0;
                swap_pulse         <= 1'b1;   // tell the fetcher: start next tile
            end
        end
    end
endmodule
```

> [!success] Done when
> Measured PE utilization (busy cycles / total cycles, read via a hardware counter over AXI-Lite) is **above 80%** on the 64×64 test case.

> [!question]- It's not working
> 1. Is `swap_pulse` actually kicking off the *next* fetch, or does the fetcher sit idle after a swap?
> 2. Are you swapping on `compute_done` alone, without checking `fetch_done_latched` — causing corrupted reads?
> 3. Is `fetch_done` a single-cycle pulse that you're missing because you didn't latch it?
> 4. Does the very first tile load (before any compute has happened) get handled as a special case, or does your state machine assume a buffer is always "already full"?
> 5. Are both buffers actually different BRAM instances/address ranges, not aliases of the same memory?
> 6. Is your utilization counter counting `compute_done`-adjacent idle cycles correctly, or is it counting cycles the array is clocked but not actually doing useful MACs?

> [!danger] Gotcha — 80% is a floor, not a target
> If your fetch tile is consistently slower than a compute tile at your realistic ~1.2 GB/s DDR bandwidth, no amount of buffering logic will save you — double buffering only *hides* a fetch that's shorter than or comparable to the compute it overlaps. If you're stuck under 80%, first check whether your tile size makes the compute-to-fetch ratio physically achievable before you go bug-hunting in the control logic.

---

## 4.3 Write your own AXI4 burst master

> [!abstract] Goal
> Build a minimal AXI4 read-burst master that pulls tiles from DDR through an HP port into BRAM, correctly handling the handshake, the beat counter, and the 4 KB split — and drive it from real C code on the A9, including the cache-coherency fix.

**What's going on:** This is the centerpiece of the whole phase. Xilinx ships an AXI DMA IP block that does this for you, but it costs roughly 2–3K LUTs — about 20% of your *entire* 7007S fabric (14,400 LUTs total) — for a general-purpose engine with far more features than you need. Your hand-rolled version, built for exactly this access pattern (INCR bursts, one direction at a time, fixed beat sizes), comes in around 700 LUTs. On a chip this small, that difference is not a nice-to-have; it's the difference between having room for your PE array and not.

**Steps:**
1. Design a state machine with states for: idle, issue address, wait for address handshake, count and receive data beats, done.
2. Before issuing a burst, run its target range through the 4 KB check from 4.1 — if `ARLEN`+1 beats × bytes-per-beat would cross a 4 KB boundary, shorten this burst to stop at the boundary and queue a second burst for the remainder.
3. Assert `ARVALID` and hold your `AR*` signals stable until `ARREADY` also goes high — never assert `ARVALID` conditionally on already having seen `ARREADY`.
4. Count `R` beats as they arrive (`RVALID && RREADY`), write each into BRAM, and stop on `RLAST`.
5. On the C side: allocate an aligned DMA buffer, flush the cache before letting the PL read, kick off the PL, wait, then invalidate the cache before the A9 trusts anything the PL wrote back.

```verilog
// Minimal AXI4 read-burst master (read channel only: AR + R)
// One HP port, INCR bursts, splits at 4KB boundaries.
module axi_read_burst_master #(
    parameter ADDR_W = 32,
    parameter DATA_W = 64          // HP port width
)(
    input  wire                 clk,
    input  wire                 rst_n,
    // control
    input  wire                 start,        // pulse: begin a burst
    input  wire [ADDR_W-1:0]    start_addr,   // requested start address
    input  wire [8:0]           req_beats,    // requested beat count (<=256)
    output reg                  done,         // pulses when all beats delivered
    // BRAM write side
    output reg  [ADDR_W-1:0]    bram_waddr,
    output reg  [DATA_W-1:0]    bram_wdata,
    output reg                  bram_we,
    // AXI4 AR channel
    output reg  [ADDR_W-1:0]    araddr,
    output reg  [7:0]           arlen,        // beats - 1
    output wire [2:0]           arsize,       // fixed: log2(DATA_W/8)
    output wire [1:0]           arburst,      // fixed: INCR
    output reg                  arvalid,
    input  wire                 arready,
    // AXI4 R channel
    input  wire [DATA_W-1:0]    rdata,
    input  wire                 rlast,
    input  wire                 rvalid,
    output reg                  rready
);
    localparam S_IDLE=0, S_AR=1, S_DATA=2, S_DONE=3;
    reg [1:0] state;
    reg [8:0] beats_left, beats_this_burst;
    reg [ADDR_W-1:0] cur_addr, bram_ptr;

    assign arsize  = 3'b011;   // 8 bytes/beat for a 64-bit HP port
    assign arburst = 2'b01;    // INCR

    // 4KB-boundary split: how many beats fit before the next 4KB line?
    wire [ADDR_W-1:0] bytes_per_beat = (1 << arsize);
    wire [ADDR_W-1:0] bytes_to_4k_line =
        13'h1000 - (cur_addr & 13'hFFF);          // bytes left in this page
    wire [8:0] beats_to_boundary =
        bytes_to_4k_line / bytes_per_beat;         // max legal beats here

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; arvalid <= 0; rready <= 0; done <= 0; bram_we <= 0;
        end else begin
            done <= 0; bram_we <= 0;
            case (state)
                S_IDLE: if (start) begin
                    cur_addr   <= start_addr;
                    beats_left <= req_beats;
                    bram_ptr   <= 0;
                    state      <= S_AR;
                end

                S_AR: begin
                    // Clamp this burst to the 4KB boundary AND the beats left.
                    beats_this_burst <= (beats_left < beats_to_boundary)
                                          ? beats_left : beats_to_boundary;
                    araddr  <= cur_addr;
                    arlen   <= ((beats_left < beats_to_boundary)
                                  ? beats_left : beats_to_boundary) - 1; // ARLEN = beats-1
                    arvalid <= 1'b1;               // VALID asserted unconditionally...
                    if (arvalid && arready) begin  // ...handshake completes here, not before
                        arvalid <= 1'b0;
                        rready  <= 1'b1;
                        state   <= S_DATA;
                    end
                end

                S_DATA: if (rvalid && rready) begin
                    bram_wdata <= rdata;
                    bram_waddr <= bram_ptr;
                    bram_we    <= 1'b1;
                    bram_ptr   <= bram_ptr + 1;
                    beats_left <= beats_left - 1;
                    if (rlast) begin
                        rready   <= 1'b0;
                        cur_addr <= cur_addr + (beats_this_burst * bytes_per_beat);
                        state    <= (beats_left == beats_this_burst) ? S_DONE : S_AR;
                    end
                end

                S_DONE: begin done <= 1'b1; state <= S_IDLE; end
            endcase
        end
    end
endmodule
```

And the C side, on the A9, driving this master and handling coherency:

```c
#include "xil_cache.h"
#include "xparameters.h"

#define TILE_BYTES  (8 * 4 * 1)   // 8x4 INT8 tile

// Aligned, non-cached-friendly buffer: align to a cache line (32B on A9)
u8 tile_buf[TILE_BYTES] __attribute__((aligned(32)));

void fetch_tile_from_ddr(u32 ddr_src_addr) {
    // 1. The A9 may have just written source weights that are still
    //    sitting in L1/L2 and haven't reached DDR yet. Flush them out
    //    so the PL's AXI read sees current data, not stale DDR contents.
    Xil_DCacheFlushRange(ddr_src_addr, TILE_BYTES);

    // 2. Kick the burst master (memory-mapped control reg via AXI-Lite)
    Xil_Out32(BURST_MASTER_BASE + START_REG, ddr_src_addr);
    Xil_Out32(BURST_MASTER_BASE + CTRL_REG,  CTRL_START_BIT);

    // 3. Poll (or wait on interrupt) for completion
    while (!(Xil_In32(BURST_MASTER_BASE + STATUS_REG) & STATUS_DONE_BIT));

    // 4. The PL wrote into a DDR-backed region the A9 will read next.
    //    Invalidate so the A9 doesn't serve stale cached copies of it.
    Xil_DCacheInvalidateRange((u32)tile_buf, TILE_BYTES);
}
```

> [!success] Done when
> You can DMA 1 MB from DDR into BRAM through the burst master and the contents match byte-for-byte, including at least one test case where the source region straddles a 4 KB boundary.

> [!question]- It's not working
> 1. **Cache coherency first**: did you call `Xil_DCacheFlushRange()` before kicking the PL, and `Xil_DCacheInvalidateRange()` after it's done? This is the single most common "my RTL must be broken" report that isn't an RTL bug at all.
> 2. Is `ARLEN` off by one — remember it's beats **minus one**, not beat count.
> 3. Does any single burst actually cross a 4 KB boundary — check with the same arithmetic as `bytes_to_4k_line` above, on paper, for your specific test addresses.
> 4. Is the HP port actually enabled in the Zynq PS configuration in your block design (Vivado's Zynq7 Processing System block has a checkbox per HP port — it's off by default)?
> 5. Is your source address 64-bit aligned to match the HP port's data width, or are you asking for an unaligned access the port can't service?
> 6. Is `RLAST` being honored correctly — does your beat counter stop exactly on `RLAST`, or could it run one beat long/short if `RVALID` deasserts unexpectedly?
> 7. Did you confirm `ARVALID` is asserted unconditionally (not gated on `ARREADY`) — a combinational VALID-waits-on-READY loop will simulate fine in isolation and then deadlock or misbehave once connected to a real interconnect.

> [!danger] Gotcha #1 — cache coherency
> The A9 writes weights to DDR, but they're sitting in L1/L2 cache. Your PL reads **stale DDR**.
>
> Fix: `Xil_DCacheFlushRange()` before kicking the PL, `Xil_DCacheInvalidateRange()` after it finishes. Or mark the buffer non-cacheable via the MMU.
>
> ==This bug looks exactly like an RTL bug and it is not one.== Budget a day for it anyway.

> [!bug] Gotcha #2 — the 4 KB rule
> AXI bursts **cannot cross 4 KB address boundaries**. Your address generator must split them. The interconnect will not do this for you.

---

## 4.4 Measure your roofline

> [!abstract] Goal
> Benchmark your actual sustained DDR bandwidth with a pure-read test, then turn that number — plus your compute throughput — into a [[concepts#Roofline|roofline]] plot that tells you which model sizes are even feasible on this chip.

**What's going on:** Every number so far (~1.2 GB/s DDR, 32 PEs, INT8 MACs) has an implication for what kinds of workloads this NPU can run at full speed versus what kinds will always be memory-bound. A roofline plot makes that implication visible in one picture instead of scattered arithmetic.

**Steps:**
1. Build a pure-read benchmark: DMA a large buffer (several MB) through your 4.3 burst master, counting cycles in hardware.
2. Convert cycles to GB/s using your known clock frequency and bytes moved.
3. Separately, compute your peak achievable compute throughput in GOPS (giga-operations/sec) from the array: 32 PEs × 2 ops/MAC (multiply + add) × clock frequency.
4. Compute **operational intensity** for your actual workload: operations performed per byte fetched from DDR (this depends on how much reuse your tiling/double-buffering scheme achieves — a matrix multiply with good tile reuse has much higher intensity than a pure-read benchmark).
5. Plot: x-axis = operational intensity (ops/byte, log scale), y-axis = achieved GOPS (log scale). Draw a horizontal line at your peak compute GOPS, and a diagonal line with slope = your measured DDR GB/s. The diagonal represents "memory-bound" performance at each intensity; the horizontal represents "compute-bound" performance once you have enough reuse. The point where they cross is the **ridge point** — workloads to the left of it are memory-bound no matter what you do to the array; workloads to the right are compute-bound and the array is your limit, not DDR.

```verilog
// Free-running cycle counter, read over AXI-Lite for benchmarking
module busy_cycle_counter (
    input  wire clk,
    input  wire rst_n,
    input  wire count_enable,   // e.g. "burst in flight" or "PE busy"
    input  wire clear,          // pulse to reset the count (AXI-Lite write)
    output reg  [31:0] count
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)       count <= 32'd0;
        else if (clear)   count <= 32'd0;
        else if (count_enable) count <= count + 32'd1;
    end
endmodule
```

```c
// Read the counter, convert to GB/s
double measure_ddr_bandwidth(u32 bytes_transferred, u32 clk_hz) {
    u32 cycles = Xil_In32(COUNTER_BASE + COUNT_REG);
    double seconds = (double)cycles / (double)clk_hz;
    double gbps = (bytes_transferred / seconds) / 1e9;
    return gbps;   // compare against your ~1.2 GB/s planning number
}
```

> [!success] Done when
> You have a measured sustained DDR bandwidth number (expect **0.8–1.5 GB/s**), a computed peak GOPS for your array, and a roofline plot with a labeled ridge point.

> [!question]- It's not working
> 1. Is the counter actually gated on a signal that means "useful work happening," or is it just a free-running clock counter that always matches wall-clock time regardless of stalls?
> 2. Are you clearing the counter between runs, or accumulating across multiple tests?
> 3. Does your GB/s calculation account for the *actual* bytes moved (including any 4 KB-boundary split overhead), not just the logical buffer size?
> 4. Is your clock frequency constant assumption correct — did you check the actual PL clock you configured in Vivado, not an assumed default?
> 5. Does your measured bandwidth land anywhere near the expected 0.8–1.5 GB/s range — if it's far outside (e.g., under 0.3 GB/s), suspect small burst sizes or 4 KB-splitting eating more of your transfers than expected.

> [!tip] This is the centerpiece of your write-up
> Plot compute intensity vs. achieved GOPS. This number tells you exactly which model sizes are feasible — and it's what makes [[phase-7-end-to-end|the Tier 2 capstone]] interpretable instead of just slow.

---

## Check your understanding

> [!question]- Why is a burst faster than the same number of individual AXI-Lite-style transfers, even though the same total number of bytes moves either way?
> > [!success]- Answer
> > Every individual transfer pays DDR3's full latency penalty (tens of cycles) before its data even starts arriving. A burst pays that latency penalty exactly **once**, for the whole burst, because only the first beat has to wait for the DRAM row to open — every subsequent beat streams out with much less delay. Same total bytes, far less latency overhead per byte.

> [!question]- Your burst master requests 256 beats at 8 bytes/beat starting at address `0x1FF0`. Does this burst need to be split, and if so, how many beats fit before the split?
> > [!success]- Answer
> > `0x1FF0` is 16 bytes below the next 4 KB boundary (`0x2000`). At 8 bytes/beat, that's only 2 beats (16 bytes) before hitting the boundary. So yes — split required. First burst: 2 beats (ARLEN=1). Remaining: 254 beats starting at `0x2000` (which itself may need further splitting if 254×8=2032 bytes still exceeds the next 4KB line — in this case it doesn't, since 2032 < 4096).

> [!question]- Why does double buffering require twice the BRAM footprint, and what would happen if you tried to save BRAM by using a single shared buffer instead?
> > [!success]- Answer
> > You need one buffer the array is actively reading from and a second buffer the burst master is actively writing into, at the same time — that's inherent to "compute while fetching the next tile." A single shared buffer would force fetch and compute to run sequentially (fetch, then compute, then fetch again), which is exactly the stalling behavior double buffering exists to eliminate. You'd be back to the "before" timeline in the double-buffering section.

> [!question]- A tile load moves an 8×4 tile of INT8 data (32 bytes) from DDR. At your ~1.2 GB/s planning bandwidth, roughly how many such tile-loads can you sustain per second, and does that comfortably keep up with an array doing one MAC per PE per cycle at, say, 100 MHz?
> > [!success]- Answer
> > 1.2 GB/s ÷ 32 bytes/tile ≈ 37.5 million tile-loads/sec, i.e. about one tile every ~26.7 ns. At 100 MHz, one array cycle is 10 ns, and a K-tile step (one MAC per PE) consumes one loaded A-tile and one loaded B-tile per cycle in the naive scheme — so a fetch every ~26.7 ns cannot keep up with a compute step every 10 ns from bandwidth alone if you re-fetch both operands every cycle. This is exactly why operand reuse (keeping an A row-block resident across several `nt` iterations, as flagged in the tiling section) matters: without reuse, DDR bandwidth — not the array — is your bottleneck, which is precisely what the roofline plot in 4.4 will show you.

---

## What's next

You've now got tiles moving from DDR to BRAM to the array and back, overlapped, measured, and bounded by a roofline you understand instead of a number you memorized. [[phase-5-vector-unit]] adds a vector unit alongside the MAC array for the non-matmul operations (activations, normalization, elementwise ops) every real model needs between matmuls.

<script src="/tutor.js" defer></script>
