---
title: "Phase 0 — Toolchain and Verilog Refresh"
description: "Getting Vivado, the UART, a first AXI-Lite peripheral, and the ILA working on a Zynq-7007S."
tags:
  - npu
  - hardware
---

# Phase 0 — Recommission the toolchain and your Verilog

**2 weeks** · Part of [[index|Zero to NPU]] · Next: [[phase-1-baseline-llm]]

> [!info] What this phase is for
> Before you can build a neural network accelerator, you need to prove three boring but load-bearing things: the toolchain works, your Verilog muscle memory still fires, and you can move a byte of data between the ARM CPU and the FPGA fabric and back. That's it. There is nothing in this phase about matrix multiplies, quantization, or anything resembling a neural network — that starts in Phase 1 and beyond. Think of Phase 0 as clearing your throat before you sing. If you skip it or rush it, every later phase gets harder to debug, because you won't be sure whether a bug is "my new NPU logic is wrong" or "I never actually got the plumbing working."

## Before you start: the mental model

Here's the thing that will make the rest of this course click: **the chip on your Blackboard is not one computer, it's two.** Xilinx (now AMD) calls this a Zynq, and yours is specifically the XC7Z007S-1CLG400C. "Zynq" is the product family name — it doesn't stand for anything you need to memorize. What matters is the split inside it.

The first half is the **PS**, or [[concepts#PS|processing system]] — a real, ordinary ARM Cortex-A9 CPU running at 666 MHz. On your board it's a single core (some bigger Zynq parts have two; yours has one). This is a normal computer. It boots Linux or bare-metal C code, has a UART for a serial terminal, talks to the 512 MB of DDR3 memory, and behaves exactly like any embedded CPU you've used before. When people say "the software side" or "the ARM side," they mean the PS.

The second half is the **PL**, or [[concepts#PL|programmable logic]] — this is the actual FPGA fabric. It is not a CPU at all. It's a sea of reconfigurable digital logic: 14,400 [[concepts#LUT|LUTs]] (Look-Up Tables — tiny pieces of memory configured to act as arbitrary logic gates), 28,800 [[concepts#Flip-Flop|flip-flops]] (single bits of storage that latch a value on a clock edge — this is what makes something "registered" instead of purely combinational), roughly 60 [[concepts#DSP slice|DSP48E1 slices]] (hardened multiply-accumulate blocks — far more efficient than building a multiplier out of LUTs), and 225 KB of [[concepts#BRAM|BRAM]] (Block RAM — small, fast, on-chip memory you can shape into whatever width and depth you want). When you write Verilog and synthesize it, you are literally deciding how these LUTs, flip-flops, DSPs, and BRAMs get wired together. Nothing runs "on" the PL the way a program runs on a CPU — the PL *becomes* the circuit you described.

These two halves talk to each other over a bus protocol called [[concepts#AXI|AXI]] (Advanced eXtensible Interface). You'll hear "AXI-Lite" (a simple, low-throughput register-style interface, good for control and status) and "AXI-Stream" (a higher-throughput, no-address, just-a-stream-of-data interface). Almost everything you build for the rest of this course will be described as "the PS does X, and the PL does Y, and they meet at an AXI interface." That sentence is the skeleton of the entire NPU project. If you're ever lost in a later phase, come back to that sentence and ask which side you're actually working on.

Analogy, if it helps: the PS is a project manager who can read email, make phone calls, and write memos (general-purpose, flexible, but not built to lift heavy things fast). The PL is a warehouse full of custom-built machinery that you configure yourself — blazing fast at exactly the job you wired it for, and totally inert until you tell it what to be. Phase 0 is you proving the project manager and the warehouse can send each other a fax.

## 0.1 Toolchain up

> [!abstract] Goal
> Get Vivado and Vitis installed, build the simplest possible PS-only design, and see text come out of the board over serial.

**What's going on:** Vivado is Xilinx/AMD's tool for designing and building PL logic (and for wiring up the PS). Vitis is the software-side IDE — it's where you write and compile the C code that runs on the ARM core. You need both, and you need them to agree on which exact chip you have, because the free WebPACK license and the board's pin definitions are specific to the XC7Z007S part. This exercise doesn't touch the PL at all yet — it's purely "can the PS boot and talk to me," which is the lowest possible bar and the right place to start.

**Steps:**
1. Download and install **Vivado WebPACK** (free tier — it supports the 7007S, you do not need a paid license) and **Vitis**, matching versions from the same release train.
2. Install the **Blackboard's board definition files** from Real Digital *before* you create a project. These tell Vivado the DDR3 timing, the MIO (multiplexed I/O) pinout, and other board-specific details automatically.
3. In Vivado, create a new RTL project targeting the XC7Z007S-1CLG400C part (or select the "Blackboard" board preset if the board files installed correctly — this fills in the part number for you).
4. Open the **Block Design** view (IP Integrator) and add a **ZYNQ7 Processing System** IP block. Run "Run Block Automation" so Vivado wires up the default clocks and DDR/MIO connections from the board files.
5. Validate the design, create an HDL wrapper, and run Synthesis → Implementation → Generate Bitstream.
6. Export the hardware (include the bitstream) to Vitis, create a new Vitis platform project from it, then create an application project using the **Hello World** template.
7. Connect a USB cable to the board's UART port, open a serial terminal (e.g., Tera Term, PuTTY, or the terminal built into Vitis) at **115200 baud**, 8N1.
8. Program the FPGA, then run the application from Vitis.

> [!success] Done when
> "Hello World" (or equivalent text) appears in your serial terminal, printed by code running on the ARM core.

> [!question]- Nothing happening? (check these in order)
> 1. Is the board actually powered on, and is the power LED lit?
> 2. Did you install the Blackboard board definition files *before* creating the project? If not, delete the project and redo step 3 — a project created without them will have the wrong DDR3/MIO configuration baked in.
> 3. Is the serial terminal pointed at the right COM port, and is it set to 115200 baud, 8 data bits, no parity, 1 stop bit?
> 4. Did you actually program the FPGA with the bitstream (not just build it)? Generating a bitstream doesn't load it onto the board by itself.
> 5. Does the Vitis platform project reference the *exact* exported hardware file (the `.xsa`), or a stale one from an earlier build?
> 6. Try a different USB cable — some are charge-only and carry no data lines.

> [!bug] Gotcha
> The Blackboard needs its **board definition files** installed, or you'll end up hand-configuring the DDR3 controller and MIO pinout yourself. Get them from Real Digital *first*, before you create your first project.

## 0.2 Verilog limbering

> [!abstract] Goal
> Write a small, correctly pipelined multiply-accumulate (MAC) unit in Verilog, and prove it's correct against a C reference model using Verilator.

**What's going on:** A MAC unit computes `accumulator = accumulator + (a * b)` — it's the single most common operation in any accelerator, NPU included, because matrix multiplication is just a huge pile of multiply-accumulates. "Pipelined" means we break the operation into clocked stages (register the inputs, multiply, register the output) rather than trying to do it all in one combinational blob — this is what lets the real chip run at a useful clock frequency instead of being limited by one giant slow path. [[concepts#Verilator|Verilator]] is an open-source tool that compiles your Verilog into a C++ model you can simulate and test on your own laptop, with no FPGA involved — this is how you catch bugs in seconds instead of waiting minutes for a real hardware build every time.

### The valid/ready handshake

Before the code: every block you build from here on will pass data around using a two-signal handshake called **valid/ready**. `valid` means "the sender has data on the bus right now." `ready` means "the receiver is able to accept it right now." Data only actually transfers on a clock edge where **both** are high at the same time.

```
clk     : _/‾\_/‾\_/‾\_/‾\_/‾\_/‾\_
valid   : ___/‾‾‾‾‾‾‾\___________     <- sender has data from cycle 2
ready   : _/‾‾‾\_______/‾‾‾‾‾‾‾‾\_    <- receiver stalls in cycle 3-4
                ^
          transfer happens here (both high, same cycle)
```

The rule that will save you months of pain: **`valid` must never depend combinationally on `ready`.** In plain terms — the sender is not allowed to look at whether the receiver is ready *this instant* and use that to decide, in the same clock cycle with zero delay, whether to assert `valid`. If you wire it that way, you create a combinational loop between the two blocks (each one's output depends instantaneously on the other's), which either won't synthesize, will synthesize into something that behaves unpredictably, or will flat-out fail timing. `valid` should be driven from a register that only looks at `ready` from the *previous* cycle. This exact mistake is the single most common bug new FPGA engineers introduce, and it's why the AXI-Stream and AXI-Lite specs are so strict about it.

**Steps:**
1. Write the pipelined MAC module below in a `.v` file.
2. Write a C++ reference model (just plain multiply-accumulate in a `uint32_t`, no hardware concepts) and a Verilator testbench that drives random `a`/`b` values in, applies the handshake, and compares the PL's output against the reference every cycle.
3. Run at least 100,000 random test vectors.

```verilog
// pipelined_mac.v — 16x16 -> 32-bit multiply-accumulate, 2-stage pipeline
module pipelined_mac (
    input  wire        clk,
    input  wire        rst_n,      // active-low synchronous reset

    input  wire [15:0] a_in,
    input  wire [15:0] b_in,
    input  wire        in_valid,   // upstream: "a_in/b_in are good this cycle"
    output wire        in_ready,   // this module can accept new inputs

    output reg  [31:0] acc_out,    // running accumulator
    output reg         out_valid,  // downstream: "acc_out is good this cycle"
    input  wire        out_ready   // downstream can accept the result
);

    // Stage 1: register the inputs (this is what makes it "pipelined")
    reg [15:0] a_reg, b_reg;
    reg        stage1_valid;

    // We only accept new inputs when stage 2 isn't stalled behind us.
    // Note: this is driven by a REGISTER (stage1_valid/out_valid state),
    // not combinationally from out_ready — see the handshake rule above.
    assign in_ready = out_ready || !out_valid;

    always @(posedge clk) begin
        if (!rst_n) begin
            a_reg        <= 16'd0;
            b_reg        <= 16'd0;
            stage1_valid <= 1'b0;
        end else if (in_ready) begin
            a_reg        <= a_in;
            b_reg        <= b_in;
            stage1_valid <= in_valid;
        end
    end

    // Stage 2: multiply the registered inputs, then accumulate into acc_out.
    // The multiply itself maps onto one of the chip's DSP48E1 slices.
    wire [31:0] product = a_reg * b_reg;

    always @(posedge clk) begin
        if (!rst_n) begin
            acc_out   <= 32'd0;
            out_valid <= 1'b0;
        end else if (in_ready) begin
            if (stage1_valid)
                acc_out <= acc_out + product;   // accumulate
            out_valid <= stage1_valid;
        end
    end

endmodule
```

```cpp
// tb_pipelined_mac.cpp — Verilator testbench skeleton
#include "Vpipelined_mac.h"
#include "verilated.h"
#include <cstdint>
#include <cstdlib>

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vpipelined_mac* dut = new Vpipelined_mac;

    uint32_t reference_acc = 0;   // plain C++ model, no hardware concepts
    dut->rst_n = 0;

    for (int cycle = 0; cycle < 200000; cycle++) {
        if (cycle == 2) dut->rst_n = 1;   // release reset after a couple cycles

        // Drive random inputs, randomly withhold in_valid / out_ready
        // to exercise the handshake, not just the happy path.
        dut->a_in     = rand() & 0xFFFF;
        dut->b_in     = rand() & 0xFFFF;
        dut->in_valid = (rand() % 4) != 0;   // mostly valid, sometimes not
        dut->out_ready = (rand() % 4) != 0;  // mostly ready, sometimes not

        dut->clk = 0; dut->eval();
        dut->clk = 1; dut->eval();

        if (dut->rst_n && dut->in_valid && dut->in_ready)
            reference_acc += (uint32_t)dut->a_in * (uint32_t)dut->b_in;

        if (dut->out_valid && dut->out_ready) {
            if (dut->acc_out != reference_acc) {
                printf("MISMATCH at cycle %d: dut=%u ref=%u\n",
                       cycle, dut->acc_out, reference_acc);
                return 1;
            }
        }
    }
    printf("PASS\n");
    return 0;
}
```

> [!success] Done when
> 100,000+ random vectors run through the Verilator testbench with zero mismatches, including cycles where you randomly withhold `in_valid` or `out_ready` (not just the all-signals-high happy path).

> [!question]- Nothing happening? (check these in order)
> 1. Does Verilator find your module at all — did you pass the right top-level module name to `verilator --cc`?
> 2. Are you calling `dut->eval()` after *every* clock edge, including the falling edge? Skipping this is the #1 cause of "signals never seem to update."
> 3. Did you actually toggle `rst_n` correctly — active-low means `0` asserts reset, `1` releases it? Easy to get backwards.
> 4. Is your reference model only accumulating on cycles where the handshake actually completed (`in_valid && in_ready`), not on every cycle?
> 5. If results are *close* but drift over time, suspect the accumulator width — 32 bits from two 16-bit inputs can still overflow if you run long enough; make sure your reference model uses the same width behavior.
> 6. Did you remember to only sample `acc_out` when `out_valid && out_ready` are both true, matching the handshake rule?

> [!bug] Gotcha
> Get the handshake right *now*. AXI-Stream semantics — **`valid` cannot depend combinationally on `ready`** — will bite you for the next 8 months otherwise. It's tempting to "just wire it up" the easy way when your testbench is simple; that habit will not survive contact with a real AXI interconnect.

## 0.3 Your first AXI-Lite peripheral

> [!abstract] Goal
> Build a minimal AXI-Lite slave with a small register file, so the PS can write a register that changes PL hardware state, and read back a register the PL wrote.

**What's going on:** This is the most important plumbing in the whole course, because it's the pattern every future NPU control interface reuses: the ARM core (PS) writes configuration and reads status through a handful of memory-mapped registers, exposed by a small piece of PL logic called an [[concepts#AXI-Lite|AXI-Lite]] slave. "Memory-mapped" means that from the PS's point of view, these registers just look like normal memory addresses — you `Xil_Out32()` to write one and `Xil_In32()` to read one, exactly like poking any other address. Underneath, your Verilog is implementing the AXI-Lite protocol's write and read channels and using the incoming address to pick which register gets touched.

**Steps:**
1. Build a custom AXI-Lite slave with (at least) 16 32-bit registers — start from the skeleton below, don't try to write the whole AXI-Lite state machine from scratch on your own the first time.
2. In your Vivado block design, connect this custom IP to the Zynq PS's AXI master (via the AXI interconnect / Address Editor). Note the base address Vivado assigns it — it will likely default to something like `0x43C00000`, but **you must check the actual Address Editor tab in your own project**; do not assume this value.
3. Wire register 0's bits out to the board's user LEDs.
4. Wire a free-running counter inside the PL into register 1 (read-only from the PS's perspective).
5. From a Vitis application, write a pattern to register 0 and confirm the LEDs change. Read register 1 twice, a short delay apart, and confirm the value increased.

```verilog
// axilite_regfile.v — minimal AXI-Lite slave skeleton, 16 x 32-bit registers
// This handles the write address/data/response and read address/data
// channels using small state machines. Comments mark the non-obvious parts.
module axilite_regfile (
    input  wire        clk,
    input  wire        rst_n,

    // ---- Write address channel ----
    input  wire [31:0] awaddr,
    input  wire        awvalid,
    output reg         awready,

    // ---- Write data channel ----
    input  wire [31:0] wdata,
    input  wire        wvalid,
    output reg         wready,

    // ---- Write response channel ----
    output reg  [1:0]  bresp,
    output reg         bvalid,
    input  wire        bready,

    // ---- Read address channel ----
    input  wire [31:0] araddr,
    input  wire        arvalid,
    output reg         arready,

    // ---- Read data channel ----
    output reg  [31:0] rdata,
    output reg  [1:0]  rresp,
    output reg         rvalid,
    input  wire        rready,

    // ---- Register file exposed to the rest of the PL ----
    output reg  [31:0] regs [0:15],   // regs[0] drives the LEDs, for example
    input  wire [31:0] reg1_pl_write  // e.g. a free-running counter -> regs[1]
);

    // AXI-Lite addresses are BYTE addresses. With 16 registers of 4 bytes
    // each, the register index is bits [5:2] of the address — NOT [3:0].
    // Getting this shift wrong is the classic off-by-four bug (see Gotcha).
    wire [3:0] waddr_idx = awaddr[5:2];
    wire [3:0] raddr_idx = araddr[5:2];

    // --- Write channel: simple two-phase handshake ---
    // We accept an address+data pair only when both AWVALID and WVALID
    // are asserted; awready/wready are asserted together for one cycle.
    always @(posedge clk) begin
        if (!rst_n) begin
            awready <= 1'b0;
            wready  <= 1'b0;
            bvalid  <= 1'b0;
            bresp   <= 2'b00;
        end else begin
            if (awvalid && wvalid && !awready) begin
                awready       <= 1'b1;
                wready        <= 1'b1;
                regs[waddr_idx] <= wdata;   // register 1 is overwritten by PL
                                            // below if reg1_pl_write is used
                bvalid  <= 1'b1;
                bresp   <= 2'b00;           // OKAY
            end else begin
                awready <= 1'b0;
                wready  <= 1'b0;
            end

            if (bvalid && bready)
                bvalid <= 1'b0;             // response accepted, clear it
        end
    end

    // --- Read channel ---
    always @(posedge clk) begin
        if (!rst_n) begin
            arready <= 1'b0;
            rvalid  <= 1'b0;
            rresp   <= 2'b00;
        end else begin
            if (arvalid && !arready) begin
                arready <= 1'b1;
                rdata   <= regs[raddr_idx];
                rvalid  <= 1'b1;
                rresp   <= 2'b00;          // OKAY
            end else begin
                arready <= 1'b0;
            end

            if (rvalid && rready)
                rvalid <= 1'b0;            // data accepted, clear it
        end
    end

    // regs[1] is continuously driven by PL-side logic (e.g. a counter),
    // so a PS write to register 1 would be overwritten next cycle —
    // decide deliberately which registers are PS-writable vs PL-writable.
    always @(posedge clk) begin
        if (!rst_n)
            regs[1] <= 32'd0;
        else
            regs[1] <= reg1_pl_write;
    end

endmodule
```

> [!success] Done when
> A call like `Xil_Out32(BASE + 0x00, 0xDEADBEEF)` from the A9 visibly changes the LEDs, and `Xil_In32(BASE + 0x04)` returns a value that increases between two calls a moment apart.

> [!question]- Nothing happening? (check these in order)
> 1. Is `BASE` actually the address Vivado's Address Editor assigned your IP — did you re-check it after any change to the block design (it can shift)?
> 2. Register index math: are you using `addr[5:2]`, not `addr[3:0]`? This is *the* classic bug here — see the Gotcha.
> 3. Did you connect the AXI-Lite slave's clock and reset to the *same* clock/reset domain the Zynq PS AXI master uses?
> 4. Is your custom IP actually included and reset-deasserted in the exported bitstream — did you re-run Generate Bitstream after your last RTL edit, or is Vitis still running against a stale `.xsa`?
> 5. If reads return all zeros or all ones, check that `rdata` is actually driven combinationally-or-registered from `regs[raddr_idx]` and not left floating in some path.
> 6. If writes to register 0 don't stick, confirm register 0 isn't being continuously overwritten by some other always-block (like register 1 is, deliberately, in the skeleton above).

> [!bug] Gotcha
> AXI-Lite addresses are **byte** addresses; your register index is `addr[5:2]`. Off-by-four here costs people entire evenings.

## 0.4 Visibility

> [!abstract] Goal
> Instrument your AXI-Lite bus with an ILA so you can watch the handshake happen in real hardware, not just in simulation.

**What's going on:** An [[concepts#ILA|ILA]] (Integrated Logic Analyzer) is a small piece of debug logic that Vivado inserts *into your own bitstream*, alongside your design. It watches signals you choose, waits for a trigger condition you specify (like "AWVALID goes high"), and then captures a window of cycles around that trigger — which you view afterward in Vivado as a waveform, over the same USB/JTAG cable you used to program the board. It's the hardware equivalent of a `printf` debugger, except it doesn't slow anything down and it shows you every signal at every clock edge, not just what you thought to print. You need this because simulation (Verilator) only tells you your *design* is correct — it says nothing about whether the real board, real clocks, and real DDR3 are behaving the way you assumed.

**Steps:**
1. In your block design, add a **System ILA** IP core (or use the "Mark Debug" feature on individual nets and let Vivado insert one automatically) attached to the AXI-Lite signals between the Zynq PS and your register file — at minimum AWVALID, AWREADY, WVALID, WREADY, ARVALID, ARREADY, and the address/data buses.
2. Set the trigger condition to a write to a specific address (e.g., AWVALID asserted with AWADDR equal to your register 0's address).
3. Re-run Synthesis → Implementation → Generate Bitstream (the ILA is now part of your design, so it must be rebuilt in).
4. Program the board, open the **Hardware Manager** in Vivado, and arm the trigger.
5. From your Vitis application (or a debugger), perform a write to that address and watch the trigger fire.

> [!success] Done when
> You can see the AWVALID/WVALID handshake — both signals rising together, then AWREADY/WREADY responding — laid out as a waveform in the Vivado Hardware Manager, captured from the real board.

> [!question]- Nothing happening? (check these in order)
> 1. Is the JTAG/USB cable connected and does Vivado's Hardware Manager actually detect the board (open target, refresh device)?
> 2. Did you rebuild the bitstream *after* adding the ILA? An ILA added to the block design but not re-synthesized won't be in the bitstream running on the board.
> 3. Is the trigger actually armed in the Hardware Manager before you run the software that performs the write? A capture that happens before you arm it is simply missed.
> 4. Are you probing signals in the right clock domain — does the ILA's clock input match the clock of the signals you're watching?
> 5. If the waveform looks static/idle forever, double check the software actually executed the write (add a UART print immediately after the `Xil_Out32` call to confirm it ran).
> 6. Is your trigger condition actually reachable — e.g., did you trigger on the exact byte address your software is really writing to, matching what you found in 0.3?

> [!tip] Why this matters
> When the NPU hangs at 3 a.m. in month 6, the ILA is how you find out which side of a handshake dropped the ball — the PS thinking it sent data the PL never saw, or vice versa. Learn to reach for it now, on a design simple enough that you already know the right answer.

## Check your understanding

> [!question]- 1. Why can't `valid` depend combinationally on `ready`?
> > [!success]- Answer
> > Because if each signal is computed instantaneously from the other in the same cycle, you create a combinational loop between the sender and receiver — a cycle with no register to break it. This either fails to synthesize, produces logic that isn't guaranteed to settle to a stable value, or fails timing closure. `valid` must come from a register that reacts to `ready` no sooner than the following cycle.

> [!question]- 2. You wrote `regs[addr[3:0]]` instead of `regs[addr[5:2]]` in your AXI-Lite slave. What happens, concretely?
> > [!success]- Answer
> > AXI-Lite addresses are byte addresses, and each register is 4 bytes wide. Using `addr[3:0]` treats every individual byte offset as a distinct register index, so writes meant for register 1 (byte address 0x04) actually land on whatever index `4` maps to under the wrong shift — in practice your reads and writes hit the wrong registers, or wrap around after only 4 "real" registers instead of 16.

> [!question]- 3. Your Verilator testbench passes with `in_valid` and `out_ready` always held high, but fails once you randomize them. What class of bug does that usually point to?
> > [!success]- Answer
> > A handshake bug — logic that only works when there's no backpressure. Common causes: `valid` being asserted for data that isn't actually held stable while the receiver is stalled, or an accumulator/counter advancing on every clock edge instead of only on cycles where the handshake actually completed (`valid && ready`).

## What's next

With the toolchain proven, your Verilog reflexes warmed up, and a working AXI-Lite path between the PS and PL, you have the two things every later phase depends on. Head to [[phase-1-baseline-llm]] to get a baseline language model running purely in software on the ARM core, before any of it moves into the fabric.

<script src="/tutor.js" defer></script>
