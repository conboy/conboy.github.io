---
title: "Phase 3 — The MAC Engine"
description: "Building an int8 systolic array from a single DSP48E1 up to a 32-PE 8x4 array at 100 MHz."
tags:
  - npu
  - hardware
---

# Phase 3 — The MAC engine

**5 weeks** · Part of [[index|Zero to NPU]] · Prev: [[phase-2-numerics]] · Next: [[phase-4-memory]]

> [!info] What this phase is for
> This is the heart of the NPU, and honestly the most fun phase in the course. Phase 1 proved matrix multiply is almost the entire cost of running the model; Phase 2 got your fixed-point arithmetic bit-exact. Now you build the hardware that does the multiplying — 32 tiny multiply-accumulate circuits wired together so they feed each other instead of fighting over memory bandwidth. By the end you'll have real hardware, in real silicon, running matrix multiplies roughly **5× faster** than the ARM core managed in Phase 1. If "systolic array" sounds like something out of a research paper, don't let it intimidate you — it's one small circuit, copy-pasted into a grid, with a clever wiring pattern. You already know how to build the small circuit; this phase is about the wiring.

---

## Why matrix multiply is the whole problem

Go back to [[phase-1-baseline-llm]] for a second. Every expensive step in a transformer layer — the Q/K/V projections, the attention-weighted sum, the FFN, the final logits projection — is a matrix multiplied by a vector (or another matrix). In plain C, a matrix-times-vector multiply is three nested loops:

```c
for (int i = 0; i < rows; i++) {        // one output element per row
    int32_t acc = 0;
    for (int k = 0; k < cols; k++) {    // walk the shared ("inner") dimension
        acc += (int32_t)x[k] * (int32_t)w[i * cols + k];
    }
    out[i] = acc;
}
```

That inner `acc +=` line is doing one **[[concepts#MAC|MAC]]** — a multiply-accumulate: multiply two numbers, add the result onto a running total. Everything this phase builds is, at bottom, a machine for doing that one line millions of times per second without the ARM core anywhere near it.

Here's the number that makes this worth building custom hardware for at all: for an `n × n` matrix times an `n`-vector, you do roughly `n²` MACs — the work grows with the *square* of the size. But the data you need to hold onto is only `n²` for the weight matrix and `n` for the vector — call it `n²` total too, but critically, **every one of those `n²` weight values gets reused `n` times** (once per output row it contributes to, or once per token if you're generating text one token at a time with the same weights). Work is O(n³) if you count every MAC across every output element and every token; data movement, if you're clever about reuse, doesn't have to grow anywhere near that fast. That gap — lots of arithmetic, comparatively little unique data — is exactly the shape of problem where dedicated hardware wins big over a general-purpose CPU core: the CPU fetches an instruction and pays overhead for every single MAC, while custom hardware can wire the MACs directly together and just let data flow.

The rest of this phase is about building hardware that exploits that reuse instead of re-fetching from memory for every single MAC — because as you're about to see, memory bandwidth, not multiplier speed, is what actually limits you if you get the architecture wrong.

---

## What a systolic array actually is

Let's build this up in stages, starting from the dumbest possible design and fixing its problems one at a time — the reasons *not* to do the naive thing are exactly what a [[concepts#Systolic array|systolic array]] solves.

**Stage 1: one multiplier.** A single multiplier and accumulator, fed one activation and one weight per cycle, doing one MAC per cycle. It works, but it's hopelessly slow — a 64-element dot product needs 64 cycles, using one DSP to do the job of one arithmetic unit at a time, while 59 other DSP48E1 slices sit idle.

**Stage 2: 32 multipliers, and a new problem.** Put down 32 multipliers, one per DSP slice this engine will use (see [[board-specs]] for why 32 and not more). Each can MAC every cycle — but now 32 multipliers each want a *fresh* activation and weight every cycle, 64 values total, from memory. On-chip [[concepts#BRAM|BRAM]] gives you only two read ports. You can't feed 32 multipliers straight from memory without an absurd number of BRAM ports, or a mux crossbar that burns LUTs you don't have (14,400 total). Parallel multipliers just move the bottleneck from "not enough math hardware" to "not enough memory ports."

**Stage 3: the systolic insight.** Instead of every PE fetching its own operands, arrange the [[concepts#PE|PEs]] (processing elements — one per DSP, each a multiplier plus accumulator) in a 2D grid, wired to immediate neighbors only. Each cycle, a PE passes the activation it just used one step to its east neighbor, and the weight it just used one step to its south neighbor. Every value is read out of BRAM exactly **once**, at the edge, then reused by every PE it passes through. Data "pulses" through the grid one PE per cycle, like a heartbeat — hence **systolic** (*systole*, the heart's contraction). You've traded many memory ports for cheap local wiring.

Here's what that looks like for a 2×2 corner of the array — activations enter from the left, weights enter from the top, and each PE forwards what it received to its neighbors on the next cycle:

```
        W[0][0]      W[0][1]
           │            │
           ▼            ▼
      ┌─────────┐  ┌─────────┐
A[0]─▶│ PE[0][0]│─▶│ PE[0][1]│─▶ (out east)
      └────┬────┘  └────┬────┘
           │ W flows     │ W flows
           ▼ down        ▼ down
      ┌─────────┐  ┌─────────┐
A[1]─▶│ PE[1][0]│─▶│ PE[1][1]│─▶ (out east)
      └────┬────┘  └────┬────┘
           ▼             ▼
       (out south)  (out south)
```

Each `PE[r][c]` does the same job every cycle: multiply whatever activation just arrived from the west by whatever weight just arrived from the north, add the result into its own accumulator, and forward both incoming values one step further (activation east, weight south) so the next PE in line can use them one cycle later.

Now, two design philosophies for *where the answer ends up* — worth knowing apart, since people mix them up constantly:

**[[concepts#Weight-stationary|Weight-stationary]]**: a PE's weight register loads once and stays fixed across many different activation vectors. Here, a layer's weights don't change token-to-token, so you load a weight tile once and stream a fresh activation vector through it for every token generated, without reloading — which matters because reloading costs DDR/BRAM bandwidth you don't have to spare (see [[phase-4-memory]]).

**Output-stationary**: each PE owns exactly one output element for the whole computation and never gives it up — partial products stream in from two directions and get summed into that one accumulator in place. Once the reduction finishes, the accumulator register simply *is* the answer, with no separate gather step.

This course's array is output-stationary for the accumulator, combined with weight-stationary reuse over time (the loaded weight sits still across many tokens; only activations and the accumulate-in-place logic move every cycle). That's what lets a tiny 32-PE grid punch above its weight: expensive weight loads happen rarely, cheap activation streaming happens constantly.

Here's exactly what a 2×2 array does, cycle by cycle, multiplying activation matrix `X` by weight matrix `W`, producing `O = X · W` where `O[r][c] = X[r][0]·W[0][c] + X[r][1]·W[1][c]`. (Both operands stream through in a single pass here — get this mechanic right before layering weight-reuse-across-tokens on top.)

| Cycle | PE[0][0] | PE[0][1] | PE[1][0] | PE[1][1] |
|---|---|---|---|---|
| 0 | multiply `X00·W00`, acc = `X00W00` | idle — inputs still in flight | idle — inputs still in flight | idle — inputs still in flight |
| 1 | **done.** holds final `O[0][0] = X00W00+X01W10` | multiply `X00·W01`, acc = `X00W01` | multiply `X10·W00`, acc = `X10W00` | idle — inputs still in flight |
| 2 | holds `O[0][0]` | **done.** holds final `O[0][1] = X00W01+X01W11` | **done.** holds final `O[1][0] = X10W00+X11W10` | multiply `X10·W01`, acc = `X10W01` |
| 3 | holds `O[0][0]` | holds `O[0][1]` | holds `O[1][0]` | **done.** holds final `O[1][1] = X10W01+X11W11` |
| 4 | idle — all outputs ready, safe to read out | idle — all outputs ready | idle — all outputs ready | idle — all outputs ready |
| 5 | ready for next tile's weight/activation load | ready for next tile | ready for next tile | ready for next tile |

Notice the pattern: `PE[r][c]` finishes at cycle `r + c + 1`. `PE[0][0]` finishes first because its operands travel no distance; `PE[1][1]` finishes last because both its inputs ripple through one neighbor first. That travel time is what the next section is about.

---

## Data skewing, and why it's necessary

Look again at the table above: `PE[0][0]` gets its first real MAC at cycle 0, but `PE[1][1]` doesn't get one until cycle 2. Feed every row and column into the array simultaneously on cycle 0, and most of the grid sees garbage for the first cycle or two — values haven't "met" their right partner yet, since they arrive at different PEs after different travel times.

The fix: feed the *input edges* on a staggered schedule instead — this is **skewing**. Row 0's activations enter on cycles 0, 1, 2...; row `r` enters `r` cycles later than row 0. Same idea for weights on columns: column `c` enters `c` cycles later than column 0. Work the arithmetic and it lines up perfectly — the activation and weight for reduction index `k` both arrive at `PE[r][c]` on exactly cycle `k + r + c`. Every value meets its correct partner, every time, with zero muxing inside the PE. The skew lives entirely in *when you inject data at the edges*.

For the 2×2 example above, the skewed feed schedule at the two input edges looks like this:

```
cycle:              0     1     2     3
row 0 activations:  X00   X01   --    --
row 1 activations:  --    X10   X11   --

col 0 weights:      W00   W10   --    --
col 1 weights:      --    W01   W11   --
```

Row 1 and column 1 are both delayed by exactly one cycle relative to row/column 0 — that one-cycle stagger is the entire skew for a 2×2 array. **Before you write the 4×4 version in 3.2, draw this exact 2×2 timing diagram by hand on paper**, including which value sits at which PE on every cycle. It takes ten minutes and it will save you days of confused Verilator waveform-staring later — almost every systolic array bug in this course traces back to a skew that's off by one cycle somewhere.

---

## 3.1 One processing element

> [!abstract] Goal
> Build a single PE in Verilog: int8 × int8 → int32 accumulate, with a weight register, that Vivado maps onto exactly one DSP48E1 slice.

**What's going on:** Every PE in your future 32-wide grid is a copy of this one module. Get it right — correctly registered, correctly typed, mapped to a real [[concepts#DSP slice|DSP slice]] instead of scattered LUTs — and 3.2 becomes "instantiate 16 of these," not "debug 16 separate multiplier bugs."

**Steps:**
1. Declare `in_act` and `in_w` as 8-bit **signed** inputs — int8, not unsigned, or your negative weights and activations from Phase 2's quantization will silently misbehave.
2. Add a `weight_en` control input: when high for one cycle, capture `in_w` into a stationary weight register. This is how you load weights once and reuse them across tokens.
3. Register the activation and weight pass-through outputs (`out_act`, `out_w`) — this is what lets a value "travel" one PE per cycle.
4. Register the accumulator itself, and give it an `acc_clear` input to zero it at the start of a new reduction pass.
5. Make sure **every** register — inputs, outputs, and the multiply result — sits inside a clocked `always @(posedge clk)` block. Nothing here should be combinational.

```verilog
module pe #(
    parameter ACC_WIDTH = 32
)(
    input  wire                        clk,
    input  wire                        rst_n,     // synchronous, active-low
    input  wire                        weight_en, // 1 cycle: latch in_w as the stationary weight
    input  wire                        acc_clear, // 1 cycle: start a fresh accumulation
    input  wire signed [7:0]           in_act,    // activation arriving from the west neighbor
    input  wire signed [7:0]           in_w,      // weight arriving from the north neighbor
    output reg  signed [7:0]           out_act,   // activation, registered, to the east neighbor
    output reg  signed [7:0]           out_w,     // weight, registered, to the south neighbor
    output reg  signed [ACC_WIDTH-1:0] acc        // this PE's stationary output element
);

    reg signed [7:0] weight_reg;  // the weight this PE currently holds "stationary"

    always @(posedge clk) begin
        if (!rst_n) begin
            weight_reg <= 8'sd0;
            out_act    <= 8'sd0;
            out_w      <= 8'sd0;
            acc        <= {ACC_WIDTH{1'b0}};
        end else begin
            // Weight-load phase: capture a new stationary weight on request.
            if (weight_en)
                weight_reg <= in_w;

            // Forward both operands to the neighbors, one cycle later --
            // this delay is exactly what makes data "flow" through the grid.
            out_act <= in_act;
            out_w   <= in_w;

            // The MAC itself: int8 * int8 fits easily in 16 bits, so the
            // 32-bit accumulator has huge headroom before it could overflow.
            if (acc_clear)
                acc <= $signed(in_act) * weight_reg;          // start a new sum
            else
                acc <= acc + ($signed(in_act) * weight_reg);  // accumulate
        end
    end

endmodule
```

**Synthesis check:** after synthesis, run `report_utilization` (Tcl console or GUI) and check the `pe` module's line reads `DSP48E1: 1`, `LUT: ~0` (a handful for reset/control logic is fine; dozens is not).

> [!success] Done when
> Utilization shows **DSP48E1 = 1, LUT ≈ 0**, and a quick testbench confirms `acc` matches `in_act * weight` accumulated correctly over several cycles.

> [!question]- It's not working
> 1. LUT count high, DSP count 0 — a register is missing; DSP48E1 inference needs the multiply's **inputs and output both registered**, no logic in between.
> 2. LUT count high, DSP count also 1 — extra combinational logic (an unregistered mux) is feeding the multiplier; move it behind a register.
> 3. Values overflow or wrap unexpectedly — check `in_act`/`weight_reg` are declared `signed`; unsigned interpretation of a negative int8 gives a huge, wrong product.
> 4. Simulation matches, synthesis differs — look for an accidental latch: every branch of your `always` block must assign every register.
> 5. `acc` never clears — check `acc_clear` and `weight_en` aren't accidentally tied to the same signal; they do different jobs.

> [!bug] Gotcha
> If LUTs > 0, the tool didn't infer the DSP. Check that your multiply is **registered on both input and output** — DSP inference needs the pipeline registers to map directly into the slice's own input/output flops.

---

## 3.2 A 4×4 output-stationary systolic array

> [!abstract] Goal
> Wire 16 copies of the PE from 3.1 into a 4×4 grid, get activations flowing west→east and weights flowing north→south, and validate the whole array against a golden C reference in Verilator.

**What's going on:** This is where the single PE becomes an *array*, and where the skewing you hand-traced above becomes real hardware. You're instantiating the same `pe` module 16 times in a grid and wiring each instance's east/south outputs to its neighbor's west/north inputs, using a `generate` loop instead of hand-writing 16 near-identical instantiations.

**Steps:**
1. Declare boundary inputs: one activation per row (west edge), one weight per column (north edge).
2. Use nested `generate for` loops over rows and columns to instantiate 16 `pe` modules.
3. Wire each PE's `out_act`/`out_w` to its east/south neighbor's `in_act`/`in_w`; tie the array's outer edges to the boundary inputs.
4. Expose all 16 `acc` outputs for the testbench.
5. Feed the array on the **skewed** schedule you hand-drew — never a simultaneous load.
6. Build a Verilator testbench: random 4×4 int8 matrices, skewed the way hardware expects, compared against a plain C reference.

```verilog
module systolic_array #(
    parameter ROWS = 4,
    parameter COLS = 4
)(
    input  wire clk,
    input  wire rst_n,
    input  wire weight_en,
    input  wire acc_clear,
    input  wire signed [7:0]  act_in [0:ROWS-1],       // west edge, one value per row
    input  wire signed [7:0]  w_in   [0:COLS-1],       // north edge, one value per column
    output wire signed [31:0] acc_out[0:ROWS-1][0:COLS-1]
);

    // Internal nets between neighboring PEs. One extra column/row so the
    // array's own east/south edges have somewhere to dangle harmlessly.
    wire signed [7:0] act_net [0:ROWS-1][0:COLS];
    wire signed [7:0] w_net   [0:ROWS][0:COLS-1];

    genvar r, c;
    generate
        for (r = 0; r < ROWS; r = r + 1) begin : ROW
            assign act_net[r][0] = act_in[r];  // tie west edge to boundary input
            for (c = 0; c < COLS; c = c + 1) begin : COL
                if (r == 0)
                    assign w_net[0][c] = w_in[c];  // tie north edge to boundary input

                pe pe_inst (
                    .clk       (clk),
                    .rst_n     (rst_n),
                    .weight_en (weight_en),
                    .acc_clear (acc_clear),
                    .in_act    (act_net[r][c]),
                    .in_w      (w_net[r][c]),
                    .out_act   (act_net[r][c+1]),   // feeds east neighbor next cycle
                    .out_w     (w_net[r+1][c]),     // feeds south neighbor next cycle
                    .acc       (acc_out[r][c])
                );
            end
        end
    endgenerate

endmodule
```

*(Unpacked-array ports, for readability; flatten into wide buses if your flow needs strict Verilog-2001 — the `generate` logic is unchanged either way.)*

**Verilator testbench sketch:**

```cpp
// 1. Build a random 4x4 int8 activation matrix X and weight matrix W.
// 2. Reference in plain C:
//    for (r) for (c) { int32_t acc=0; for (k) acc += X[r][k]*W[k][c]; ref[r][c]=acc; }
// 3. Skew both inputs as derived by hand: row r starts at cycle r, column c at cycle c.
// 4. Toggle the clock (dut->eval() each half-cycle), driving skewed inputs.
// 5. After row+col+1 cycles (plus reset/load overhead), compare every
//    dut->acc_out[r][c] bit-exactly against ref[r][c]. Repeat for hundreds of matrices.
```

> [!success] Done when
> A Verilator testbench multiplies hundreds of random 4×4 matrices through the array and every accumulator matches the C reference **exactly**, with no tolerance/fuzz allowed — int8 MACs have no rounding error to hide behind.

> [!question]- It's not working
> 1. Every output wrong by the same fixed amount — you likely forgot `acc_clear` before a new matrix; you're accumulating on top of the last one's leftovers.
> 2. Only the far corner (`PE[ROWS-1][COLS-1]`) is wrong — classic skew-by-one bug; re-check your hand-drawn 2×2 diagram against what the testbench actually drives.
> 3. Every PE off by one cycle's worth of data — your testbench's skew schedule and the hardware's registered pass-through delay disagree; count registers, not intentions.
> 4. Some PEs pass, others don't, no pattern — check the `generate` loop's boundary tie-offs (`if (r == 0)` for weights, `act_net[r][0]` for row edges); an off-by-one there leaves a wire floating or double-driven.
> 5. Works once, fails on the second matrix — you're not re-asserting `acc_clear` (and `weight_en`, if weights changed) at the start of every new pass.

> [!bug] Gotcha — data skewing
> Row `i` of the activation matrix must enter `i` cycles later than row 0, so operands meet at the right PE on the right cycle.
>
> **Draw the timing diagram by hand for a 2×2 array before you write the 4×4.**

---

## 3.3 Scale to 8×4 (32 PEs)

> [!abstract] Goal
> Widen the validated 4×4 design to 8 columns × 4 rows — 32 PEs, 32 DSPs — and close timing at 100 MHz on real silicon.

**What's going on:** The *logic* doesn't change — a correct 4×4 array generalizes to 8×4 by changing two parameters. What changes is you now have to prove it on the actual part, at the actual clock speed, not just in simulation. 32 PEs means more routing distance and DSP-to-DSP wiring on a speed-grade **-1** (slowest available) chip — exactly the kind of change that turns a design that "worked in RTL simulation" into one that fails **[[concepts#Timing closure|timing closure]]** on hardware.

**Steps:**
1. Instantiate `systolic_array` with `ROWS=4, COLS=8` instead of `4, 4` — the entire structural change, thanks to 3.2's `generate` loop.
2. Re-run the Verilator testbench against 8×4-shaped golden data; confirm bit-exact results.
3. Run Vivado synthesis + implementation at 100 MHz and open `report_timing_summary`.
4. Check **WNS** (Worst Negative Slack) on the setup path. Positive WNS: every path finished with room to spare, you're fine. Negative WNS: your slowest path took longer than one clock period — not guaranteed to work on real silicon, even if simulation looked perfect.
5. If WNS is negative, find the failing path (Vivado names it) — usually a long unregistered pass-through wire, or fan-out on `weight_en`/`acc_clear` hitting all 32 PEs at once. Add a pipeline register, or buffer the fan-out, and re-run.

> [!success] Done when
> `report_timing_summary` shows **WNS ≥ 0** at 100 MHz, and the testbench still passes bit-exactly on the wider 8×4 shape.

> [!question]- It's not working
> 1. WNS negative on a control signal, not a data path — `weight_en`/`acc_clear` is fanning out combinationally to all 32 PEs; register it once and broadcast the registered version.
> 2. WNS negative on the accumulator path — the multiply-then-add may not be fully absorbed into the DSP48E1's own pipeline registers; revisit 3.1's registering rules.
> 3. DSP count isn't 32 — a stale 4×4 instantiation is probably still in the build alongside the new 8×4 one.
> 4. Simulation passes but hardware output is wrong — classic sign of a timing violation simulation can't see (zero-delay model); go check WNS before assuming a logic bug.
> 5. Timing passes easily with huge slack — good, but don't chase 150 MHz yet; get Phases 4-7 working at 100 MHz first, revisit speed in [[phase-8-optimization]].

---

## 3.4 Two int8 MACs per DSP

> [!warning] Stretch goal — budget for this not working
> This is optional. If it doesn't pan out, you fall back to 1 MAC per DSP (which is exactly what 3.1–3.3 already give you) and lose nothing you were counting on.

**What's going on:** A DSP48E1 slice has a 25×18-bit multiplier and 48-bit accumulator — wider than the 8×8 multiply and 32-bit accumulate one int8 MAC actually needs. Xilinx's app note WP486 packs **two** weight values into the wide 25-bit input port, multiplies both against a shared activation in one multiply, then splits the wide product back into two results — two MACs out of hardware that normally gives you one.

**The packing arithmetic:** pack weights `w0`, `w1` as `w_packed = w0 + (w1 << 18)`, multiply by shared activation `a`. The product lands as `a*w0` in the low bits and `a*w1` shifted up 18 bits — *if* there's enough headroom ("guard bits") that `a*w0` never grows large enough to corrupt `a*w1`'s field, and vice versa for sign extension.

> [!danger] The DSP48E1 vs DSP48E2 risk
> WP486 targets **DSP48E2** (UltraScale, **27×18** multiplier) — two extra guard-bits versus this board's **DSP48E1** (**25×18**). With only 25 bits split between two signed 8-bit products, the guard band is tight and may not hold cleanly for **signed** int8 without extra correction logic that eats back the savings. Treat this as a research exercise: verify bit-exactly across the full signed range (both operands negative included), and if it doesn't hold up, fall back to 3.1's one-MAC-per-DSP design without regret — that's what [[board-specs]] already budgets around. Pure upside, never a dependency.

> [!success] Done when
> You can demonstrate 64 independent, bit-exact MAC results driven from only 32 DSP48E1 slices — **or** you've documented exactly where and why the guard bits broke down for signed int8, and reverted to the 1-MAC-per-DSP design from 3.1–3.3.

---

## Check your understanding

> [!question]- Why does a systolic array need neighbor-to-neighbor wiring at all — why not just add more BRAM ports instead?
> > [!success]- Answer
> > BRAM gives you only a small, fixed number of ports (two, typically), and 32 multipliers would need 64 fresh operands per cycle if each fetched straight from memory. You can't add "enough" ports, and a read crossbar burns LUTs you don't have (14,400 total). A systolic array reads each value out of memory exactly once, at the edge, and lets it ripple through every PE that needs it via cheap local wiring — scarce memory ports traded for cheap wiring.

> [!question]- Trace `X[1][0]` (row 1, column 0 of the activation matrix) through the 2×2 array from the cycle-by-cycle table above. Which PEs does it visit, and on which cycles?
> > [!success]- Answer
> > Injected at the west edge of row 1 on cycle 1 (row 1 is skewed one cycle behind row 0). `PE[1][0]` consumes it that same cycle, against `W[0][0]`. `PE[1][0]` forwards a registered copy east, arriving at `PE[1][1]` on cycle 2, multiplied against `W[0][1]`. Column 1 is the last column, so its journey ends there — two PEs, two cycles, two MACs.

> [!question]- What's the actual difference between "weight-stationary" and "output-stationary," and which one does this course's array use?
> > [!success]- Answer
> > Weight-stationary: a weight loads once and stays fixed while many activation vectors stream through over time — valuable since a layer's weights are reused every generated token. Output-stationary: a PE owns one output accumulator for the whole computation and never passes it elsewhere; partial products stream in and sum in place. This course combines both — output-stationary accumulation, plus weight-stationary reuse across tokens.

> [!question]- Why does 3.1 insist on registering both the inputs and the output of the multiply, instead of just the output?
> > [!success]- Answer
> > DSP48E1 inference looks for a multiply whose inputs *and* output are both driven by registers with no logic in between — that maps directly onto the DSP slice's own pipeline flops. Register only the output, and Vivado usually can't prove the multiply is clean and isolated, so it falls back to LUTs instead. That's exactly the failure the 3.1 utilization check (DSP48E1=1, LUT≈0) catches.

---

## What's next

You now have 32 DSPs wired into a systolic array that multiplies matrices roughly 5× faster than the ARM core, validated bit-exactly against your Phase 1 golden model. [[phase-4-memory]] tackles the problem this phase deliberately deferred: an array this fast is only as fast as the data reaching it, and right now it has no pipe to DDR3 at all.

<script src="/tutor.js" defer></script>
