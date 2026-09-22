---
title: "Phase 8 — Optimization and Write-Up"
description: "4-bit weights, timing closure, power measurement, and publishing the roofline."
tags:
  - npu
  - hardware
---

# Phase 8 — Make it good

**4 weeks** · Part of [[index|Zero to NPU]] · Prev: [[phase-7-end-to-end]]

> [!info] What this phase is for
> Phase 7 proved the design works: a real transformer, generating real text, on hardware you built yourself. That's tempting to call done. This phase is where you stop asking "does it work" and start asking "is it *good*" — smaller, faster, and honestly measured, the way a design gets defended rather than demoed once and abandoned. Concretely: finish the 4-bit weight path properly instead of leaving it half-optimized, learn to read a timing report well enough to push your clock rate instead of guessing at RTL changes, measure power instead of assuming dedicated hardware is efficient just because it's dedicated, and write the whole thing up as something you could hand to another engineer. None of this is as glamorous as Phase 7's capstones, but it's the difference between a project that worked once and one you actually understand and can defend under questioning.

---

## 8.1 4-bit weights everywhere

Phase 7.2 introduced the idea that two 4-bit weights pack into one byte, and treated the unpacking step as a black box so you could focus on getting the whole system generating text. This section is where you build that box properly and understand why it lives exactly where it does.

**The packing format.** Fix one convention and use it everywhere, in both the software that writes the packed weight file and the hardware that reads it: the **high nibble** of each byte holds the first weight, the **low nibble** the second. A 4-bit signed weight has a much smaller range than int8 (typically -8 to 7) — half the storage, coarser values. If software's packing order and hardware's unpacking order ever disagree, every weight is systematically wrong in a consistent, non-crashing way — the "nibble order swapped" bug flagged in [[phase-7-end-to-end|Phase 7.2's troubleshooting]]. Worth re-checking here now that you're building the real thing.

**Where unpacking happens.** A PE can't multiply half a byte directly — something has to split the byte into two nibbles and sign-extend each back out to a full-width signed integer (matching the width your Phase 3 PEs already expect). *Where* that splitting happens has a real resource-cost answer, not just a style preference.

```verilog
// Weight unpacker, placed once in the weight-load path (between BRAM and
// the array), not inside each PE. Splits packed bytes into sign-extended
// values the array's existing int8-style PEs can consume unchanged.
module weight_unpack #(
    parameter N_BYTES = 4,   // bytes arriving per load-path cycle
    parameter OUT_W   = 8    // width the PE datapath already expects
)(
    input  [N_BYTES*8-1:0]        packed_in,   // raw packed bytes from BRAM
    output signed [OUT_W-1:0]     unpacked_hi [0:N_BYTES-1], // 1st weight/byte
    output signed [OUT_W-1:0]     unpacked_lo [0:N_BYTES-1]  // 2nd weight/byte
);
    genvar i;
    generate
        for (i = 0; i < N_BYTES; i = i + 1) begin : UNPACK
            wire [7:0] this_byte = packed_in[i*8 +: 8];
            wire signed [3:0] hi_nib = this_byte[7:4];   // first weight
            wire signed [3:0] lo_nib = this_byte[3:0];   // second weight

            // Sign-extend 4 bits -> OUT_W bits by replicating the sign bit.
            // This is the entire "unpack" operation: a slice and a replicate,
            // done twice per byte.
            assign unpacked_hi[i] = {{(OUT_W-4){hi_nib[3]}}, hi_nib};
            assign unpacked_lo[i] = {{(OUT_W-4){lo_nib[3]}}, lo_nib};
        end
    endgenerate
endmodule
```

That's the whole operation: a bit-slice and a sign-bit replicate, maybe a few dozen LUTs for a handful of bytes per cycle. Now consider putting this same logic inside every PE instead, so each of your 32 PEs independently unpacks its own weight nibble — functionally identical, but now **32 copies** of the same shift/mask/sign-extend logic. On a 14,400-LUT part ([[board-specs]]), "a few dozen LUTs, once" becomes "a few hundred to a thousand LUTs, spent 32 times over" for the same result. Do it once, centrally, in the load path, and every PE stays exactly as designed in Phase 3 — it never knows the weight it received started life packed two-to-a-byte.

**The accuracy check to run.** 4-bit is a much more aggressive cut than the int8 quantization from [[phase-1-baseline-llm|Phase 1.3]], which held perplexity delta under 1%. Don't expect that bound here — 16 representable values instead of 255 is a real precision loss, and visible quality drop is expected. Check that the model still produces *recognizable, coherent* short stories, not gibberish: rerun Phase 1.3's fp32-vs-quantized methodology, now comparing 4-bit against both fp32 and int8 so you have both deltas documented. A noticeably larger but still-coherent delta is a pass; actual gibberish means check packing order and requantization shift before blaming 4-bit itself.

---

## 8.2 Timing closure

If you've never closed timing on real hardware before, this section is worth reading slowly — it's a skill, not a checklist, and it's one you'll use for the rest of your career every time you push a design's clock rate.

**What a timing path even is.** Every signal in your design travels from one flip-flop (the "launch" flop, which updates the signal on a clock edge) through some combinational logic (LUTs, wires, routing) to another flip-flop (the "capture" flop, which samples the signal on the *next* clock edge). That launch-to-capture route is a **timing path**. Your clock period is a promise: "every path's signal will settle and be stable well before the next clock edge arrives to sample it." Timing closure is the process of finding every path that breaks that promise and fixing it.

**Setup slack, precisely.** For any path, setup slack = (time available before the next clock edge) − (combinational + routing delay) − (the capture flop's own setup-time requirement, a small fixed overhead every flip-flop needs before it can reliably sample). **Positive** slack means the signal arrives with time to spare. **Negative** slack means the signal is still changing when the capture flop samples it — risking a metastable (undefined, neither-0-nor-1) value. This is a real correctness bug waiting to happen on silicon, not a "probably fine" situation, even if it happens to simulate correctly.

**Reading `report_timing_summary`.** Vivado's timing summary reports two numbers you need to know cold:

```
    WNS(ns)      TNS(ns)  TNS Failing Endpoints  TNS Total Endpoints
    -------      -------  ---------------------  --------------------
     -1.243     -187.560                    212                  4096
```

- **WNS — Worst Negative Slack.** The single worst path's slack. `-1.243` means your worst-offending path arrives 1.243 ns *too late*. WNS ≥ 0 means every path meets timing — your closure target.
- **TNS — Total Negative Slack.** The *sum* of slack across every failing path, not just the worst one. `-187.560` across 212 endpoints tells you this isn't one freak outlier — it's spread widely, which usually means one *structural* cause (a shared, overly-deep combinational block feeding many destinations) rather than 212 unrelated bugs. Fix the shared cause and most of TNS clears at once.

A small WNS with a huge failing-endpoint count is usually the easier fix (one shared bottleneck); a large WNS on very few endpoints is a genuinely deep logic chain that needs restructuring.

**The standard fixes, in the order to actually try them:**

1. **Pipeline the critical path.** Find the combinational chain that's too long (the timing report shows the exact logic cells on the worst path) and insert a register partway through it, splitting one long delay into two shorter ones across two clock cycles. The most reliable fix, and the one to try first — but it adds a cycle of latency somewhere, so double-check anything downstream (like the sequencer's timing assumptions) still holds.
2. **Retiming.** Let the tool move register boundaries for you (`phys_opt_design -retime`, or synthesis-time retiming) to balance delay per stage without rewriting RTL. Worth trying when pipeline registers already exist but are unevenly spaced.
3. **Physical optimization (`phys_opt_design`).** A post-placement pass that restructures high-fanout logic and rebalances routing without touching RTL semantics. Cheap to try before manual floorplanning.
4. **Pblock floorplanning.** Constrain a critical block (the PE array, or the burst-master interconnect) to a defined physical region of the die. On a small, dense part like the 7007S, the placer can scatter related logic across the chip to satisfy other constraints, adding routing delay unrelated to logic depth — a Pblock keeps that cluster physically together. The most invasive fix; reach for it last, once you've confirmed the delay is about distance, not depth.

> [!info] Speed grade -1 — be honest with yourself about the ceiling
> This is the slowest speed grade Xilinx makes for this part. 100 MHz is a comfortable starting target; 150 MHz is an ambitious, real, and reachable stretch goal once you've closed timing once already. But don't be discouraged if the array tops out closer to **125 MHz** even after working through every fix above — that's a legitimate, honest result on this silicon, not a sign you did something wrong. Document whatever frequency you actually close timing at, and why, rather than chasing a number the part may simply not support.

---

## 8.3 Power

Everything so far has optimized for speed and size. This section optimizes for — and measures — the number that's actually the strongest argument for building dedicated inference hardware at all: **energy per token generated**, not just tokens per second.

**How to actually measure it.** The simplest reliable method is an inline USB power meter (reports volts, amps, watts) between the wall adapter and the Blackboard's power input — this captures *total board power*: PS, PL, DDR3, everything, under real load. Onboard current-sense rails or a PMBus monitor are a finer-grained alternative if your board exposes them, but total-board-at-the-wall is legitimate and simpler. Either way, measure **during active generation**, not idle — run long enough that the reading settles.

**What to compare against.** You already have the ingredients:
- Your Tier 1 capstone's tok/s (from [[phase-7-end-to-end|Phase 7.2]]) and the board's measured power under that workload.
- The plain ARM Cortex-A9 baseline from [[phase-1-baseline-llm|Phase 1.1]] — same model, no PL acceleration at all — and its power draw running the same generation loop.
- If you have access to a laptop or desktop, the same or a comparably-sized model running on its CPU (and GPU, if available), measured the same way — at the wall, under load.

Compute the one number that makes all of these comparable regardless of how different the hardware is:

```
tok/s/W = (tokens generated per second) / (watts measured under that load)
```

**Why this is the real argument for NPUs.** A discrete GPU will almost certainly out-run your FPGA in raw tok/s — that's not in question. What a GPU isn't built to do is run a 260K-parameter model *efficiently*: it's an enormous general-purpose processor carrying capability this tiny model never touches, and every idle transistor still draws power. Your NPU was built to do exactly what this model needs and nothing else — no unused execution units, no oversized memory hierarchy. That specificity is what dedicated inference silicon trades on, and tok/s/W is the number that makes the trade visible instead of asserted. Expect your number to look very good — and now you'll have measured it yourself, instead of citing someone else's whitepaper claim.

---

## 8.4 Write it up

An FPGA sitting on your desk, or a repo full of Verilog, is worth much less to anyone else — including future-you in a job interview — than a clear writeup that shows what you built, what it cost, and what you learned. This section is about producing that document.

**What a good writeup contains, concretely:**

- **A roofline plot** — the one from [[phase-4-memory|§4.4]], with **both capstones' operating points plotted on it**: Tier 1 sitting compute-bound to the right of the ridge point (all on-chip, no DDR traffic), Tier 2 sitting memory-bound to the left of it (streamed weights, bandwidth-capped). Seeing both on one picture is the single most convincing artifact in the entire writeup — it makes the memory-wall lesson from [[phase-7-end-to-end|Phase 7.3]] visually undeniable instead of a paragraph of prose.
- **A resource utilization table**, measured against the budget in [[board-specs]] — LUTs, DSPs, and BRAM blocks actually consumed versus what the part offers, broken down by block (array, vector unit, burst master, sequencer, KV cache) the same way [[board-specs]] already organizes it.
- **The quantization accuracy delta** — both the int8 result from [[phase-1-baseline-llm|Phase 1.3]] and the 4-bit result from §8.1 above, ideally as a small table: fp32 baseline, int8 delta, 4-bit delta, with example output text at each level so a reader can judge "coherent" for themselves rather than trusting a single perplexity number.
- **tok/s/W**, from §8.3, compared against whatever baselines you measured — ARM-only, and CPU/GPU if you got there.
- **Both capstone results side by side** — Tier 1's tok/s and zero DDR traffic, Tier 2's tok/s and its measured bandwidth ceiling, presented as one clean comparison, not two separate afterthoughts.

**A suggested outline**, if you want a structure to start from rather than a blank page:

1. **Introduction** — the board's real constraints (14,400 LUTs, 60 DSPs, 225 KB BRAM, ~1.2 GB/s DDR3), and why that makes this an interesting NPU design problem rather than a trivial one.
2. **Architecture overview** — one block diagram: PE array, vector unit, memory pipeline, command sequencer, KV cache, referencing the phases that built each one.
3. **Quantization** — the accuracy-vs-footprint story, int8 then 4-bit.
4. **Resource utilization** — the table against [[board-specs]]'s budget.
5. **Performance** — both capstones' tok/s, and the roofline plot with both plotted.
6. **Power** — tok/s/W and what it's compared against.
7. **Conclusion** — what you'd do differently, and a pointer to where a bigger design would go next (see below).

> [!important] This is a portfolio piece
> "I designed an int8/4-bit NPU that runs a real transformer language model on a $149 FPGA with 14,400 LUTs, and here's the roofline analysis showing exactly why a 60×-larger model is 15× slower" is a genuinely strong, specific, defensible thing to be able to say in an interview — and now you have the plot, the table, and the measured numbers to back every word of it.

---

## Check your understanding

> [!question]- Why does unpacking 4-bit weights once in the load path cost so much less than unpacking inside each PE, even though the logic itself (a slice and a sign-extend) is nearly free?
> > [!success]- Answer
> > The logic itself is small — a bit-slice and a sign-bit replicate, maybe a few dozen LUTs. The cost comes from *replication*: doing it once in the shared load path means every PE downstream is unchanged from its Phase 3 int8 design. Doing it inside each of the 32 PEs means building 32 independent copies of that same small circuit — on a 14,400-LUT part, a "few dozen LUTs, once" becomes "a few hundred to a thousand LUTs, spent 32 times over" for a functionally identical result.

> [!question]- Your timing report shows WNS = -0.05 ns but TNS = -40 ns across 500 failing endpoints. What does this combination suggest about where the problem lives, versus a report showing WNS = -3.2 ns on just 2 failing endpoints?
> > [!success]- Answer
> > A tiny WNS spread across hundreds of endpoints usually points to one shared structural bottleneck — a piece of logic with high fanout feeding many destinations, or a systemic routing congestion issue — because it's unlikely that 500 unrelated paths would all independently fail by almost the same razor-thin margin. Fixing that one shared cause (retiming, or restructuring the high-fanout driver) tends to clear most of the failures at once. A large WNS on only 2 endpoints instead points to a specific, genuinely deep logic chain on those exact two paths that needs direct attention — pipelining or restructuring that particular path — rather than a general systemic fix.

> [!question]- Why is tok/s/W a more meaningful number than raw tok/s when the point you're trying to make is "dedicated inference hardware is a good idea"?
> > [!success]- Answer
> > Raw tok/s alone favors whichever chip has the most raw silicon, and a big GPU will win that comparison against a $149 FPGA every time — that comparison proves nothing about efficiency. tok/s/W normalizes for the cost of achieving that throughput, which is exactly what "dedicated" is supposed to buy you: hardware built to do only the operations a tiny model needs, with no unused general-purpose capability burning power in the background. A GPU running a 260K-parameter model is enormously over-provisioned for that job; your NPU isn't. tok/s/W is the number that makes that specificity-to-efficiency argument visible instead of asserted.

---

## Where to go next

This course targeted the smallest Zynq part on purpose, to force every design decision to be deliberate. If you want to keep going, a few directions worth exploring, roughly in order of how much new hardware they'd require:

- **A bigger board** — an XC7Z020 or an UltraScale+ part gives you dramatically more BRAM and DSPs; the interesting exercise is figuring out how much of this course's design scales up unchanged versus needs rethinking at a larger size.
- **Flash attention** — a well-known technique for computing attention without ever materializing the full score matrix in memory, reducing memory traffic for longer contexts. Directly relevant once you outgrow the 128-token cap this course settled on.
- **INT4 activations, not just weights** — this course took weights to 4-bit; pushing activations down too roughly doubles your effective compute density, at a further accuracy cost worth measuring the same way §8.1 measured the weight-only cut.
- **Speculative decoding** — pair a small, fast model with a larger, slower one; the small model guesses several tokens ahead and the large model verifies them in parallel, trading extra compute for fewer sequential round-trips. An interesting fit for exactly the kind of latency-bound loop this NPU runs.
- **A second core** — Zynq's PS is dual-core even though this course only ever used one; parallelizing PS-side work (or running two independent generation streams) is a natural next question once the PL side is this well understood.

Keep it public. A working repo, a clear writeup, and a board on your desk that generates real text is a rare and specific thing to be able to show someone.

<script src="/tutor.js" defer></script>
