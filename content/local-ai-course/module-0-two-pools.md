---
title: "Module 0 — The Slow Pool"
description: "Measuring DDR5 bandwidth honestly with STREAM Triad, recording the 4-DIMM tax, and sweeping thread count for a number every later module reuses."
tags:
  - local-ai
  - memory-bandwidth
  - ddr5
  - benchmarking
---

# Module 0 — The Slow Pool

**5 hours** · Part of [[index|Two Pools]] · Next: [[module-1-toolchain]]

> [!info] What this module is for
> This course is built on one organizing insight: your workstation has two memory pools, roughly 11x apart in bandwidth, and almost every configuration decision from here forward is really a decision about which bytes live in which pool and how often you pay to move them between the two. The equation is simple to write and easy to underestimate:
>
> `t_token(n) = V(n)/BW_vram + D(n)/BW_ddr`
>
> — the time to produce token n is the VRAM-resident bytes divided by VRAM bandwidth, plus the DDR-resident bytes divided by DDR bandwidth. Notice what that equation does *not* contain: a single "GPU speed" number. The roofline for any given workload is a function of where its bytes are placed, not a property of the hardware alone. Before you can reason about that placement, you need real, measured numbers for both pools — not the numbers on the spec sheet. This module gets you the first one: the slow, large pool, DDR5 system memory. It needs no GPU and no ROCm, so it’s the one piece of this course you can do before you’ve even installed Linux, and it’s worth running on both Windows and your eventual Linux install for a free comparison of platform overhead. Module 1 does the same honesty check for the fast, small pool — VRAM — and only once you have both measured numbers does the 11x from the spec sheets become a real ratio you can trust.

---

## 0.1 Record what the BIOS thinks you have

> [!abstract] Goal
> Before running a single benchmark, write down what the platform actually configured your memory to — not what you paid for.

**What’s going on:** The 9950X’s official memory support is population-dependent: with 2 DIMMs installed (single- or dual-rank) it supports DDR5-5600, but with all 4 DIMM slots populated — which is what "128GB as 4x32GB" means — the supported speed drops to DDR5-3600. This is not a bug or a bad BIOS setting; it’s a signal-integrity tax that comes from routing four ranks of DIMMs on a two-channel bus instead of two. It’s often called the "4-DIMM tax" informally, and it applies before you’ve touched a single BIOS toggle. DDR5 dual-channel is 128 bits wide, i.e. 16 bytes per transfer, so bandwidth in GB/s is simply 16 x (MT/s): DDR5-3600 gives a theoretical **57.6 GB/s**, versus 89.6 GB/s if you could actually run at DDR5-5600. You are not going to see 89.6 GB/s with four DIMMs installed, and you should know that going in rather than discovering it after an hour of confused benchmarking.

**Steps:**
1. On Windows or Linux, check the BIOS/UEFI memory page directly first and note the configured speed, not just the DIMM-rated XMP/EXPO speed printed on the sticker.
2. On Linux, get the authoritative OS-level view:
   ```bash
   sudo dmidecode --type 17
   ```
   Look for `Size`, `Rank`, `Speed` (the DIMM’s rated maximum), and **`Configured Memory Speed`** (what the platform is actually running) for each of the four slots.
3. Cross-check with:
   ```bash
   sudo lshw -C memory
   ```
4. Record all four DIMMs’ `Configured Memory Speed` in one table — they should agree with each other; if they don’t, you have a mixed or misconfigured population.

> [!success] Done when
> You have a written table of all 4 DIMM slots — size, rank, rated speed, and configured speed — pulled from `dmidecode`, not assumed from the box the RAM came in.

> [!question]- It’s not working
> 1. Does `dmidecode` need root? Re-run with `sudo` if fields print as empty or "Not Specified."
> 2. Are all 4 slots actually populated — check `Size` isn’t `No Module Installed` for any slot.
> 3. Does `Configured Memory Speed` disagree between DIMMs? That’s a real finding, not a tool error — note it and investigate BIOS memory training logs before benchmarking.
> 4. On Windows, `wmic memorychip get` or CPU-Z’s SPD tab give the equivalent view if you’re doing the pre-Linux-install pass.

---

## 0.2 Build and run STREAM Triad

> [!abstract] Goal
> Compile McCalpin’s STREAM benchmark from source and get a first Triad number, at a single thread count, as a sanity baseline.

**What’s going on:** STREAM measures sustained memory bandwidth using four simple vector kernels — Copy, Scale, Add, and Triad — but **Triad is the one that matters** and is conventionally reported as the headline number: `a[i] = b[i] + q*c[i]`. It touches two input arrays and writes one output array (2 reads + 1 write per element), it does one multiply-add of scalar work per element (nowhere near enough arithmetic to hide the memory traffic), and critically the arrays are sized so **nothing fits in cache** — every element is a fresh trip to DRAM. That combination is what makes Triad a fair measurement of sustained bandwidth rather than a cache-residency benchmark in disguise. The 9950X’s two CCDs carry a combined 64MB of L3; STREAM’s own sizing rule is that the total working set (sum of all arrays involved) should be at least 4x the sum of all cache levels in the system, so with 64MB of L3 alone that already argues for a working set comfortably over 256MB — in practice most people size it much larger to be safe and to get a runtime long enough to average over.

**Steps:**
1. Download the source directly from the reference implementation:
   ```bash
   wget https://www.cs.virginia.edu/stream/FTP/Code/stream.c
   ```
2. Compile with OpenMP enabled and an array size large enough to blow past 64MB of L3 by a wide margin — 200 million double-precision elements per array (three arrays, ~1.6GB each, ~4.8GB total) is a safe, generous choice on a 128GB machine:
   ```bash
   gcc -O3 -march=native -fopenmp \
       -DSTREAM_ARRAY_SIZE=200000000 -DNTIMES=20 \
       stream.c -o stream_triad
   ```
3. Run it once, single-threaded, as a sanity baseline:
   ```bash
   OMP_NUM_THREADS=1 ./stream_triad
   ```
4. Look for the results table STREAM prints — a `Function / Best Rate MB/s / Avg time / Min time / Max time` block with a row each for Copy, Scale, Add, and Triad. Record the **Triad** "Best Rate" line; that’s your number, in MB/s (divide by 1000 for GB/s).

> [!success] Done when
> You have a single-threaded Triad number in GB/s, printed by your own build of STREAM, from an array size you’ve confirmed exceeds your L3 by a wide margin.

> [!question]- It’s not working
> 1. Did the compile actually succeed, or is `stream.c` still an unmodified download with a tiny default array size (the stock source often ships sized for 1990s cache sizes)? Re-check your `-DSTREAM_ARRAY_SIZE`.
> 2. Is `-fopenmp` actually in your compile line — without it, `OMP_NUM_THREADS` does nothing and you’re silently single-threaded regardless of what you set.
> 3. Does the binary crash or hang at a huge array size? You may be swapping — confirm `free -h` shows enough headroom before the 3-array footprint you chose.
> 4. Are your numbers suspiciously close to a cache bandwidth figure (many tens of GB/s higher than DDR5-3600’s 57.6 theoretical ceiling)? Your array size is too small and some of it is staying resident in L3 — increase `STREAM_ARRAY_SIZE` and rebuild.

> [!bug] Gotcha
> `mbw` (a common quick memory-bandwidth tool) is *not* an adequate substitute for STREAM here. It typically runs single-threaded, uses a much simpler access pattern, and doesn’t give you the compiler-vectorized, cache-bypassing, multi-kernel comparison STREAM does — numbers from `mbw` and STREAM are not directly comparable, and `mbw` alone will not give you the thread-scaling curve the next section needs.

## 0.3 Sweep thread count

> [!abstract] Goal
> Run Triad across a range of thread counts and find the point where added threads stop buying more bandwidth — the memory-bandwidth saturation point.

**What’s going on:** A single CPU thread cannot saturate a memory controller’s available bandwidth on its own — it’s limited by how many outstanding memory requests one core can keep in flight. Adding threads increases the number of concurrent outstanding requests, so bandwidth climbs as you add threads, but only up to the point where the DRAM channels themselves are saturated; beyond that point, more threads buy you nothing (or make it slightly worse from contention). The 9950X has 16 physical cores and 32 threads via SMT, but for a bandwidth-bound test — as opposed to a compute-bound one — the elbow of that curve is very unlikely to be all the way out at 32; it should show up well before you use every SMT thread. Finding that elbow *is the deliverable of this section*: it becomes the `-t` value you’ll pass to `llama-bench` and `llama.cpp` binaries in later modules, because those tools’ own `--threads -1` default (`hardware_concurrency()`) double-counts SMT threads and is not a substitute for actually measuring your own machine.

**Steps:**
1. Sweep a spread of thread counts, pinning threads to physical placement so the OS scheduler isn’t randomly bouncing them between cores mid-run:
   ```bash
   for t in 1 2 4 6 8 10 12 14 16 20 24 28 32; do
     echo "=== threads=$t ==="
     OMP_NUM_THREADS=$t OMP_PLACES=cores OMP_PROC_BIND=close \
       ./stream_triad | grep -A1 '^Triad'
   done
   ```
2. Record Triad GB/s at every thread count into a table or a quick plot (thread count on x, GB/s on y).
3. Identify the elbow — the last thread count that gave a meaningful GB/s increase over the previous one. A reasonable definition: the first thread count where the next step up gains less than ~2-3% more bandwidth.
4. Note whether 4x32GB actually shows meaningfully lower total bandwidth than a hypothetical 2-DIMM population would, and whether the elbow shifts noticeably between JEDEC and EXPO once you do 0.4 below — this data point matters again after that A/B.

> [!success] Done when
> You have a thread-count-vs-GB/s table or plot, and you can state a specific thread count where the curve visibly flattens — this is the number you carry forward as your `-t` argument in later modules.

> [!question]- It’s not working
> 1. Is the curve flat from thread=1 onward with no rise at all? Check `OMP_PLACES`/`OMP_PROC_BIND` actually took effect — an unpinned run can show noisy, non-monotonic results.
> 2. Is bandwidth still climbing at 32 threads with no elbow in sight? That would be unusual for a 2-channel DDR5 system — double check you’re measuring Triad specifically and not accidentally grepping a different kernel’s line.
> 3. Are results wildly inconsistent run-to-run at the same thread count? Background load (browser, indexing, thermal throttling) can add noise — close everything non-essential and rerun the noisy thread counts.

## 0.4 A/B JEDEC vs EXPO

> [!abstract] Goal
> Test whether AMD EXPO actually posts with all four DIMM slots populated, and if it does, measure what it buys you over JEDEC defaults.

**What’s going on:** EXPO (AMD’s DDR5 overclocking profile standard, analogous to Intel XMP) is validated by DIMM vendors against specific population and rank scenarios — and a 4-DIMM population is the hardest case to validate for, precisely because of the same signal-integrity issue behind the 4-DIMM tax in 0.1. It is entirely possible your board simply will not boot at the EXPO-rated speed with all four slots full, and that outcome is itself useful data, not a failure of this exercise.

**Steps:**
1. Boot into BIOS, note the current profile (should be JEDEC default, i.e. DDR5-3600 per 0.1), and record a baseline Triad number at your module-0.3 elbow thread count if you haven’t already.
2. Enable the EXPO profile in BIOS. Save and reboot.
3. If the system fails to POST: that’s your answer for this A/B. Reset BIOS to defaults (or fall back to a manually-set safer speed, e.g. DDR5-4800, if your board allows an intermediate step) and record that it did not post at full EXPO with 4 DIMMs installed.
4. If it does POST: boot into the OS, re-run `dmidecode --type 17` to confirm the new `Configured Memory Speed`, then rerun STREAM Triad at your elbow thread count from 0.3.
5. Compare the two Triad numbers directly.

> [!success] Done when
> You have a definite answer — "EXPO posted at [speed] and Triad measured [X] GB/s" or "EXPO failed to POST with 4 DIMMs, reverted to JEDEC DDR5-3600" — plus the JEDEC baseline Triad number either way.

> [!question]- It’s not working
> 1. Black screen after enabling EXPO? Most boards let you clear CMOS (jumper or button) to force a revert to defaults — this is expected behavior for an out-of-spec population, not a broken board.
> 2. EXPO posts but the system is unstable under load later (not during this short test)? Note it anyway — instability under sustained load is a separate, real finding from "did it POST."
> 3. Is `Configured Memory Speed` still showing the old value after a profile change? Some BIOS/OS combinations need a full power cycle (not just a warm reboot) for SPD-level changes to be reflected in `dmidecode`.

## 0.5 Compute your efficiency

> [!abstract] Goal
> Turn your measured numbers into the two figures every later module will reference: sustained DDR5 bandwidth, and your DDR efficiency ratio.

**What’s going on:** "60-90% of theoretical" is the normal, expected range for sustained DDR bandwidth on a real system — you lose some to refresh cycles, bank conflicts, ECC or scrambling overhead if enabled, and imperfect access-pattern overlap even in an idealized benchmark like STREAM. A number in that band is not a sign anything is wrong; a number well outside it (much lower) usually points back to an array-sizing or thread-pinning mistake from 0.2-0.3, and a number well above the 57.6 GB/s theoretical DDR5-3600 ceiling means you’re still measuring cache, not DRAM.

**Steps:**
1. Take your best (JEDEC, elbow-thread-count) Triad number from 0.3 in GB/s. Call it `measured`.
2. Compute `η_ddr = measured / 57.6` (using the DDR5-3600 theoretical ceiling, since that’s what a 4-DIMM 9950X is officially rated for).
3. If EXPO posted in 0.4, compute the same ratio against whatever theoretical ceiling that configured speed implies (`16 x MT/s`), and compare both the raw GB/s and the efficiency ratio side by side.
4. Write down all three final numbers together: measured sustained DDR5 GB/s, η_ddr, and the elbow thread count from 0.3.

> [!success] Done when
> You have measured sustained DDR5 bandwidth in GB/s, η_ddr (measured / 57.6), and the thread count at which bandwidth visibly saturates — all three written down as three numbers you can quote, not estimate.

> [!bug] Gotcha
> Don’t skip recomputing η_ddr against the *actual* configured speed if EXPO changed it. 57.6 GB/s is the JEDEC DDR5-3600 ceiling specifically — if you’re running EXPO at a different rate, your theoretical denominator changes too, and comparing a JEDEC-relative efficiency against an EXPO-relative one will make you think something moved when it didn’t.

---

## Check your understanding

> [!question]- 1. Why is STREAM Triad specifically the right kernel to headline, rather than Copy or Add?
> > [!success]- Answer
> > Triad (`a[i] = b[i] + q*c[i]`) touches two reads and one write per element with essentially no reused data and only a trivial amount of arithmetic (one multiply-add) per element loaded — the arithmetic is nowhere near enough to hide or overlap the memory traffic, and there’s no cache reuse across iterations because the arrays are sized far past your last-level cache. That combination makes it a close-to-pure measurement of sustained memory bandwidth, whereas Copy does less per-element work and Add does one fewer operand — Triad is the closest to "how fast can this system move bytes" once compute is out of the way.

> [!question]- 2. Your Triad number comes back around 90 GB/s on a system you confirmed is 4-DIMM JEDEC DDR5-3600. What’s the most likely explanation?
> > [!success]- Answer
> > 90 GB/s is well above the 57.6 GB/s theoretical ceiling for DDR5-3600, so this isn’t real sustained DRAM bandwidth — it’s very likely that `STREAM_ARRAY_SIZE` is too small and some or all of the working set is still fitting in the 9950X’s 64MB of combined L3 cache, so you’re measuring cache bandwidth, not DRAM bandwidth. Increase the array size until the combined footprint is well past 256MB (4x the total cache) and rerun.

> [!question]- 3. Why does `--numa distribute/isolate/numactl` matter for STREAM on a dual-socket server but do essentially nothing useful on the 9950X?
> > [!success]- Answer
> > Those NUMA modes exist to control which memory-controller domain (NUMA node) a thread’s memory gets allocated from, which only matters when a system has more than one such domain — typically multi-socket servers, or some multi-die designs that expose multiple NUMA nodes. A single 9950X, with its one integrated memory controller servicing both CCDs, presents as one NUMA node to the OS, so NUMA placement flags are effectively a no-op here — worth knowing so you don’t cargo-cult server tuning advice onto a single-socket desktop.

## What’s next

You now have two honest numbers about the slow, large memory pool: sustained DDR5 GB/s and the thread count where that bandwidth saturates. Neither number means anything on its own yet — they become useful once you have the equivalent numbers for the fast, small pool. [[module-1-toolchain]] gets Linux and ROCm 10 installed, builds llama.cpp against your RX 9070 XT twice (HIP and Vulkan), and measures VRAM bandwidth the same way you just measured DDR — so you can finally compute the *real* ratio between the two pools, instead of the one on the spec sheet.

<script src="/tutor.js" defer></script>
