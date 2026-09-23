---
title: "Module 8 — The Quantization Ladder as a Budget Problem"
description: "Re-deriving the optimal n_cpu_moe at each quant level and showing the resulting speedup is superlinear against naive byte-scaling — checked against the Aider Polyglot instrument, with one deliberate quality-collapse endpoint."
tags:
  - local-ai
  - quantization
  - gguf
  - moe
  - benchmarking
---

# Module 8 — The Quantization Ladder as a Budget Problem

**7 hours** · Part of [[index|Two Pools]] · Prev: [[module-7-backend-shootout]] · Next: [[module-9-kv-quantization]]

> [!info] What this module is for
> "Smaller quant runs faster" is not a finding — it’s the first thing anyone guesses, and it’s true in the boring, obvious way that a smaller file loads faster. That fact alone isn’t worth seven hours. What *is* worth the time is a second-order effect that the boring version hides: a smaller quant doesn’t just mean fewer bytes to stream per token, it means those bytes take up less VRAM, which means **more expert layers fit resident on the GPU before you spill to DDR5**, which means the [[module-5-expert-placement|Module 5]] budget equation hands you a *different, smaller* optimal `n_cpu_moe` at every quant level. Two effects compound in the same direction — fewer bytes per read, and fewer of those reads paying DDR5’s tax at all — and the result is that real speedup from quantization should be **superlinear** relative to a naive "bytes shrank by X, so tok/s grows by X" prediction. This module derives that superlinearity factor on paper, then checks whether your GPU actually delivers more than the naive story would predict, or whether the paper prediction was itself too optimistic. Either answer is the module’s real result — the naive framing was never going to be it.

---

## 8.1 The reframe: why "smaller quant, faster inference" is the wrong question

> [!abstract] Goal
> State precisely what this module measures instead of the obvious quantization-speed relationship, before doing any arithmetic.

**What’s going on:** Every quantization tutorial on the internet will tell you Q4 is faster than Q8. That’s true and it’s not interesting, because it requires no model of *why* — it’s just "fewer bytes, less time," the same statement you could make about a Ethernet file transfer. What makes this course’s version of the question interesting is that quantization interacts with a decision you already made in [[module-5-expert-placement]]: **the optimal `--n-cpu-moe` is itself a function of how much VRAM the non-expert weights and each resident expert layer consume** — and that consumption *changes* at every quant level. A smaller quant doesn’t just shrink the read; it shrinks the layer footprint, which changes how many layers the [[module-4-vram-budget|budget equation]] lets you keep GPU-resident, which changes `n_cpu_moe`, which changes the read pattern *again*, on top of the byte-size change alone.

Two independent effects, same direction, same lever:

1. **Direct effect** — smaller quant, fewer bytes per active-expert read, at a *fixed* `n_cpu_moe`.
2. **Indirect effect** — smaller quant, smaller resident-layer footprint, so the [[module-4-vram-budget|budget equation]] permits a *smaller* `n_cpu_moe` at the same context length, meaning more of those already-smaller reads happen from the fast pool instead of DDR5 at all.

The naive prediction only captures effect 1. This module’s job is to quantify effect 2 as well, and report the gap between "naive byte-scaling prediction" and "budget-aware prediction, `n_cpu_moe` re-derived at each quant" as a single number: the **superlinearity factor**.

**Steps:**
1. Write, in your own words, the difference between "the read is smaller" and "there are fewer reads that pay the DDR5 tax at all" — these are not the same claim, and confusing them is the trap this module exists to correct.
2. State what you expect the sign of the superlinearity factor to be *before* computing it — do you expect real speedup to beat, match, or lag the naive byte-scaling story? Write your guess down; you’ll check it in 8.4.

> [!success] Done when
> You can state, without notes, why a smaller quant changes `n_cpu_moe` and not just the bytes-per-read — and you have a written pre-registered guess about whether the effect helps or hurts the naive prediction.

## 8.2 Model bytes across the ladder

> [!abstract] Goal
> Compute non-expert weight size, expert weight size, and per-layer expert footprint at Q8_0, Q6_K, Q5_K_M, and Q4_K_M.

**What’s going on:** [[module-5-expert-placement|Module 5]] gave you the Q4_K_M split directly: 0.89 GB non-expert weights, 17.41 GB expert weights, 18.30 GB total, at roughly 4.85 effective bits per weight averaged across the whole model (K-quants mix precision by tensor role, so this is a blended figure, not a single quant type applied uniformly). Scaling that blended bits-per-weight figure by the commonly cited approximate bits-per-weight of the other K-quant types gives a predicted table — **predicted, not measured, until you’ve actually downloaded each GGUF and checked its real file size**, which is step 1 below.

| Quant | Approx bits/weight | Non-expert GB | Expert GB | Total GB |
|---|---|---|---|---|
| Q8_0 | ~8.5 | 1.56 | 30.50 | 32.06 |
| Q6_K | ~6.56 | 1.20 | 23.56 | 24.76 |
| Q5_K_M | ~5.5 | 1.01 | 19.75 | 20.76 |
| Q4_K_M | ~4.85 | 0.89 | 17.41 | 18.30 |

Per-layer expert footprint (`expert_GB / 48 layers`), the figure the budget equation actually consumes:

| Quant | GB per resident expert layer |
|---|---|
| Q8_0 | 0.635 |
| Q6_K | 0.491 |
| Q5_K_M | 0.412 |
| Q4_K_M | 0.363 |

**Steps:**
1. Download (or locate, if already on disk) the actual GGUF file for each of these four quant levels and record its real file size — replace every number in the table above with your own measured figure before proceeding. Bits-per-weight figures quoted for K-quant types vary slightly by tensor mix and by which model produced the quant, so treat the table as a starting estimate, not ground truth.
2. Recompute the per-layer expert footprint (`expert_GB / 48`) from your own measured expert-weight total at each quant level.
3. Confirm your Q4_K_M row still matches the 0.89 / 17.41 / 18.30 GB split from [[module-5-expert-placement]] — if it doesn’t, something about your download or your accounting differs from Module 5’s, and you should resolve that discrepancy before trusting anything downstream in this module.

> [!success] Done when
> You have your own measured (not estimated) non-expert, expert, and per-layer GB figures for all four quant levels, with the Q4_K_M row cross-checked against Module 5.

> [!bug] Gotcha
> Don’t assume every tensor in a "Q8_0" GGUF is actually stored at 8 bits. Embedding and output-projection tensors are frequently kept at higher precision than the nominal quant name suggests, in every quant scheme including this one — that’s part of why the bits-per-weight figures above are blended averages, not exact per-tensor truths. When in doubt, a tool that dumps per-tensor quant types from the GGUF header (several exist in the llama.cpp ecosystem) settles the question directly instead of trusting the filename.

## 8.3 Re-deriving `n_cpu_moe` from the Module 4 budget equation

> [!abstract] Goal
> Apply the exact budget equation from [[module-4-vram-budget]] and [[module-5-expert-placement|Module 5, section 5.4]] at each quant level, holding context length fixed at 32K, and find the new minimum `n_cpu_moe` at each.

**What’s going on:** This is the step the naive "smaller quant = faster" story skips entirely. The budget equation doesn’t care what quant level produced a given non-expert or per-layer figure — it just consumes whatever GB numbers you feed it:

```
budget = 16 GB − non_expert_GB − KV_bytes(ctx) − overhead_GB
resident_layers = floor(budget / expert_GB_per_layer)
n_cpu_moe = 48 − resident_layers
```

Using 32K context (KV = 3.22 GB, from Module 4) and the same ~1.0 GB overhead placeholder Module 5 used (replace with your own 4.4 measured constant):

| Quant | non_expert | budget | expert_GB/layer | resident_layers | n_cpu_moe |
|---|---|---|---|---|---|
| Q8_0 | 1.56 | 10.22 | 0.635 | 16 | 32 |
| Q6_K | 1.20 | 10.58 | 0.491 | 21 | 27 |
| Q5_K_M | 1.01 | 10.77 | 0.412 | 26 | 22 |
| Q4_K_M | 0.89 | 10.89 | 0.363 | 30 | 18 |

The Q4_K_M row reproduces Module 5’s own `n_cpu_moe = 18` at 32K context exactly — a useful sanity check that this table is internally consistent with work you already did, not a new, disconnected calculation.

Notice the shape: **Q8_0 requires nearly double the offload of Q4_K_M at the same context length**, purely because its bigger non-expert and per-layer footprints eat further into the same 16 GB budget. This is effect 2 from 8.1, made concrete.

**Steps:**
1. Recompute this table using your own 8.2 measured GB figures and your own Module 4.4 overhead constant.
2. Confirm your Q4_K_M row still lands at `n_cpu_moe = 18` (or your own Module 5 figure, if you used a different overhead constant there) — if it doesn’t match, the discrepancy is in this table, not in Module 5.
3. Note, in one sentence, why Q8_0 needing `n_cpu_moe = 32` at the *same* context length Q4_K_M handles at `n_cpu_moe = 18` is not a coincidence — it’s the direct consequence of Q8_0’s bigger non-expert and per-layer bytes eating a bigger share of the same fixed 16 GB.

> [!success] Done when
> You have your own re-derived minimum `n_cpu_moe` at all four quant levels, at a fixed 32K context, with the Q4_K_M row matching your Module 5 figure.

## 8.4 Predicting tok/s — and the naive byte-scaling comparison

> [!abstract] Goal
> Predict tok/s at each quant level’s own budget-derived `n_cpu_moe`, then compute what naive byte-scaling alone would have predicted, and report the gap as a superlinearity factor.

**What’s going on:** Feed each row’s `n_cpu_moe` from 8.3 into the same `t_token(n)` model from [[module-5-expert-placement|Module 5, section 5.3]], using each quant’s own active-expert-bytes figure (`(8/128) × expert_GB`) and the theoretical bandwidth figures from [[workstation-specs]] (640 GB/s VRAM, 57.6 GB/s DDR5):

| Quant | n_cpu_moe | active expert GB | D(n) | V(n) | t_token (ms) | tok/s (budget-aware prediction) |
|---|---|---|---|---|---|---|
| Q8_0 | 32 | 1.906 | 1.271 | 2.195 | 25.49 | 39.2 |
| Q6_K | 27 | 1.473 | 0.828 | 1.844 | 17.27 | 57.9 |
| Q5_K_M | 22 | 1.234 | 0.566 | 1.679 | 12.45 | 80.3 |
| Q4_K_M | 18 | 1.088 | 0.408 | 1.570 | 9.54 | 104.9 |

The Q4_K_M row matches Module 5’s own predicted table exactly (`t_token ≈ 9.54 ms`, `~105 tok/s` at `n=18`) — again, a consistency check, not a new number.

Now the naive comparison: if quantization only shrank bytes, with **no** `n_cpu_moe` re-derivation — i.e., if you kept `n_cpu_moe = 32` (Q8_0’s own budget-derived value) fixed and just asked "how much faster does Q4_K_M run at *that same* offload level, purely from smaller reads" — total bytes shrank from 32.06 GB to 18.30 GB, a ratio of 1.75×. Naive prediction for Q4_K_M’s tok/s: `39.2 × 1.75 ≈ 68.7 tok/s`.

The budget-aware prediction for Q4_K_M, using Q4_K_M’s *own* re-derived `n_cpu_moe = 18`, is **104.9 tok/s** — about **53% faster than the naive byte-scaling prediction alone**. That 53% is the superlinearity: pure byte-count reduction accounts for the 1.75× ratio, but letting more layers move back onto the fast pool as the model shrinks buys an *additional* speedup the naive story has no way to predict.

```
superlinearity_factor = budget_aware_predicted_tok/s / naive_byte_scaled_predicted_tok/s
                       = 104.9 / 68.7 ≈ 1.53
```

**Steps:**
1. Recompute the `t_token(n)` table using your own 8.2/8.3 figures.
2. Recompute the naive byte-scaling prediction (fixed `n_cpu_moe`, bytes-only ratio) independently, and confirm your own superlinearity factor.
3. Repeat the naive-vs-budget-aware comparison for the Q6_K and Q5_K_M rows as well, not just Q4_K_M — confirm the superlinearity factor is positive (budget-aware beats naive) at every rung, not just the one worked example above.

> [!success] Done when
> You have your own predicted `t_token`/tok/s table across all four quants at their own re-derived `n_cpu_moe`, plus a naive byte-scaling comparison and a superlinearity factor for at least two of the four rungs — all still labeled *predicted*, not measured.

> [!bug] Gotcha
> Every number in this section is still a theoretical-bandwidth prediction, exactly as flagged in Module 5’s own 5.3 gotcha. Don’t write any of these tok/s figures down as a measured result — 8.7 is where you actually run `llama-bench` and find out how much of this prediction survives.

## 8.5 The format landscape: K-quants, IQ-quants, imatrix, and what won’t run here

> [!abstract] Goal
> Know which quantization formats are actually usable on this workstation’s Vulkan/HIP stack, and which published formats simply don’t apply.

**What’s going on:** Not every quantization scheme you’ll read about is a candidate here, and knowing which ones to skip saves real time.

- **K-quants** (`Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`, and further down the ladder) organize weights into **256-weight superblocks**, each with its own scale and (for some types) minimum, which is what lets a "4-bit" format still carry enough per-block correction to avoid catastrophic error — this is the family this module’s ladder uses throughout.
- **IQ-quants** (`IQ4_XS`, `IQ3_M`, and similar) are **codebook-based** rather than block-scale-based — they’re roughly 10% smaller than the equivalent K-quant at matched quality, and are recommended mainly **below Q4**, where the extra encoding complexity starts to pay for itself. Above Q4, K-quants are simpler to reason about and the size advantage of IQ-quants shrinks.
- **imatrix calibration** — an importance matrix computed from a calibration text corpus, used to weight which values in each block matter most during quantization — is broadly beneficial across quant levels, but **the calibration corpus itself is a genuinely contested choice, even among quantization tool maintainers**: different corpora produce measurably different downstream quality, and there’s no single agreed-upon "correct" corpus. The practical takeaway is not to build your own calibration pipeline from scratch — **prefer imatrix quants published by well-known, established quantizers** rather than a from-scratch build, since it sidesteps a genuinely unsettled question rather than guessing at an answer to it yourself.
- **AWQ and GPTQ** — two quantization schemes with substantial published quality data — have **no native Vulkan-runtime support**. If your practical inference path is `llama.cpp` on this GPU, **GGUF is the format that actually runs here**; AWQ/GPTQ checkpoints would need conversion, and even then the runtime support story is not the one this course’s stack is built on.
- **MXFP4**, notably the native format `gpt-oss` ships in (4.25 bits per element), has been added across `llama.cpp`’s CUDA, Vulkan, Metal, and CPU backends. But **hardware-accelerated FP4 matmul is CDNA4/datacenter-only** — RDNA4 can run MXFP4-encoded weights, but without the dedicated FP4 matmul path, meaning the arithmetic is emulated rather than natively accelerated on this card. RDNA4’s WMMA instructions natively support `f16`, `bf16`, `fp8` (both `e4m3` and `e5m2`), `int8`, and `int4` — but **not `fp4`**. If you evaluate a `gpt-oss`-family model on this workstation, that distinction is worth knowing before you interpret its throughput relative to a K-quant model of similar size.

**Steps:**
1. Confirm, from `llama-server --help` or your build’s documentation, which quant types your specific `llama.cpp` build actually supports loading — format support can lag behind the format’s publication date.
2. If you try an IQ-quant of this model, note its file size against the equivalent K-quant at a similar bit width, and check whether the ~10% size claim holds for this specific model.
3. If you have access to a `gpt-oss`-family GGUF, load it and compare its measured tok/s against a same-size K-quant model — note whether the lack of native RDNA4 FP4 acceleration shows up as a visible throughput penalty relative to what its file size alone would suggest.

> [!success] Done when
> You can state which quant families your build supports, and you know — without checking twice — that AWQ/GPTQ are off the table for this stack while GGUF K-quants and IQ-quants are the actual candidates.

## 8.6 Quality evidence from others’ benchmarks

> [!abstract] Goal
> Know the shape of the accuracy-vs-quantization curve reported elsewhere, clearly labeled as someone else’s measurement on different hardware and models — not yours.

**What’s going on:** These numbers come from published third-party evaluations, not this workstation, and not necessarily this exact model. Treat them as the shape you’re checking your own Aider Polyglot numbers against in 8.7, not as a substitute for running your own instrument.

On a Terminal-Bench-style agentic coding evaluation:

| Precision | Reported pass rate |
|---|---|
| BF16 | ~75% |
| Q4_K_M | ~75% (no measurable loss) |
| Q2_K_XL | ~70% |
| 1-bit | unusable |

On GPQA Diamond (a harder, knowledge-and-reasoning benchmark, not a coding benchmark):

| Precision | Reported accuracy |
|---|---|
| BF16 | ~96% |
| Q8_0 | ~95% |
| Q4_K_M | ~94% |
| Q2_K_XL | ~93% |
| 1-bit | ~50% (near random) |

The consistent shape across both: **the curve is flat from BF16 down through Q4_K_M**, with the real cliff appearing between Q2_K_XL and 1-bit, not gradually across the whole ladder. That’s the empirical basis for treating Q4_K_M as this course’s quality floor rather than a conservative starting guess — going below it is where the risk actually starts accumulating, per these third-party numbers.

**Steps:**
1. Note which of these two cited benchmarks is closer in spirit to what you actually care about (agentic coding vs. general knowledge/reasoning) and weight your own trust in the flat-region claim accordingly.
2. Write down, in your own words, why "flat from BF16 to Q4_K_M, cliff after Q2_K_XL" is a *different* claim than "quality degrades smoothly and proportionally with bit width" — the second claim is the intuitive one, and the cited data doesn’t support it.

> [!success] Done when
> You can state both cited curves from memory, correctly labeled as third-party measurements, and you can explain why the shape is a plateau-then-cliff rather than a smooth decline.

## 8.7 Measure the real ladder

> [!abstract] Goal
> Run `llama-bench` at each quant level’s own re-derived `n_cpu_moe`, and check the real superlinearity factor against your 8.4 prediction — using the Module 3 Aider Polyglot instrument, with variance bars, for quality.

**What’s going on:** Everything up to here has been arithmetic. This is where you find out how much of the 8.4 prediction survives contact with your GPU, and whether pass rate actually holds flat across this ladder the way the cited third-party curves suggest it should.

**Steps:**
1. For each of the four quant levels, run `llama-bench` at that quant’s own 8.3-derived `n_cpu_moe`, at 32K context:
   ```bash
   ./build-hip/bin/llama-bench -m qwen3-coder-30b-a3b-<quant>.gguf \
       -c 32768 -ncmoe <n_cpu_moe_for_this_quant> -t <your-elbow-thread-count> \
       -p 512 -n 128 -r 5 -o md
   ```
2. Tabulate measured `tg128` tok/s (mean ± stddev across the `-r 5` repetitions) against your 8.4 predicted figures for all four quants.
3. Compute your own **measured** superlinearity factor: `measured_Q4_K_M_tok/s ÷ naive_byte_scaled_prediction_from_8.4`. Compare it against the 1.53× predicted figure — note whether real hardware delivers more, less, or about the same superlinearity as the paper prediction.
4. Run the Module 3 Aider Polyglot instrument’s fast ~30-exercise subset against each quant level, recording pass rate and its stddev (from your Module 3 noise-floor measurement) at every rung. **Any difference between two quant levels smaller than your Module 3 sigma is not a result** — say so explicitly rather than reading tea leaves into noise.
5. Plot a Pareto chart: pass rate on one axis, tok/s on the other, one point per quant level, with error bars on both.

> [!success] Done when
> You have a measured tok/s table across all four quants (with stddev), your own measured superlinearity factor compared against the 1.53× prediction, a pass-rate-vs-tok/s Pareto plot with variance bars on the Aider Polyglot subset, and an explicit statement of which pass-rate differences clear your Module 3 noise floor and which don’t.

> [!question]- It’s not working
> 1. Measured superlinearity factor comes out *negative* (naive prediction beat the budget-aware one)? Double-check you’re actually running each quant at its own 8.3-derived `n_cpu_moe`, not accidentally holding `n_cpu_moe` fixed across all four rows — that would silently collapse this into the naive comparison and erase the entire effect this module is built around.
2. Pass rate looks identical (within noise) across Q8_0 through Q4_K_M? That’s consistent with the cited 8.6 plateau — not a failed run. Report it as a genuine flat result rather than searching for a difference that isn’t there.
3. tok/s stddev is unusually wide at one particular quant level? Check whether that quant’s GGUF uses a mixed-precision tensor layout that produces more variable per-token compute cost than the others — worth a note, not necessarily a rerun.

> [!bug] Gotcha
> Don’t compare your Q8_0 pass rate against a Q4_K_M pass rate from a *different* context length or a different `n_cpu_moe` sweep run days apart under different system load — hold every variable except quant level constant within a single comparison, the same discipline as the Module 7 shootout matrix.

## 8.8 The deliberate Q3 failure

> [!abstract] Goal
> Run one quant level below your working floor on purpose, and use it as a demonstrated collapse — not as a rung you’d actually deploy.

**What’s going on:** This step exists to make Q4_K_M’s "floor, not a starting guess" status something you’ve personally watched fail, not just something the cited 8.6 tables assert. Q3_K_M sits meaningfully below Q4_K_M — roughly 3.9 bits/weight against Q4_K_M’s ~4.85 — and the cited GPQA Diamond curve in 8.6 shows the real cliff starting past Q2_K_XL, so Q3_K_M itself is closer to the plateau’s edge than a guaranteed catastrophe. That’s exactly why it’s worth running: it’s a controlled test of "how far past the floor can you go before Aider Polyglot pass rate visibly moves outside your Module 3 noise floor," not a test that’s rigged to fail by construction.

Using the same scaling as 8.2 (~3.9 bits/weight), Q3_K_M totals roughly 14.72 GB (0.72 GB non-expert, 14.00 GB experts, 0.292 GB per resident layer). At 32K context, its own budget-derived `n_cpu_moe` works out to 11 — fewer offloaded layers than Q4_K_M’s 18, and a predicted tok/s comfortably above the Q4_K_M row, since a smaller quant both shrinks the read and shrinks the required offload simultaneously.

**Steps:**
1. Compute Q3_K_M’s own `n_cpu_moe` at 32K context, following the same 8.3 procedure, from your own measured GB figures for this quant.
2. Run the Module 3 Aider Polyglot fast subset against Q3_K_M at that `n_cpu_moe`, and compare the resulting pass rate against your Q4_K_M pass rate from 8.7, using the same Module 3 stddev.
3. Note the tok/s figure too, but treat it as a secondary result here — the point of this step is the quality number, not the speed number, which you’d expect to look good almost by construction at this bit width.
4. Write one sentence stating whether Q3_K_M’s pass rate cleared your Module 3 noise floor as a real regression against Q4_K_M, or whether it landed within noise — either outcome is informative, and neither is "the module failed."

> [!success] Done when
> You have a measured Q3_K_M pass rate on the Aider Polyglot subset, directly compared against your Q4_K_M pass rate using your own Module 3 stddev, with an explicit statement of whether the difference is real or noise.

---

## Check your understanding

> [!question]- 1. Why does quantization produce a *superlinear* speedup relative to naive byte-scaling, specifically on a MoE model offloaded with `--n-cpu-moe`?
> > [!success]- Answer
> > Because quantization does two things at once, not one. It shrinks the bytes read per active expert (the direct effect naive byte-scaling captures), but it also shrinks the per-layer VRAM footprint, which lets the Module 4 budget equation keep *more* layers resident on the fast pool at the same context length — meaning a smaller fraction of reads pay DDR5’s bandwidth tax at all. Naive byte-scaling only accounts for the first effect; the measured or predicted speedup beats it because the second effect compounds on top.

> [!question]- 2. Your measured Q3_K_M pass rate on the Aider Polyglot subset differs from Q4_K_M by less than your Module 3 stddev. Is that a null result?
> > [!success]- Answer
> > No — it’s a genuine, reportable result: it means this particular quality drop, at this particular bit width, isn’t distinguishable from run-to-run noise on this instrument. That’s different from claiming "Q3_K_M is exactly as good as Q4_K_M" — it means your current instrument doesn’t have the resolution to tell them apart, which is itself useful information about where your noise floor sits relative to the effect size you’re looking for.

> [!question]- 3. AWQ has strong published quality data at low bit widths. Why doesn’t this module include an AWQ rung in the ladder?
> > [!success]- Answer
> > Because AWQ has no native Vulkan-runtime support, and this course’s practical inference path on this GPU is `llama.cpp`, which runs GGUF. Strong published quality data on a format that doesn’t run on your actual stack isn’t a candidate — GGUF K-quants and IQ-quants are the formats that are actually usable here, which is why the ladder is built from them.

## What’s next

You now have a re-derived `n_cpu_moe` at every quant level, a predicted and measured superlinearity factor against the naive byte-scaling story, a Pareto plot checked against your own Module 3 noise floor, and a deliberately measured collapse point below the Q4_K_M floor. Every quant decision so far has assumed the KV cache itself stays at full f16 precision — the next lever is quantizing that cache independently of the weights, and [[module-9-kv-quantization]] takes it on with a warning this module didn’t need: KV-cache quantization carries a specific, documented danger for agentic tool-calling that weight quantization alone does not.

<script src="/tutor.js" defer></script>
