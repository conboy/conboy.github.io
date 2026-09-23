---
title: "Capstone 1 — The Predicted Machine"
description: "Assembling every measured constant from this course into one predictive model, writing down tok/s and TTFT predictions for three configurations never run — including two models roughly four times this course’s main model — before running a single one."
tags:
  - local-ai
  - capstone
  - roofline
  - moe
  - prediction
---

# Capstone 1 — The Predicted Machine

**2 weeks** · Part of [[index|Two Pools]] · Prev: [[module-13-claude-code]] · Next: [[capstone-2-effective-context]]

> [!info] What this capstone is for
> Every module in this course has produced one measured constant: a DDR5 bandwidth figure, a VRAM bandwidth figure, a calibration factor between theory and reality, a per-token KV-cache byte count, an optimal `n_cpu_moe` curve, a prefill/decode crossover, a backend decision rule, a quantization Pareto front, a tool-call failure rate. Individually, each of those is a fact about this workstation. Assembled together, they are a *model* of this workstation — a small set of equations that should, in principle, predict the behavior of a model you have never loaded, on a configuration you have never run. This capstone is the moment that claim gets tested for real, in public, with the predictions written down before the answer is known. You will predict tok/s and TTFT for three configurations you have not run — including **gpt-oss-120b** and **GLM-4.5-Air**, both roughly three to four times this course’s main model by parameter count — using nothing but your own measured constants and each model’s own published architecture. Then you will run them, at thermal steady state, not a cold burst. Being right is the headline result this course was built to produce. Being wrong, with the gap explained, is exactly as publishable, and this section says so more than once because it’s the single easiest thing to forget under the pressure of wanting the plot to look clean.

---

## C1.1 Assemble the model from your own constants

> [!abstract] Goal
> Write down, in one place, every measured constant this course has produced, and confirm each is a **measurement**, not a spec-sheet number, before using any of them to predict anything.

**What’s going on:** The predictive model this capstone builds is the same `t_token(n) = V(n)/BW_vram + D(n)/BW_ddr` equation from [[module-5-expert-placement|Module 5]] and [[workstation-specs]] — but every term in it now needs to be your own number, not the theoretical 640 GB/s and 57.6 GB/s used as a first-pass illustration back in Module 5.3. Gather:

- **η_ddr and η_vram** — your Module 0/1 calibration constants, the ratio between theoretical and sustained bandwidth on each pool. These convert `BW_vram`/`BW_ddr` in the equation above from spec-sheet numbers into what your actual card and your actual DIMMs deliver.
- **η_model** — your Module 2 calibration factor, absorbing kernel-launch overhead, attention compute, and everything else the pure-bandwidth model doesn’t capture.
- **The VRAM overhead constant** — your Module 4.4 measured figure for compute buffers and allocator overhead, which the budget equation needs to compute how many layers actually fit resident.
- **The KV-cache byte-per-token figure** — 98,304 bytes/token at FP16 for this course’s main model, and the block-layout-derived bytes/element figures from [[module-9-kv-quantization|Module 9.2]] for whichever quant level you plan to run a new model at.
- **The optimal-`n_cpu_moe` curve shape** — not the specific Qwen3-Coder numbers, but the *method*: given a model’s non-expert weight size, expert weight size, layer count, and your VRAM budget, [[module-5-expert-placement|Module 5.4]]’s procedure produces a minimum feasible `n_cpu_moe` at any context length, for **any** MoE model whose architecture you can look up.
- **The prefill/decode crossover behavior** from [[module-6-prefill-decode|Module 6]] — expect it to shift for a much larger model, and say in which direction before checking.
- **The backend decision rule** from [[module-7-backend-shootout|Module 7.5]] — confirm it still applies, or note explicitly if a new model’s shape falls outside the hidden_size range you tested it on.

**Steps:**
1. Build a single reference table — a page, a spreadsheet, whatever you’ll actually use — with every constant above, each one labeled with which module produced it and the exact command or measurement that generated it.
2. For each constant, write one sentence confirming it’s a measurement from your own hardware, not a number you copied from this course’s illustrative tables (which, per every "Gotcha" in Modules 5 and 8, use theoretical bandwidth deliberately and are not your numbers to reuse here).
3. Confirm you can reconstruct `t_token(n)` for a **hypothetical** model with made-up architecture numbers, purely from this reference table, before moving to C1.2 and plugging in a real model’s real numbers.

> [!success] Done when
> You have one assembled reference table of every measured constant this course produced, each traceable to the module and command that generated it, and you’ve confirmed you can mechanically apply the full `t_token(n)` model to an architecture you haven’t seen yet.

## C1.2 Choose three configurations you have never run

> [!abstract] Goal
> Select three genuinely unseen configurations to predict — two are specified for you, the third is your own choice, made deliberately rather than as an afterthought.

**What’s going on:** "Never run" is a real constraint, not a formality. A configuration is disqualified the moment you’ve already measured it, or interpolated it closely enough from an existing sweep that the prediction is really just reading your own graph back to yourself.

Two configurations are fixed:

- **gpt-oss-120b** — 117 B total parameters, 5.1 B active per token, roughly 63 GB at its native low-bit format, mostly resident in system RAM on this workstation because it categorically cannot fit in 16 GB of VRAM.
- **GLM-4.5-Air** — 106 B total parameters, 12 B active per token, roughly 60 GB — a much heavier per-token DDR5 bill than gpt-oss-120b despite a similar total footprint, because more than twice as many active parameters have to be read every single token.

The third is yours to choose, with one rule: it must add real information to the signature figure in C1.6, not just a fourth data point clustered near ones you already have. Two reasonable choices: a genuinely different MoE model with a distinct active-parameter count from either of the two above, or a configuration of this course’s own main model at a context length or `n_cpu_moe` value that falls **strictly outside** every sweep you ran in Modules 5 through 9 — not an interpolation of two points you already measured, a real extrapolation past the edge of your existing data.

**Steps:**
1. Look up each fixed model’s actual published config — layer count, expert count, active experts per layer, hidden size, head configuration — from its model card or `config.json`, the same discipline [[workstation-specs]] used for this course’s main model. Do not substitute a guess for a number you can actually go look up.
2. Pick your third configuration and write one sentence justifying why it adds information the other two don’t — if you can’t articulate that sentence, pick a different third configuration.
3. Confirm, in writing, that you have never run any of the three — check your own benchmark logs from Modules 5–9 if you’re not sure.

> [!success] Done when
> You have three configurations selected, with real architecture numbers looked up (not guessed) for the two fixed models, and a written justification for why your third choice adds genuine information to the eventual plot.

## C1.3 Predict — before running anything

> [!abstract] Goal
> Produce a written, timestamped prediction for tok/s and TTFT on all three configurations, using only your C1.1 constants and C1.2 architecture numbers — locked in before a single benchmark runs.

**What’s going on:** This is the discipline the whole capstone is built around, restated as plainly as possible: **predict first, run second, in that literal order, with evidence you did.** A prediction written after you’ve already peeked at a result isn’t a prediction — it’s a rationalization wearing a prediction’s clothes, and it’s worth being honest with yourself about which one you’re doing.

Work each of the three configurations through two tiers of prediction, mirroring the naive-vs-budget-aware framing [[module-8-quantization|Module 8]] used for the quantization ladder:

**Tier A — naive, DDR5-bound floor.** Because gpt-oss-120b and GLM-4.5-Air are both roughly 4–7x this course’s main model’s active-parameter footprint and neither fits meaningfully into 16 GB of VRAM, treat almost all of their active bytes as forced through DDR5 every token — the same "what if this were dense and had to stream everything" framing from [[module-5-expert-placement|Module 5.3]], applied here to the *active* parameter count rather than the full model, since MoE sparsity is still real even though VRAM residency mostly isn’t:

```
active_bytes_per_token ≈ active_params × (total_size_GB / total_params)
t_token ≈ active_bytes_per_token / (BW_ddr × η_ddr)
predicted_tok/s ≈ 1 / t_token
```

Work this through for gpt-oss-120b: roughly 0.54 bytes/param at its native ~4.3-bit format (63 GB ÷ 117 B params), so active bytes/token ≈ 5.1 B × 0.54 ≈ 2.75 GB. And for GLM-4.5-Air: roughly 0.57 bytes/param (60 GB ÷ 106 B params), so active bytes/token ≈ 12 B × 0.57 ≈ 6.8 GB — visibly the heavier bill the course context already told you to expect, now with a number attached.

**Tier B — budget-aware, accounting for whatever sliver of VRAM residency you can actually claim.** Neither model’s non-expert weights and router are free to ignore — look up each model’s actual non-expert weight size and confirm how much of the 16 GB card that leaves for anything else, then apply the same `V(n)/BW_vram + D(n)/BW_ddr` split from Module 5, using your *own* measured bandwidths, not the illustrative ones. For a model this large, expect the budget-aware correction to be **small relative to the Tier A floor** — there is so little VRAM headroom left after a ~60+ GB model’s non-expert weights and KV cache that the placement curve barely has room to move, unlike the rich `n_cpu_moe = 0…48` continuum Module 5 found for the much smaller main model. Say this explicitly in your prediction write-up: for models this size, placement stops being much of a lever, and the naive DDR5-bound floor and the budget-aware prediction should land close together.

**Steps:**
1. Compute both tiers, using your own C1.1 constants, for all three configurations, and write every intermediate number down — not just the final tok/s figure, so a wrong prediction can be diagnosed later rather than just relabeled.
2. Predict TTFT the same way, using your own [[module-6-prefill-decode|Module 6]] prefill methodology — note explicitly that prefill for a model this size, mostly resident in DDR5, should be even more compute-bound-vs-bandwidth-bound-mismatched than the main model’s own Module 6 crossover, and say which direction you expect that to push TTFT before checking.
3. Commit the full prediction table to a dated file or note **before** downloading any of the three models’ weights, and treat that timestamp as the actual scientific commitment this capstone is asking for — not a formality, the entire point.

> [!success] Done when
> You have a timestamped, written prediction table — Tier A and Tier B tok/s, plus TTFT — for all three configurations, computed entirely from your own measured constants, committed to a file before you’ve run a single one of them.

> [!bug] Gotcha
> Resist the temptation to "adjust" a prediction after a quick informal test run "just to sanity check the setup." If you run anything before your prediction is committed, you no longer have a prediction — you have a hindsight-adjusted guess, and the entire value of this capstone depends on not doing that, even once, even informally.

## C1.4 Run at thermal steady state, not a cold burst

> [!abstract] Goal
> Verify your predictions against a sustained, hours-scale agentic workload with full thermal and power logging — not a `llama-bench` burst — because a burst and a real coding session are measuring different machines.

**What’s going on:** `llama-bench` reports numbers from a run lasting seconds to at most a few minutes. An actual agentic coding session runs for hours, with the CPU doing continuous expert-matmul work the entire time it’s serving DDR5-resident experts. Clocks throttle. DDR5 controllers can behave differently under sustained load than under a burst. Nobody publishes whether the [[module-5-expert-placement|Module 5]] knee — the point where increasing `n_cpu_moe` stops buying meaningful VRAM headroom relative to the tok/s given up — shifts once the machine has been running flat out for the better part of an hour, and that gap is exactly what this section fills.

**Steps:**
1. Set up continuous logging for CPU clock speed, CPU package power, CPU temperature, and (if your tooling supports it) GPU clock/power/temperature — `lm-sensors`, `turbostat`, and `rocm-smi` (or its current equivalent) cover this on Linux; log at a fixed interval (e.g. once per 10–30 seconds) for the entire run.
2. Run a sustained, realistic agentic coding workload — not a synthetic benchmark loop, a real multi-hour session generating and editing code, ideally reusing your Module 3/8 Aider Polyglot harness on repeat, or a long real coding session — for 45 to 60 minutes continuously, against each of your three C1.2 configurations in turn.
3. Compare tok/s and TTFT in the first five minutes of each run against the last five minutes, at matched offload/context configuration, and check specifically whether the Module 5-style placement knee has moved: does the same `n_cpu_moe` value that looked optimal cold still look optimal once the CPU has been under sustained expert-matmul load for the better part of an hour?
4. Note any thermal throttling events directly from your logged clock-speed data, and correlate them with any tok/s degradation you observe across the run.

> [!success] Done when
> You have continuous clock/temperature/power logs across a 45–60 minute sustained run for each configuration, a direct first-five-minutes-vs-last-five-minutes tok/s comparison, and an explicit statement of whether the placement knee shifted at thermal steady state.

## C1.5 Run the three predictions

> [!abstract] Goal
> Actually run the three configurations from C1.2, at thermal steady state per C1.4, and record measured tok/s and TTFT against your locked-in C1.3 predictions.

**What’s going on:** This is the step everything else in this capstone was building toward — and it should feel almost anticlimactic if C1.1 through C1.4 were done properly, because by this point there should be very little left to decide. You’re executing a plan you already wrote down.

**Steps:**
1. Load each of the three configurations and run the full sustained-workload procedure from C1.4, recording steady-state tok/s and TTFT for each.
2. Record every deviation from your intended configuration honestly — a model that wouldn’t load at your planned `n_cpu_moe`, a context length you had to reduce, anything that forced a change from what C1.3 actually predicted for. A prediction checked against a silently-modified configuration is not a real check.
3. Tabulate measured tok/s and TTFT next to your Tier A and Tier B predictions for all three configurations, in one table.

> [!success] Done when
> You have measured steady-state tok/s and TTFT for all three configurations, tabulated directly next to your C1.3 predictions, with any configuration deviations noted honestly rather than smoothed over.

## C1.6 The signature figure

> [!abstract] Goal
> Plot tok/s against active-bytes-per-token, with your predicted 1/x curve and all real MoE models — including this course’s own main model — sitting on or off it.

**What’s going on:** This is the course’s headline visual, and it’s worth building carefully. The `t_token(n) ≈ active_bytes_per_token / BW` relationship predicts a **1/x curve**: tok/s should fall off in inverse proportion to how many active bytes a token has to read, once you’ve accounted for where those bytes actually live. Plotting real measured points from multiple, genuinely different MoE models against that predicted curve is the strongest single piece of evidence this course can produce that "the roofline is a function of placement and active bytes, not a fixed hardware ceiling" is a real, falsifiable, and — ideally — confirmed claim, rather than an abstract framing device from the course’s own introduction.

**Steps:**
1. Plot active-bytes-per-token on the x-axis and measured tok/s on the y-axis.
2. Draw the predicted 1/x curve using your own measured bandwidth constants (η_ddr-corrected, not theoretical).
3. Place at least four points on the plot: this course’s main model at its own Module 5/8 measured active-bytes-per-token and tok/s, plus your three C1.2/C1.5 configurations.
4. For any point that sits meaningfully off the curve, write one sentence naming the most likely reason — a placement constraint that pinned it away from the ideal split, a thermal effect from C1.4, a measured bandwidth that turned out to differ from your assumed constant at that particular access pattern.

> [!success] Done when
> You have one plot with a predicted 1/x curve and at least four real measured points on it, each point’s distance from the curve either explained or explicitly flagged as unexplained.

## C1.7 Prediction error, and the anchor check

> [!abstract] Goal
> Compute prediction error per configuration, and compare your results against one external anchor — used only as a sanity check, never as a target.

**What’s going on:** Report error honestly, in both directions. A config where you predicted 20 tok/s and measured 22 is a 10% error; a config where you predicted 20 and measured 8 is a 60% error and a real finding about where your model broke down — both belong in the same table, with no softening language on the second one.

**One external anchor, for context only:** a different machine — a 12 GB GPU paired with dual-channel DDR5 — measured 23.9 to 25 tok/s on gpt-oss-120b with expert offload. The same rig, with RAM clocked down to 2000 MT/s, fell to 9.7–11 tok/s — a roughly 3x swing driven by RAM speed alone, nothing else changed. This is **someone else’s machine, with a different memory topology and a different bandwidth ratio than yours** — it is not a target, and a measured number on your own workstation landing far from it is not automatically a failure. Use it only to confirm your own number is in a plausible neighborhood for this class of hardware, and note explicitly if it isn’t.

**Steps:**
1. Compute `|predicted − measured| / measured × 100` for tok/s (both Tier A and Tier B predictions) and for TTFT, for each of your three configurations.
2. Compare your gpt-oss-120b measurement against the cited anchor range, and write one sentence on whether your own number’s relationship to it makes sense given the known differences in memory topology between your machine and the anchor’s.
3. For the configuration with the largest prediction error, write a real diagnosis — which specific assumption in C1.3 (a bandwidth constant, a VRAM-residency estimate, a KV-cache figure) is the most likely source of the gap.

> [!success] Done when
> You have a prediction-error percentage for every configuration, a written comparison against the external anchor that treats it correctly as context rather than a target, and a real diagnosis for whichever configuration missed by the widest margin.

---

## Check your understanding

> [!question]- 1. Why does this capstone insist the prediction be written down and timestamped before running anything, rather than just trusting yourself to remember your reasoning afterward?
> > [!success]- Answer
> > Because a prediction adjusted — even slightly, even unconsciously — after seeing a partial result stops being a prediction and becomes a rationalization that looks like one. The entire evidentiary value of "I predicted X and measured Y" depends on X having been fixed before Y was known. A timestamped, committed file is what makes that claim checkable by someone other than you, including a future version of yourself who might otherwise misremember how confident the original guess really was.

> [!question]- 2. Why should the budget-aware (Tier B) correction be small relative to the naive DDR5-bound floor (Tier A) for gpt-oss-120b and GLM-4.5-Air, when it was a large, central effect for this course’s main 30B model in Module 5?
> > [!success]- Answer
> > Because the size of the correction depends on how much VRAM headroom is actually available to move the placement curve around in, and a ~60 GB+ model leaves almost none. Module 5’s main model had a rich `n_cpu_moe = 0…48` continuum precisely because the whole model was only ~18 GB — comparable to the card’s own 16 GB capacity, so where experts sat made a large difference. A model three to four times that size can only ever keep a thin sliver of non-expert weights (and maybe a few resident expert layers) in VRAM regardless of configuration, so almost all of its active bytes are DDR5-bound no matter what you do — the naive floor and the budget-aware prediction converge because there’s so little budget left to be "aware" of.

> [!question]- 3. One of your three C1.2 configurations measures 40% slower than predicted at thermal steady state, but matched prediction closely in a short cold-burst check. What does this most likely indicate, and is it a failed capstone?
> > [!success]- Answer
> > It most likely indicates real thermal throttling or a sustained-load DDR5 behavior that a short burst never triggers — exactly what C1.4 was built to catch, since `llama-bench`-style bursts and hours-long agentic sessions can be measuring different machines. It is not a failed capstone — it’s the capstone doing its job. A clean cold-burst match that falls apart under sustained load is a genuine, reportable finding about the gap between benchmark numbers and real usage, and matches this course’s explicit ground rule that being wrong, with the gap explained, is exactly as publishable as being right.

## What’s next

You now have a complete predictive model built from your own measured constants, three genuinely unseen configurations predicted before they were run — including two models roughly four times this course’s main model — measured at thermal steady state rather than a cold burst, and the course’s signature figure with real points checked against a predicted curve. Whether your predictions landed close or missed by a wide margin, you have the number and the explanation, which is the actual deliverable. [[capstone-2-effective-context]] turns from throughput to correctness, and asks a harder question this capstone deliberately left alone: even a model running at exactly the tok/s you predicted is worthless past the point its context stops working — and that point, it turns out, is not the number on the model card.

<script src="/tutor.js" defer></script>
