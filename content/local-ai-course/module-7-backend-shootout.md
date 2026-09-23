---
title: "Module 7 — Backend and Flags"
description: "A controlled HIP vs. Vulkan shootout on gfx1201, crossed with flash-attention and offload state, testing an open upstream report of a 4.7-6.7x Vulkan decode regression on hidden_size >= 4096 models."
tags:
  - local-ai
  - rocm
  - vulkan
  - flash-attention
  - benchmarking
---

# Module 7 — Backend and Flags

**9 hours** · Part of [[index|Two Pools]] · Prev: [[module-6-prefill-decode]] · Next: [[module-8-quantization]]

> [!info] What this module is for
> Every measurement so far has quietly assumed a backend — HIP, almost always — without treating that choice as a variable worth testing. This module treats it as one, and it happens to land on two genuinely open questions in the public record simultaneously. First: an open upstream `llama.cpp` issue reports Vulkan token generation running 4.7–6.7x slower than HIP on gfx1201 for models with `hidden_size` ≥ 4096, and smaller dense models have been separately reported to *beat* HIP on Vulkan by 10–35% — which means backend choice is model-shape dependent, not a fixed ranking, and nobody has published a controlled comparison across shapes for this specific GPU. Second: flash attention on ROCm/RDNA4 is asserted in passing far more often than it’s actually benchmarked with a controlled on/off comparison. Both gaps get closed, on your hardware, in this module — and both closures are equally valuable whether they confirm or refute what’s currently just an open issue and a folk assumption.

---

## 7.1 The hypothesis under test: issue #26663

> [!abstract] Goal
> State the exact, falsifiable claim from the open upstream issue, without treating it as established fact about your card.

**What’s going on:** `llama.cpp` issue #26663, **currently open** at time of writing, reports that on gfx1201 (the RX 9070 XT’s architecture) Vulkan token generation runs **4.7–6.7x slower** than HIP for models with `hidden_size ≥ 4096`, with effective memory bandwidth collapsing to roughly **70–100 GB/s** under Vulkan versus HIP’s reported **~455–464 GB/s** on the same hardware. The suspected cause is a `KHR_coopmat` shader/shape constraint forcing a slow decode code path for wider hidden dimensions. As a contrast point, the same report notes a 4B model with `hidden_size` 2560 was *unaffected* — roughly 183 tok/s and ~424 GB/s effective bandwidth — which is offered as support for the ≥4096 threshold specifically, not a blanket "Vulkan is slow" claim.

**This is a hypothesis to test on your own hardware, not a property of your GPU that you should assume going in.** Open issues can be stale, workload-specific, or fixed by the time you read this — treat #26663 exactly the way you’d treat any unverified claim from Module 3 onward: something your own measurement either reproduces or refutes, with the refutation being just as publishable a result as the reproduction.

Worth noting before you design the test matrix: **Qwen3-Coder-30B-A3B has `hidden_size` 2048** — below the reported 4096 threshold. That makes it an interesting test case specifically *because* the issue’s own threshold predicts it should be largely unaffected, not because it’s a foregone conclusion either way. If your measurements on this model show a large HIP/Vulkan gap anyway, that’s evidence the threshold doesn’t hold cleanly; if they don’t, that’s consistent with (not proof of) the reported threshold.

**Steps:**
1. Read issue #26663 in full, including any linked discussion or maintainer replies, and write down the exact claim in your own words — the specific hidden_size threshold, the specific reported slowdown range, and the suspected mechanism.
2. Confirm Qwen3-Coder-30B-A3B’s `hidden_size` from its own `config.json` — verify 2048 yourself rather than trusting this module’s restated figure.
3. Note explicitly, in writing, that this issue describes *someone else’s* hardware and workload — your RX 9070 XT and your build of `llama.cpp` are the actual subjects of this module, and the issue is only the hypothesis being tested against them.

> [!success] Done when
> You can state the exact hidden_size threshold, slowdown range, and suspected mechanism from issue #26663 from memory, and you’ve independently confirmed Qwen3-Coder-30B-A3B’s own hidden_size from its config rather than taking this module’s word for it.

## 7.2 Flash attention: the `auto` default and the `llama-bench` caveat

> [!abstract] Goal
> Understand flash attention’s current default behavior and a specific, dated pitfall in comparing your numbers against older published benchmarks.

**What’s going on:** `-fa` (flash attention) is a **tri-state** flag: `on`, `off`, or `auto`. As of PR #15434 (August 30, 2025), the default is **`auto`** — `llama.cpp` decides per-request whether flash attention is applicable and beneficial, rather than requiring the user to force it. This matters for reading *anyone’s* benchmark numbers, including your own from earlier modules: a build from before that PR defaults to `off`, and a build from after it defaults to a decision-making `auto` that may or may not match what forcing `on` would give you.

The specific, dated caveat: **`llama-bench` itself hardcoded `-fa off` internally until build b9437**, independent of whatever the server’s own default was. This means older published `llama-bench` numbers — even ones run on a `llama.cpp` version whose *server* already defaulted to `auto` — may reflect flash-attention-off performance regardless, purely because the benchmarking tool itself hadn’t been updated to match. **No controlled AMD-specific `-fa on` vs. `-fa off` benchmark exists publicly at the time of writing** — this module is where you produce one, deliberately, rather than relying on an assumption carried over from CUDA-side benchmarking culture.

One more thing worth flagging as historical and possibly stale, specifically so you verify it rather than repeat it: on non-NVIDIA Vulkan backends, flash attention has at times fallen back to running on the **CPU entirely**, with predictably catastrophic performance. Two 2026 PRs — #27952 and #28507 — added `coopmat1` int8 Vulkan flash-attention and MMQ shader paths specifically targeting RDNA3/RDNA4, which may have addressed this. Whether your current build still exhibits any CPU fallback for `-fa` on Vulkan is itself worth checking directly rather than assuming either the old broken behavior or the new fixed behavior.

**Steps:**
1. Check your `llama.cpp` build date/commit against both PR #15434 (Aug 30, 2025, the `-fa auto` default) and build b9437 (the `llama-bench` `-fa off` hardcoding fix) to know which defaults actually apply to your binaries.
2. Run a quick check: with `-fa on` explicitly forced on your Vulkan build, watch CPU utilization during a decode run — if a CPU core pins near 100% in a way that doesn’t match expected thread-pool behavior from your Module 0 elbow measurement, that’s a sign of the historical CPU-fallback behavior still being present.
3. Note, in writing, which of `-fa on`, `-fa off`, and `-fa auto` your 7.3 test matrix will explicitly force at each cell — don’t leave any cell at an ambiguous, unstated default.

> [!success] Done when
> You know your exact `llama-bench` and `llama.cpp` build’s relationship to both dated fixes, and you’ve confirmed (not assumed) whether Vulkan `-fa on` runs on GPU or silently falls back to CPU on your current build.

> [!bug] Gotcha
> If you ever compare your numbers against an older benchmark post, check which side of the b9437 line it came from before treating any difference as a real backend or hardware effect — an old post’s "`-fa` had no effect" conclusion may simply mean their `llama-bench` was hardcoded to `-fa off` regardless of what flag they thought they were passing.

## 7.3 Design the controlled shootout

> [!abstract] Goal
> Lay out the full test matrix — backend × flash attention × offload state × model shape — before running any of it, so the results are comparable cell-to-cell.

**What’s going on:** This module crosses three binary-ish variables and one multi-valued one, and the value of the whole exercise depends on holding everything except the variable under test constant within each comparison. The variables:

- **Backend:** HIP vs. Vulkan (your two Module 1 builds)
- **Flash attention:** `-fa on` vs. `-fa off` (forced explicitly, per 7.2 — not left at `auto`, which would hide the effect you’re trying to isolate)
- **Offload state:** minimal offload vs. your Module 5 chosen-optimum `n_cpu_moe`
- **Model shape:** at least four models spanning the hidden_size range the #26663 threshold cares about

For model shape, span meaningfully below and above the reported 4096 threshold — for example: a small dense model around `hidden_size` 1536–2560 (well below threshold), Qwen3-Coder-30B-A3B itself at `hidden_size` 2048 (below threshold, but a MoE model rather than dense — a useful second axis), a mid-size dense model around `hidden_size` 3584–4096 (near the threshold), and a larger dense model at `hidden_size` 5120 or above (well above threshold). **Check each model’s actual `hidden_size` from its own config before assuming a number from memory or from a model card summary** — family naming conventions (e.g., "7B", "14B") do not reliably predict hidden_size across different model families.

That’s a 2 × 2 × 2 × 4 = 32-cell matrix at minimum. Not every cell needs a full `llama-bench -r 5` treatment if time is tight — but the backend × flash-attention comparison, specifically, needs enough repetitions to report a real variance figure, since that’s the comparison with no existing public data at all.

**Steps:**
1. List your four (or more) test models with their confirmed `hidden_size` values, sorted, and mark which side of 4096 each falls on.
2. Build the full matrix as a table before running anything — rows are model shapes, columns are the backend × flash-attention × offload combinations — so you know exactly what "done" looks like before you start.
3. Decide your repetition count per cell (`-r` in `llama-bench`) up front, and hold it constant across every cell so variance figures are comparable to each other.

> [!success] Done when
> You have a written, empty test matrix — every cell identified, every model’s hidden_size confirmed from its own config — ready to fill in during 7.4.

## 7.4 Run the full matrix

> [!abstract] Goal
> Fill in the matrix from 7.3 with real `tg128` (decode) and `pp512` (prefill) numbers, per-cell variance included, for both backends.

**What’s going on:** This is the actual data-collection step. Run each cell identically across backends so the only thing that changes is the backend binary itself (and, deliberately, the flash-attention and offload settings you’re crossing it with).

**Steps:**
1. For each model shape, run the HIP build across both flash-attention states and both offload states:
   ```bash
   ./build-hip/bin/llama-bench -m <model>.gguf -ngl 99 -fa on \
       -t <your-elbow-thread-count> -p 512 -n 128 -r 5 -o md
   ./build-hip/bin/llama-bench -m <model>.gguf -ngl 99 -fa off \
       -t <your-elbow-thread-count> -p 512 -n 128 -r 5 -o md
   ```
   (Repeat with your Module 5 `-ncmoe` offload setting in place of full `-ngl 99` residency, for the models large enough to need it.)
2. Repeat the identical set of runs against the Vulkan build, changing only the binary path.
3. Record `tg128` and `pp512` averages **and their stddev** for every cell — the stddev is not optional here, since a 4.7–6.7x claim needs to be clearly outside your own measurement noise to mean anything.
4. For any cell where Vulkan decode collapses dramatically relative to HIP, cross-check with a system monitor during the run — confirm the GPU is actually the bottleneck (high GPU utilization, unremarkable CPU) rather than an unrelated CPU-bound stall producing a misleading number.

> [!success] Done when
> Every cell of the 7.3 matrix has a measured `tg128` and `pp512` figure with stddev, for both HIP and Vulkan, across all tested model shapes.

> [!question]- It’s not working
> 1. Vulkan numbers look catastrophically bad on *every* model shape, not just the ones above the hidden_size threshold? That would be evidence against the specific ≥4096 threshold from #26663 — a real and useful finding, but double check first that your Vulkan build itself is healthy (confirm it’s using the discrete GPU, not a software/CPU Vulkan fallback device) before concluding the threshold is wrong.
> 2. HIP and Vulkan numbers are suspiciously close on a model well above the threshold? Confirm you’re not accidentally running the same binary for both rows (a copy-paste path error is the most common cause of "no difference found where a difference was expected").
> 3. Per-cell stddev is large enough to make the HIP/Vulkan gap ambiguous? Increase `-r` for just the ambiguous cells rather than the whole matrix, to get a tighter estimate where it actually matters.

## 7.5 Write the decision rule

> [!abstract] Goal
> Turn the filled matrix into a stated, actionable rule: which backend to use, under which conditions, on this GPU.

**What’s going on:** A pile of numbers is not a conclusion. The point of running a controlled matrix is to be able to state a rule of the shape "use backend X when hidden_size ≥ Y and offload is Z" — precise enough that a future reader (including future you, several modules from now) can apply it without re-running the whole shootout.

**Steps:**
1. Plot decode tok/s (HIP vs. Vulkan) against hidden_size across your tested models, with flash-attention state as a separate series or facet.
2. Identify, from your own data, whether a threshold hidden_size exists past which Vulkan decode degrades sharply relative to HIP — state your own measured threshold, and compare it explicitly to the ≥4096 figure from #26663 (confirming, refuting, or partially matching it).
3. Do the same for the flash-attention on/off comparison: is there a consistent, stddev-clearing effect in either direction, on either backend?
4. Write the final decision rule as a plain sentence, e.g.: *"On this RX 9070 XT, use HIP for hidden_size ≥ [your measured threshold] regardless of offload state; below that threshold, Vulkan is [faster / slower / indistinguishable] by [X]%, and flash attention should be forced [on / off] because [reason]."*

> [!success] Done when
> You have a stated decision rule of the form "use backend X when hidden_size ≥ Y and offload is Z," backed by a table with per-cell variance, that someone could apply to a new model without re-running this module’s matrix themselves.

---

## Check your understanding

> [!question]- 1. Why is it important to test model shapes both below and above the hidden_size 4096 threshold from issue #26663, rather than just testing Qwen3-Coder-30B-A3B?
> > [!success]- Answer
> > Because #26663’s claim is specifically about a *threshold* — a hidden_size below which Vulkan is fine or even faster, and above which it collapses. Testing only one model (especially one below the threshold, like Qwen3-Coder-30B-A3B at hidden_size 2048) can only tell you about that one point; it can’t confirm or refute a threshold shape at all. You need models on both sides of the reported boundary to find out whether your own hardware shows the same discontinuity, a different threshold, or no threshold at all.

> [!question]- 2. Your `llama-bench` build predates b9437. What does that mean for any `-fa on` vs. `-fa off` comparison you try to run with it?
> > [!success]- Answer
> > It means `llama-bench` may hardcode `-fa off` internally regardless of what flag you actually pass, so a comparison that looks like "on vs. off" could really be "off vs. off" — showing no difference not because flash attention has no effect, but because both sides of your comparison were silently running without it. You’d need to update past b9437 before this module’s flash-attention comparison means anything.

> [!question]- 3. Your matrix shows Vulkan beating HIP by 15% on a small dense model (hidden_size 2048) but losing to HIP by 5x on a larger dense model (hidden_size 5120). Is this a contradiction?
> > [!success]- Answer
> > No — this is exactly the shape of result the #26663 report and the separate "Vulkan beats HIP on smaller models" reports both predict together: backend performance is model-shape dependent, not a fixed ranking. A result like this is consistent with (though not identical to) the reported pattern, and it’s exactly the kind of finding that justifies stating a hidden_size-conditioned decision rule instead of a single blanket "use backend X" recommendation.

## What’s next

You now have a decision rule for backend and flash-attention choice that’s grounded in your own controlled measurements rather than a forum consensus or a single open GitHub issue taken on faith — and, depending on what you found, either a reproduction or a refutation of a currently-open upstream report, both of which are genuinely useful to have documented. [[module-8-quantization]] takes the placement and backend choices settled across Modules 4 through 7 and holds them constant while sweeping the one variable that trades model quality directly for VRAM headroom: the quantization level itself, from the Q4_K_M floor this course has assumed throughout, down through Q2_K_XL and into 1-bit territory, checked every step of the way against the Aider Polyglot harness and noise floor built in [[module-3-eval-instrument]].

<script src="/tutor.js" defer></script>
