---
title: "Module 2 — Calibrating on a Model That Fits"
description: "Predicting tok/s from measured VRAM bandwidth, checking that prediction against llama-bench, and watching a 30B MoE fail to load in 16GB."
tags:
  - local-ai
  - llama-bench
  - quantization
  - benchmarking
---

# Module 2 — Calibrating on a Model That Fits

**6 hours** · Part of [[index|Two Pools]] · Prev: [[module-1-toolchain]] · Next: [[module-3-eval-instrument]]

> [!info] What this module is for
> Modules 0 and 1 gave you two measured bandwidth numbers and a real ratio between them. This module turns those numbers into a prediction, and then checks the prediction against reality — that gap is the actual point of the module, not a footnote. You’ll run two Qwen2.5-Coder models fully resident in VRAM, predict their decode speed from nothing but your measured VRAM bandwidth and the models’ byte sizes, then measure with `llama-bench` and compute how far off you were. That gap gets a name — η_model — and it’s a correction factor you’ll reuse every time you predict throughput for the rest of this course. The module ends by deliberately trying something that doesn’t fit: Qwen3-Coder-30B-A3B at Q4_K_M is 18.30GB, and your card has 16GB of VRAM. Watching it fail to load is not a detour — it’s the reason the rest of this course exists. If everything fit in VRAM, there would be no "two pools" problem to reason about.

---

## 2.1 Learn `llama-bench` before you trust it

> [!abstract] Goal
> Understand what `llama-bench`’s flags and output actually measure before running a single real number.

**What’s going on:** `llama-bench` is llama.cpp’s built-in benchmarking tool, and its output separates two fundamentally different phases of running a language model, because they have fundamentally different performance characteristics:

- **`pp512`** — prompt processing (prefill) at a batch of 512 tokens. This is **compute-bound**. Prefill processes many tokens in one batch, so each weight loaded from VRAM gets reused across every token in that batch — the arithmetic intensity (operations performed per byte loaded) scales up with batch size. With enough batched tokens, the GPU’s compute throughput (FLOPs), not its memory bandwidth, becomes the limit.
- **`tg128`** — text generation (decode) for 128 tokens. This is **memory-bandwidth-bound**. Decoding one token at a time is a matrix-vector product, not a matrix-matrix product: arithmetic intensity is approximately 1 operation per byte, because the *entire* active weight set has to stream out of VRAM again for every single token produced, with almost no reuse across tokens. This is exactly the case the `t_token(n)` equation from Module 0 describes, and it’s why decode speed is the number this whole course keeps coming back to.

The other flags you’ll use: `-m` (model path), `-p`/`-n` (prompt-processing / text-generation token counts for the test), `-d` (prefill *depth* — prefixes the test with N tokens of context already present, so you can measure `pp512 @ d5000` — how prefill behaves once there’s already 5000 tokens of context, not just a cold start), `-b`/`-ub` (batch and micro-batch size), `-t` (CPU thread count — use your Module 0 elbow value here, not `-1`, since llama.cpp’s own `-1` default calls `hardware_concurrency()`, which double-counts SMT threads rather than reporting physical cores), `-ngl` (GPU layers to offload — 99 forces "all of them" in practice), `-r` (repetitions — llama-bench reruns each test `-r` times and reports **avg ± stddev**, not a single sample, specifically because a single run of a memory-bound benchmark is noisy), and `-o` (output format, e.g. `-o md` for a Markdown table). Tokenization and sampling time are deliberately excluded from the timed region — you’re measuring the model’s forward pass, not your prompt template’s string handling.

**Steps:**
1. Read `llama-bench --help` in full once, matching each flag above to what you just read.
2. Note that `-fa` (flash attention) is tri-state — `on`/`off`/`auto`, default `auto` as of PR #15434 in the upstream project — but `llama-bench` itself hardcoded `-fa off` in older builds, up until build b9437. If you ever compare your numbers against older published benchmarks, check which side of that line the other numbers came from; flash-attention-off numbers and flash-attention-auto/on numbers are not directly comparable.
3. Confirm which llama.cpp build (HIP or Vulkan, from Module 1) you’re running by checking the binary path, and pick your `-t` value from Module 0’s measured elbow, not from a guess.

> [!success] Done when
> You can explain, without looking it up, why `pp512` is compute-bound and `tg128` is memory-bandwidth-bound, and you know your own `-t` value going into every test in this module.

> [!question]- It’s not working
> 1. Different llama.cpp versions have added/renamed flags over time — if a flag from this list doesn’t exist in your build, check `--help` for the current equivalent rather than assuming the manual is wrong.
> 2. Confused about `-d`? It’s easy to misread as a *replacement* for `-p` rather than an addition — `-p 512 -d 5000` measures prefilling 512 new tokens *on top of* an existing 5000-token context, reported as `pp512 @ d5000`, not a 5000-token prefill on its own.

## 2.2 Load Qwen2.5-Coder-7B, fully VRAM-resident

> [!abstract] Goal
> Get the 7B model loaded with every layer on the GPU, and confirm it before benchmarking anything.

**What’s going on:** Qwen2.5-Coder-7B at Q4_K_M quantization is roughly 4.5GB — comfortably under your 16GB of VRAM even with room left over for context and the GPU’s own overhead. `-ngl 99` is the standard llama.cpp idiom for "offload every transformer layer to GPU" (there are never 99 layers in a 7B model; the flag just needs to be at least as large as the real layer count to mean "all of them").

**Steps:**
1. Download the Qwen2.5-Coder-7B-Instruct GGUF at Q4_K_M quantization from its model repository.
2. Confirm the file size lands close to the expected ~4.5GB.
3. Run a quick load-and-check with `llama-cli` (from Module 1’s HIP build) to confirm all layers land on GPU:
   ```bash
   ./build-hip/bin/llama-cli -m qwen2.5-coder-7b-instruct-q4_k_m.gguf -ngl 99 -p "test" -n 8
   ```
4. Check the startup log lines for the offloaded-layer count — it should match (or exceed) the model’s real layer count, confirming nothing spilled to CPU.

> [!success] Done when
> The 7B model loads with startup logs confirming 100% of its layers offloaded to GPU, and produces output with no CPU-fallback warnings.

## 2.3 Predict, then measure — 7B

> [!abstract] Goal
> Write down a predicted decode speed *before* measuring, then run `llama-bench` and compute how wrong you were.

**What’s going on:** The prediction is the simplest possible application of the memory-bound decode model: if decode has to stream the entire active weight set out of VRAM once per token, then `tok/s ≈ BW_vram_measured / model_bytes`. This ignores KV cache traffic, attention overhead, and kernel/launch inefficiency — it’s a floor-level theoretical estimate, not a claim that real hardware will hit it. The ratio between measurement and this prediction is η_model, and the entire reason this section exists is to find out what that ratio actually is on your hardware, not to confirm a guess. Expect somewhere in the 0.4-0.8 range — but the number is the point, not the expectation.

**Steps:**
1. Compute the **prediction**: `predicted_tok_s = measured_vram_GBps / model_bytes_GB` using your Module 1 measured VRAM bandwidth and the 7B model’s actual file size in GB. Write this number down before you run anything.
2. Run `llama-bench` for real, using your Module 0 thread count and forcing full GPU offload:
   ```bash
   ./build-hip/bin/llama-bench -m qwen2.5-coder-7b-instruct-q4_k_m.gguf \
       -ngl 99 -t <your-elbow-thread-count> -p 512 -n 128 -r 5 -o md
   ```
3. Read the `tg128` row’s average tok/s (with its ± stddev from the `-r 5` repetitions).
4. Compute `η_model = measured_tok_s / predicted_tok_s`.

> [!success] Done when
> You have a written-down prediction, a measured `tg128` average ± stddev from 5 repetitions, and η_model as one concrete number for the 7B model.

> [!question]- It’s not working
> 1. Is measured tok/s *higher* than predicted? That would be surprising given the prediction is meant as a rough ceiling estimate — double check `model_bytes_GB` is the actual quantized file size, not the unquantized parameter count times 4 bytes.
> 2. Is stddev across the 5 repetitions unexpectedly large? Check nothing else is contending for the GPU or CPU during the run, and rerun in a quiet system state.
> 3. Does `llama-bench` report `-ngl` didn’t take effect (layers still on CPU)? Recheck the exact flag spelling and that you’re pointing at the HIP or Vulkan build you intend, not a stray CPU-only build from earlier testing.

## 2.4 Predict, then measure — 14B

> [!abstract] Goal
> Repeat the exact same prediction-then-measurement exercise on the larger 14B model, and see whether η_model holds steady or shifts.

**What’s going on:** Qwen2.5-Coder-14B at Q4_K_M is roughly 9GB — still comfortably under 16GB, but more than double the 7B’s footprint, and getting closer to VRAM’s practical ceiling once context and overhead are accounted for. Running the identical methodology at a second model size is what turns η_model from "one number you measured once" into "a constant you can trust to generalize" — or, just as usefully, into evidence that it *doesn’t* generalize and depends on model size.

**Steps:**
1. Download the Qwen2.5-Coder-14B-Instruct GGUF at Q4_K_M, confirm the file size lands near ~9GB.
2. Compute the prediction the same way as 2.3, using this model’s actual byte size.
3. Run the identical `llama-bench` invocation, swapping only the model path:
   ```bash
   ./build-hip/bin/llama-bench -m qwen2.5-coder-14b-instruct-q4_k_m.gguf \
       -ngl 99 -t <your-elbow-thread-count> -p 512 -n 128 -r 5 -o md
   ```
4. Compute η_model for the 14B model and compare directly against the 7B’s η_model from 2.3.

> [!success] Done when
> You have a second η_model figure for the 14B model, and a written statement of whether it’s close to the 7B figure or meaningfully different — either outcome is useful, and this comparison is what makes η_model something you can extrapolate from rather than a single anecdote.

> [!question]- It’s not working
> 1. Does 14B’s tok/s come back *proportionally* lower than 7B’s, roughly matching the ~2x size difference? That’s actually the expected memory-bound behavior — decode time scales with bytes streamed, so this is a good sign your measurement methodology is consistent, not a problem.
> 2. Is η_model noticeably lower at 14B than at 7B? That could point to KV cache and context overhead becoming a larger fraction of total VRAM traffic at the bigger model size — worth a note, not necessarily something to "fix."

## 2.5 Try the model that doesn’t fit

> [!abstract] Goal
> Attempt to fully offload Qwen3-Coder-30B-A3B at Q4_K_M — 18.30GB total — into 16GB of VRAM, and record exactly how it fails.

**What’s going on:** Qwen3-Coder-30B-A3B is a mixture-of-experts model: "A3B" signals roughly 3B *active* parameters per token even though the full parameter count (and therefore the full set of expert weights that must be resident somewhere) is much larger. At Q4_K_M quantization the full model is 18.30GB total — 0.89GB of non-expert weights (shared/routing components) plus 17.41GB of expert weights. The "active 3B" framing describes *compute* per token, not *memory footprint*: every expert has to be resident somewhere the model can reach it, because routing decides per-token, per-layer which experts fire, and you can’t predict in advance which ones you’ll need. 18.30GB does not fit in a 16GB card with `-ngl 99`, full stop — this isn’t a close call or a tuning problem, it’s the module’s deliberate, designed failure.

**Steps:**
1. Download the Qwen3-Coder-30B-A3B GGUF at Q4_K_M quantization and confirm the file size lands near 18.30GB.
2. Attempt the exact same "load everything onto GPU" invocation you used for the two earlier models:
   ```bash
   ./build-hip/bin/llama-cli -m qwen3-coder-30b-a3b-q4_k_m.gguf -ngl 99 -p "test" -n 8
   ```
3. Record the exact failure — an out-of-memory allocation error, a driver-level allocation failure, or `llama.cpp` refusing to offload all layers and silently falling back to partial CPU offload (check the startup logs carefully; this can look like a "success" that’s actually much slower than expected, rather than a clean crash).
4. Compute, on paper, why it doesn’t fit: 18.30GB of weights alone already exceeds 16GB before you add a single byte of KV cache, activation buffers, or the GPU’s own reserved overhead.

> [!success] Done when
> You have the exact error text (or the exact partial-offload behavior) from attempting to load an 18.30GB model into a 16GB card, and a one-line written explanation of why — weights alone exceed capacity, independent of the "3B active" framing, which describes compute not memory footprint.

> [!question]- It’s not working (i.e., it appears to "work")
> 1. Did the process actually complete a full `-ngl 99` GPU-only load, or does the startup log show some layers silently placed on CPU despite the flag? Read the per-layer offload summary carefully — llama.cpp does not always hard-fail when a model doesn’t fit; it can fall back to partial CPU placement, which will look like "it worked" but run at CPU-DDR speeds for the spilled layers.
> 2. Did the process crash instead with an OS-level OOM kill rather than a clean llama.cpp/ROCm allocation error? Check `dmesg` for an OOM killer entry if the process just silently disappears with no error text.

> [!bug] Gotcha
> "Active parameters" (the A3B part of the name) describes how much *compute* happens per token in a mixture-of-experts model — it does not describe how much *memory* the model needs resident. Every expert layer has to live somewhere reachable, because the router picks different experts on different tokens; you cannot predict which 3B-worth of experts you’ll need next and pre-evict the rest. This distinction — compute footprint vs. memory footprint — is the single most common confusion people bring to MoE models, and it’s exactly what this exercise is designed to make concrete.

---

## Check your understanding

> [!question]- 1. Why is `pp512` compute-bound while `tg128` is memory-bandwidth-bound, given they’re running through the same weights on the same GPU?
> > [!success]- Answer
> > Prefill (`pp512`) processes a batch of tokens together, so each weight loaded from VRAM is reused across every token in that batch — arithmetic intensity (operations per byte) scales up with batch size, and eventually the GPU’s raw compute throughput becomes the limit rather than how fast bytes arrive. Decode (`tg128`) produces one token at a time, which is a matrix-vector product: arithmetic intensity is roughly 1 operation per byte because there’s no batch to amortize the weight load over, so the entire active weight set has to stream out of VRAM again for essentially every token, making memory bandwidth the hard limit.

> [!question]- 2. Your η_model comes back at 0.55 for the 7B model. Is that a bad result?
> > [!success]- Answer
> > No — the brief for this exercise explicitly expects something in the 0.4-0.8 range, and the point of the exercise is to measure this ratio, not to hit a specific target. η_model < 1 is expected because the naive prediction (`BW_vram / model_bytes`) ignores real overhead: KV cache reads/writes, attention computation, kernel launch overhead, and imperfect memory-access efficiency all eat into the theoretical ceiling. 0.55 is a real, usable calibration constant for this hardware and this model family — not a failure.

> [!question]- 3. Why does the Qwen3-Coder-30B-A3B failure matter for the rest of this course, rather than just being "pick a smaller model and move on"?
> > [!success]- Answer
> > Because it’s the first concrete demonstration that VRAM capacity, not just VRAM bandwidth, is a hard constraint — and once a model’s weights don’t fit entirely in the fast pool, you’re forced into decisions about *which* bytes live in VRAM and which spill to DDR (or get evaluated some other way), which is exactly the "two pools" placement problem the whole course equation (`t_token(n) = V(n)/BW_vram + D(n)/BW_ddr`) describes. Every module from here on either designs around that constraint or measures its consequences — this is the moment the course’s central problem stops being abstract.

## What’s next

You now have a calibration constant, η_model, measured on two model sizes that fit, and a hard, documented failure on one that doesn’t — the exact boundary this course is built around. But none of these measurements mean anything for real coding work until you can show a config change actually helps or hurts *task performance*, not just tok/s. [[module-3-eval-instrument]] builds that instrument: a local Aider Polyglot harness, run enough times to know its own noise floor before you trust a single comparison against it.

<script src="/tutor.js" defer></script>
