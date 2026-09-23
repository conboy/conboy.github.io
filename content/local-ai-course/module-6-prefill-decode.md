---
title: "Module 6 — Prefill vs. Decode"
description: "Measuring the crossover context length past which the decode win from expert placement is erased by the prefill cost — and why agentic coding lives in exactly that territory."
tags:
  - local-ai
  - moe
  - performance
  - llama.cpp
---

# Module 6 — Prefill vs. Decode

**7 hours** · Part of [[index|Two Pools]] · Prev: [[module-5-expert-placement]] · Next: [[module-7-backend-shootout]]

> [!info] What this module is for
> Module 5 measured a beautiful decode-speed curve and found a placement that reaches ~105 tok/s at 32K context instead of ~3 tok/s for a dense-equivalent model. That result is real, but it is also only half of what happens when you send a prompt to this model, and it’s the half agentic coding tools lean on the least. Cline and Roo don’t send you one token and wait — they resend the entire growing conversation, 10 to 40 thousand tokens, on **every single turn**, and all of that has to be prefilled before the first new token comes back. Prefill and decode are computationally opposite regimes: prefill is compute-bound, decode is memory-bandwidth-bound, and `--n-cpu-moe` — the flag that just won you a 33x decode speedup — moves work onto the CPU, which is comparatively far worse at the compute-bound job than at the bandwidth-bound one. This module measures exactly how much worse, finds the context length past which that penalty erases the whole Module 5 win, and puts time-to-first-token next to tok/s in every table from here forward, because a config that decodes fast but keeps you staring at a spinner for 45 seconds first is not actually a good coding experience.

---

## 6.1 Prefill vs. decode: the missing second clause

> [!abstract] Goal
> State precisely why prefill and decode have opposite bottlenecks, using the same arithmetic-intensity reasoning Module 2 introduced for `pp512`/`tg128`.

**What’s going on:** Module 2 already drew this line once, at the level of `llama-bench` flags; this module is where it becomes load-bearing. **Prefill** processes a batch of *B* prompt tokens through the model in one forward pass. Every weight loaded from memory during that pass gets reused across all *B* tokens in the batch — arithmetic intensity (operations performed per byte moved) scales up with batch size. Once *B* is large enough, the GPU’s raw compute throughput (FLOPs), not how fast bytes arrive from memory, becomes the limiting factor: prefill is **compute-bound** above the hardware’s ridge point.

**Decode** produces one token at a time. Each step is a matrix-vector product, not a matrix-matrix product — arithmetic intensity is approximately 1 operation per byte, because there is no batch to amortize a weight load over. The *entire* active weight set has to stream out of memory again, essentially from scratch, for every single token. This is exactly the regime the `t_token(n)` equation from Module 5 describes, and it’s why decode is **memory-bandwidth-bound**, full stop, regardless of model size.

Now the consequence that makes this module necessary: `--n-cpu-moe` moves expert weights onto the CPU/DDR5 side specifically to exploit decode’s bandwidth-bound nature — DDR5 is slower than VRAM, but decode only reads the small *active* fraction of experts per token anyway, so the penalty is bounded. Prefill has no equivalent mercy. During prefill, a batch of many tokens is being processed together, and because different tokens in the batch can route to *different* experts, a large enough batch will, in the worst case, touch a large fraction of all experts across the batch — closer to the full 17.41 GB than the single-token 1.088 GB active figure from Module 5. Compute-bound work on the CPU is also simply much slower than compute-bound work on a modern GPU, independent of any bandwidth argument at all. **So offload should hurt prefill considerably more than it helps decode**, and there should be some context length past which the prefill penalty from a given `n_cpu_moe` setting outweighs the decode benefit it was chosen for.

**Steps:**
1. State, in your own words, why decode’s arithmetic intensity is "approximately 1 operation per byte" regardless of model size, while prefill’s intensity scales with batch size.
2. Restate, specifically for expert offload, why a large prefill batch is more likely to touch a large fraction of all 128 experts than a single decode step is — tie this to the routing behavior from Module 5.2.
3. Write down, before measuring anything, your own prediction: does the crossover context length you’re about to look for happen at a short prompt or a long one? State your reasoning, not just a guess.

> [!success] Done when
> You can state the prefill/decode arithmetic-intensity distinction without notes, and you’ve written down a falsifiable prediction for where the crossover context length should land, before running 6.3.

## 6.2 Quadratic attention: O(n²) FLOPs, O(n) storage

> [!abstract] Goal
> Understand why a longer prompt costs disproportionately more prefill compute than its length alone would suggest — and why flash attention fixes the memory problem but not the compute problem.

**What’s going on:** The KV cache from [[module-4-vram-budget]] is O(n) in context length — storage grows linearly, one KV pair added per token. But the attention **score matrix** — every query token’s dot product against every key token — is O(n²): doubling context length quadruples the number of pairwise score computations, because every one of the n query positions has to attend to all n key positions. An 8× longer prompt means up to **64× more attention compute**, entirely independent of the linear growth in KV storage. This is easy to miss precisely because the KV cache figure you computed in Module 4 *is* linear, and it’s tempting to assume attention cost scales the same way — it doesn’t.

[[concepts#Flash attention|Flash attention]] tiles this computation so the full n×n score matrix is never materialized in memory all at once — it processes attention in blocks, keeping a running softmax normalization, so peak memory usage stays roughly linear instead of quadratic. This is a genuine and important fix for the **memory** blowup. It does **not** reduce the underlying **FLOP** count — you still perform the same O(n²) volume of arithmetic; flash attention just avoids paying an O(n²) memory cost to do it. For this module’s purposes, that distinction matters: flash attention should help you fit longer contexts without running out of memory mid-computation, but it should not be expected to flatten the prefill-time-vs-context-length curve you’re about to measure in 6.3 — that curve is fundamentally quadratic in the attention component no matter how efficiently it’s tiled.

**Steps:**
1. Compute the ratio of attention score-matrix entries between a 4K-token prompt and a 32K-token prompt (8× longer) — confirm it’s 64×, not 8×.
2. State, in one sentence, which problem flash attention solves (memory) and which it does not (FLOP count).
3. Note, for 6.3, that you should not expect flash attention on vs. off to change the *shape* of your measured prefill-time-vs-length curve — only, potentially, whether very long contexts are reachable at all before running out of memory.

> [!success] Done when
> You’ve computed the 64× figure yourself for an 8× context-length increase, and can state precisely what flash attention does and does not fix.

> [!bug] Gotcha
> Don’t confuse "flash attention makes long context memory-feasible" with "flash attention makes long context fast." It solves a real memory problem, cleanly — but the underlying quadratic compute cost of attention at long context is untouched by it, and it’s exactly that compute cost that drives the prefill side of this module’s crossover measurement.

## 6.3 Measure TTFT and prefill tok/s across prompt sizes

> [!abstract] Goal
> Measure both time-to-first-token and prefill throughput at 1K, 4K, 16K, and 32K prompt sizes, at two configurations: your Module 5 minimum-offload and your Module 5 chosen-optimum.

**What’s going on:** [[concepts#TTFT|Time-to-first-token]] (TTFT) is the wall-clock time between sending a request and receiving the first generated token back — for a single request with no prior cached prefix, it’s dominated almost entirely by prefill time, since decode of the very first token only happens after the full prompt has been processed. This is the number that determines whether a coding session *feels* responsive, and it’s easy to lose track of if you only ever look at steady-state `tok/s`.

You’ll compare two configurations from Module 5: the **minimum offload** you found feasible at a short context (e.g., `n_cpu_moe=12` at 8K from Module 5.4), and the **chosen optimum** for a longer target context (e.g., `n_cpu_moe=18` at 32K). The question is whether the second configuration — which offloads *more* experts to win decode speed at longer context — pays a meaningfully larger prefill penalty than the first, and at what prompt length that penalty becomes large enough to matter.

**Steps:**
1. Use `llama-bench`’s `-d` (prefill depth) flag to measure prefill throughput with an existing context already present, reported as `pp512 @ dN` — this simulates prefilling *new* tokens on top of a conversation that already has N tokens of history, which is closer to what an agentic tool actually does turn-over-turn than a cold `pp512` alone:
   ```bash
   ./build-hip/bin/llama-bench -m qwen3-coder-30b-a3b-q4_k_m.gguf \
       -ncmoe 12 -p 512 -d 1000 -d 4000 -d 16000 -d 32000 \
       -t <your-elbow-thread-count> -r 5 -o md
   ```
2. Repeat the identical sweep with `-ncmoe 18` (or whatever your Module 5 chosen-optimum was), changing nothing else.
3. For TTFT specifically, stand up `llama-server` at each configuration and measure wall-clock time to the first streamed token using a script that sends a prompt of the target length and records the timestamp of the first SSE chunk — `curl`’s `-w` timing variables report total request time but not first-byte time for a streamed response, so a short Python script using `requests` with `stream=True`, timestamping the first non-empty chunk, is more reliable here.
4. Tabulate, for both configurations, at all four prompt lengths: prefill tok/s (from `llama-bench`, avg ± stddev) and measured TTFT (from your streaming script).

> [!success] Done when
> You have a table of prefill tok/s and TTFT at 1K/4K/16K/32K prompt length, for both the minimum-offload and chosen-optimum configurations — eight data points per configuration, sixteen total.

> [!question]- It’s not working
> 1. `-d` flag not recognized, or behaving like a plain `-p`? Check your `llama-bench` build is recent enough to support prefill-depth testing — this is a newer feature than the base `-p`/`-n` flags.
> 2. TTFT measurements wildly inconsistent between repeated runs at the same prompt length? Check whether `cache_prompt` (see 6.5) is causing some runs to hit a cached prefix from a previous test — restart the server, or vary the prompt content slightly, to force a genuine cold prefill each time you’re trying to measure one.
> 3. TTFT via your streaming script reads suspiciously close to zero? Confirm you’re timestamping the first *content* chunk, not a connection-open or headers-received event that arrives before any actual generation has happened.

## 6.4 Find the crossover

> [!abstract] Goal
> Identify the prompt length past which the chosen-optimum configuration’s prefill penalty outweighs the decode-speed advantage it was chosen for, relative to minimum offload.

**What’s going on:** This is where the two tables from 6.3 get combined into the actual answer this module is chasing. At short prompts, the chosen-optimum configuration’s larger `n_cpu_moe` should cost you relatively little in prefill time (the prompt is short, so even a compute-bound penalty is small in absolute terms), while still buying you its full decode-speed advantage on the response tokens. At long prompts, the same configuration’s prefill penalty grows — plausibly faster than linearly, per 6.2 — until the extra seconds spent in prefill exceed whatever decode-speed advantage the response tokens would have recovered.

**Steps:**
1. For each of the four prompt lengths, compute total time for a fixed-size response (pick a constant, e.g., 200 output tokens) under both configurations: `total_time = TTFT + (response_tokens / decode_tok_s)`.
2. Plot `total_time` for both configurations across all four prompt lengths on the same axes.
3. Identify the prompt length at which the two curves cross — below it, the chosen-optimum configuration should still win overall (decode speed dominates); above it, minimum offload should start winning on total wall-clock time despite its slower decode, because it never sacrificed as much prefill speed for that decode gain.
4. State the crossover prompt length explicitly, and note where typical agentic-tool turn sizes (10–40K tokens, per this module’s opening) fall relative to it.

> [!success] Done when
> You have two total-time curves and a stated crossover context length, with an explicit note on whether typical 10–40K-token agentic turns fall above or below it for your measured configurations.

> [!question]- It’s not working
> 1. Curves never cross across your tested range? That’s a real, useful result too — it means one configuration dominates the other across the entire range you tested, which is itself worth stating plainly rather than forcing a crossover that isn’t there.
> 2. Crossover point looks suspiciously close to one of your four sampled prompt lengths rather than falling cleanly between two of them? Add an intermediate sample (e.g., 8K or 24K) to narrow down the actual crossover more precisely instead of reporting an approximate range as if it were exact.

## 6.5 Prompt caching: what makes agentic turns fast without client cooperation

> [!abstract] Goal
> Understand `llama-server`’s prompt-caching behavior well enough to know when a "long prompt" measurement in 6.3–6.4 is actually a cold prefill versus a cache hit — and configure it deliberately rather than by accident.

**What’s going on:** `llama-server`’s `cache_prompt` option defaults to **true**, and it performs longest-common-prefix matching between an incoming request and whatever was processed for the *previous* request on that server. In an agentic coding session, each new turn resends the entire growing conversation — but almost all of it is identical to what was just sent a moment ago, with only a small amount of new content appended at the end. Longest-common-prefix caching means the server only has to prefill the *new* suffix, not the whole resent conversation, which is precisely what makes growing agentic conversations stay fast turn-over-turn **with zero cooperation required from the client** — Cline and Roo don’t need to know or care that this caching exists; it happens transparently on the server side purely because the resent prefix matches.

`--cache-reuse N` (PR #9866) extends this further: it reuses **non-contiguous** cached chunks by shifting KV cache entries around internally, rather than requiring an exact contiguous prefix match. This helps when something *changes* mid-conversation — a file edit inserted earlier in the context, for instance — that would otherwise break a naive longest-common-prefix match at the point of the edit and force a full re-prefill of everything after it. The maintainers have suggested `--cache-reuse 256` as a reasonable starting value. Note also that `--prompt-cache FNAME` is a **`llama-cli`-only** flag for saving/loading a prompt cache to/from disk between separate process invocations — it does not apply to `llama-server`, which manages its cache in-memory across requests within a single running process instead.

To verify whether a given request actually hit the cache rather than assuming it did, check the response fields `cache_n` (tokens served from cache) and `prompt_n` (tokens actually prefilled) in `llama-server`’s response — if `cache_n` is close to your prompt’s full length, you measured a cache hit, not a cold prefill, and any 6.3 measurement you thought was "cold" needs rechecking.

**Steps:**
1. Confirm `cache_prompt` is at its default (true) or explicitly set, and understand that this means your 6.3 measurements, if run as repeated requests against the same long-lived server process without deliberately varying the prompt, may have been silently benefiting from cache hits rather than measuring true cold prefill.
2. Rerun a small check: send the same prompt twice in a row to a freshly started server, and compare `prompt_n`/`cache_n` and TTFT between the first (necessarily cold) and second (likely cached) request — this quantifies how large the caching effect actually is on your hardware.
3. For your 6.3/6.4 "cold prefill" measurements specifically, either restart the server between each measured prompt, or vary the prompt content enough that no meaningful prefix match is possible, so you know you measured what you intended to measure.
4. Separately, try `--cache-reuse 256` against a prompt with a small edit inserted partway through (simulating a file diff mid-conversation), and confirm via `cache_n`/`prompt_n` whether it successfully avoided a full re-prefill.

> [!success] Done when
> You have a direct before/after comparison of TTFT and `cache_n`/`prompt_n` for a cold vs. cached identical-prompt request, confirming whether your 6.3/6.4 measurements were genuinely cold prefills or partially cache-assisted, and a corrected note on any measurement that turned out to be a cache hit in disguise.

---

## Check your understanding

> [!question]- 1. Why should expert offload (`--n-cpu-moe`) hurt prefill more than it hurts decode, given that both phases read the same expert weights?
> > [!success]- Answer
> > Decode reads only the small active-expert fraction per token (top-8 of 128, per Module 5) and is memory-bandwidth-bound, so moving that fraction to slower DDR5 costs a bounded, small penalty per token. Prefill processes many tokens in one batch, and different tokens in that batch can route to different experts — a large batch is likely to touch a much larger fraction of the full 17.41 GB of expert weights than any single decode step does. Combined with prefill being compute-bound (where CPU is comparatively far slower than GPU, independent of bandwidth at all), the same offload setting that costs decode very little can cost prefill considerably more.

> [!question]- 2. A configuration reaches 118 tok/s decode but has a 45-second TTFT on a 32K-token prompt. Is this a good configuration for agentic coding?
> > [!success]- Answer
> > Not on this evidence alone, and probably not in practice. Agentic tools like Cline resend the growing conversation every turn, so a 45-second wait before the *first* token of every response — even at excellent steady-state decode speed afterward — makes each turn feel unresponsive in a way a single `tok/s` number completely hides. This is exactly why TTFT has to be reported next to tok/s from this module forward: a config that wins decisively on decode can still be a poor overall choice once prefill cost at realistic prompt lengths is accounted for.

> [!question]- 3. Why doesn’t flash attention flatten the prefill-time-vs-prompt-length curve you measured in 6.3, even though it’s specifically designed to make long-context attention tractable?
> > [!success]- Answer
> > Flash attention avoids materializing the full n×n attention score matrix in memory by tiling the computation and keeping a running softmax — this fixes a real O(n²) *memory* problem. It does not reduce the O(n²) *FLOP* count: the same volume of pairwise attention arithmetic still happens, just without ever holding the whole matrix in memory at once. So flash attention can be the difference between a long prompt fitting in memory at all and it not fitting — but it doesn’t change the fundamentally quadratic shape of how prefill compute time grows with prompt length.

## What’s next

You now have two curves — prefill and decode — instead of the single decode number Module 5 measured, plus a stated crossover context length past which expert offload’s decode win gets erased by its prefill cost, and a clear picture of how much of that crossover measurement was genuinely cold versus quietly cache-assisted. [[module-7-backend-shootout]] adds a third axis to all of this: whether HIP or Vulkan is even the right backend to be running any of these measurements on in the first place, since the answer — on this exact GPU architecture — turns out to depend on model shape in ways nobody has published a controlled comparison for yet.

<script src="/tutor.js" defer></script>
