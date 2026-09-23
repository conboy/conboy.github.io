---
title: "Module 3 — The Eval Instrument"
description: "Building an Aider Polyglot harness against your local endpoint, defining a stratified subset, and measuring the noise floor before trusting any comparison."
tags:
  - local-ai
  - evaluation
  - aider
  - methodology
---

# Module 3 — The Eval Instrument

**8 hours** · Part of [[index|Two Pools]] · Prev: [[module-2-calibration]] · Next: [[module-4-vram-budget]]

> [!info] What this module is for
> Every module before this one measured hardware: bandwidth, ratios, tokens per second. None of that tells you whether a config change makes your local model *better at coding*, which is presumably the actual point of running one on this workstation at all. This module builds the instrument that answers that question — Aider’s Polyglot benchmark, run in Docker against your own local endpoint — and then does the one thing almost every local-LLM benchmark blog post skips: it measures the instrument’s own noise before using it to compare anything. You will run one identical configuration three times and compute the standard deviation across those three runs. That number is not a footnote — it’s the single most important methodological result in this course. Any later difference between two configs that’s smaller than this sigma is not a result, it’s noise, and treating it as a result is the most common mistake in this entire space. Establishing that noise floor now, before you’ve spent a single hour comparing configs, is nearly free credibility that most public local-LLM benchmarks never bother to earn.

---

## 3.1 Stand up the local endpoint

> [!abstract] Goal
> Get an OpenAI-compatible endpoint serving a coding model from your Module 1 llama.cpp build, reachable from a Docker container.

**What’s going on:** Aider (and its benchmark harness) talks to models through an OpenAI-compatible chat completions API — it doesn’t need to know or care that the model is running locally, as long as something answers at the expected URL shape. llama.cpp’s `llama-server` binary provides exactly that: point it at a GGUF file, and it exposes a local HTTP server with an OpenAI-compatible `/v1/chat/completions` route.

**Steps:**
1. Start the server using your Module 1 HIP build and a coding model from Module 2 (the 7B or 14B Qwen2.5-Coder model is a reasonable starting point — full VRAM residency, known η_model):
   ```bash
   ./build-hip/bin/llama-server -m qwen2.5-coder-7b-instruct-q4_k_m.gguf \
       -ngl 99 -t <your-elbow-thread-count> --host 0.0.0.0 --port 8080
   ```
2. Confirm it answers on the host itself first:
   ```bash
   curl http://localhost:8080/v1/models
   ```
3. If your benchmark run will happen inside Docker, confirm the container can reach the host’s port — on Linux this typically means running the container with `--network host`, or using the host’s LAN IP rather than `localhost`/`127.0.0.1` from inside the container, since `localhost` inside a container refers to the container itself, not the host.

> [!success] Done when
> A `curl` request to `/v1/chat/completions` against your local `llama-server`, issued from wherever your benchmark container will actually run from, returns a valid completion.

> [!question]- It’s not working
> 1. Getting connection refused from inside a container? That’s almost always the container-networking gotcha above — verify with `--network host` or the host’s real IP, not `localhost`, from inside the container.
> 2. Server starts but every request times out? Check the model actually finished loading (watch the server’s own startup log) before sending requests — a large model can take a while to memory-map and offload.

## 3.2 Define the stratified subset

> [!abstract] Goal
> Build a fixed, reproducible subset of roughly 30 exercises out of Aider Polyglot’s full 225, chosen to represent the same spread of languages the full set covers, and reserve the full 225 for headline numbers only.

**What’s going on:** Aider Polyglot’s exercises are organized per programming language, and the full set is large enough (225 exercises, multi-turn, at local decode speeds) that running it once per configuration is not a realistic iteration loop — a single full run is plausibly 8-15 hours of wall-clock inference time, and this course compares multiple configurations, which makes full runs per config completely infeasible for day-to-day iteration. The fix is a stratified subset: sampled proportionally across whatever languages the full benchmark actually contains, so the subset’s difficulty and language mix roughly mirrors the full set’s, rather than accidentally testing only the easiest language or the hardest one. Do not assume a fixed layout or exercise count from memory or from an older README — the benchmark’s exact directory structure and exercise counts have changed across Aider releases, so enumerate what you actually have before sampling from it.

**Steps:**
1. Clone Aider and locate its benchmark harness and the Polyglot exercise set it pulls in:
   ```bash
   git clone https://github.com/Aider-AI/aider.git
   cd aider/benchmark
   ```
2. Enumerate what’s actually present — don’t assume; count it:
   ```bash
   find . -maxdepth 2 -type d | sort
   ```
   Identify the top-level grouping (per-language directories, or a manifest file listing each exercise’s language) and get an exact count of exercises per language.
3. Compute a proportional sample size per language: for each language with `N_lang` exercises out of `225` total, target `round(30 * N_lang / 225)` exercises from that language, adjusting the last language’s count up or down by a couple of exercises so the total lands at (or very near) 30.
4. Within each language, pick that many exercises deterministically — e.g., sorted alphabetically and evenly spaced (every k-th exercise) rather than randomly, so the exact same subset is reproducible on a rerun without needing to store a random seed.
5. Write the resulting exercise list to a plain text or JSON manifest file you check into your own notes — this file *is* your subset definition from here forward, and every "run the subset" invocation in this module and later modules should reference it explicitly.

> [!success] Done when
> You have a written manifest of ~30 exercise names, proportionally sampled across every language the full 225-exercise set actually contains, saved somewhere you can point the benchmark harness at deterministically.

> [!question]- It’s not working
> 1. Can’t find a clean per-language directory structure? Check the harness’s own benchmark configuration or a `dirs.txt`/similar manifest it generates — the exact layout is a moving target across Aider versions, so trust what you find on disk over what any older guide (including this one) describes.
> 2. Proportional math not landing on exactly 30? That’s fine — 30 is a target, not a hard requirement; document your actual final count and move on.

## 3.3 Run the identical config three times

> [!abstract] Goal
> Run your ~30-exercise subset against the same model, same quantization, same everything, three separate times, with nothing intentionally changed between runs.

**What’s going on:** This is the whole point of the module. If you ran the subset once and reported "62% pass rate," you would have a single sample from a noisy process — sampling temperature, any nondeterminism in the local server’s batching/scheduling, and even small timing-dependent differences in multi-turn conversation state can all shift a run’s outcome from the next one, even with identical configuration. Running the exact same thing three times is how you find out how much that noise actually is, in your own setup, on your own hardware — not a number borrowed from someone else’s system.

**Steps:**
1. Point the benchmark harness at your subset manifest from 3.2 and your local endpoint from 3.1. The exact invocation depends on the current version of `aider/benchmark/benchmark.py` — check its `--help` output, but the general shape is pointing an OpenAI-compatible base URL and a model identifier at your `llama-server`:
   ```bash
   export OPENAI_API_BASE=http://<host-ip>:8080/v1
   export OPENAI_API_KEY=sk-no-key-required
   python benchmark.py --model openai/local-qwen2.5-coder-7b \
       --exercises-file /path/to/your/subset-manifest.txt \
       --num-tests 1
   ```
2. Record the pass rate and total wall-clock time for that run.
3. Repeat the identical invocation two more times, changing nothing — same model, same manifest, same server process if possible (restart it between runs only if you must, and note if you did).
4. Tabulate all three pass rates and all three wall-clock times side by side.

> [!success] Done when
> You have three pass-rate numbers and three wall-clock times, from three otherwise-identical runs of your subset, recorded in one table.

> [!question]- It’s not working
> 1. Does the benchmark harness’s exact flag names not match what’s shown above? Aider’s benchmark CLI has changed across releases — run `python benchmark.py --help` and adapt; the shape (point it at an OpenAI-compatible endpoint and a set of exercises) is stable even when flag spelling isn’t.
> 2. Are results wildly different between runs (not just noisy, but categorically different, e.g. one run failing to connect halfway through)? Check your local server didn’t crash or get OOM-killed mid-run — a partial run isn’t a noise data point, it’s a broken run; discard and rerun it.
> 3. Is each run taking far longer than expected? Multi-turn exercises mean the model may be prompted several times per exercise (initial attempt, then retries against test failures) — this is expected behavior for Aider’s benchmark loop, not a hang, but it’s also exactly why the full 225-exercise set is a multi-hour undertaking.

## 3.4 Compute the noise floor

> [!abstract] Goal
> Turn the three runs from 3.3 into a standard deviation across pass rate, and write down the rule this number enforces for every later comparison.

**What’s going on:** Standard deviation across three identical runs gives you a first, honest estimate of how much a pass-rate number can move on its own, with nothing about the configuration changed. This is not a rigorous statistical confidence interval — three samples is a small n — but it is vastly better than the zero runs of noise-checking that most local-LLM benchmark write-ups do before publishing a single comparison table. The rule that falls out of this number is blunt and easy to apply: **if a later config change moves the pass rate by less than this sigma, you cannot claim that change helped or hurt — you can only say the two configs were indistinguishable at this sample size.**

**Steps:**
1. Compute the mean and standard deviation of the three pass rates from 3.3.
2. Compute the mean and range (or standard deviation) of the three wall-clock times as well — cost-per-run matters for planning later comparisons, independent of pass-rate noise.
3. Write down, in one sentence, the threshold this sigma implies: e.g., "any config comparison in this course must show a pass-rate difference greater than [X] percentage points to be treated as a real effect, given this baseline’s measured run-to-run variance."
4. Keep this sentence visible — literally paste it at the top of whatever tracking document you use for later config comparisons.

> [!success] Done when
> You have a baseline mean pass rate, a standard deviation across the three identical runs, and wall-clock cost per run — three concrete numbers, plus a written threshold statement that later modules are obligated to respect.

> [!bug] Gotcha
> Don’t be tempted to skip straight to comparing configs because three runs "felt" consistent. Write the actual number down, even if the three runs looked identical — a felt sense of consistency is not the same as a computed standard deviation, and the entire value of this exercise is having a number to point to later, not a vague impression.

## 3.5 Plan around the 225-exercise wall

> [!abstract] Goal
> Confirm, on paper, why full 225-exercise runs are reserved for headline numbers only, and are infeasible as your iteration loop.

**What’s going on:** A documented real-world local run of a similarly-sized MoE model on laptop-class hardware took roughly 50 hours of inference wall-clock time for the full 225-exercise set; a 25-exercise subset fit comfortably inside a 4-hour window on the same hardware. Even accounting for your workstation likely being faster than a laptop-class machine, the arithmetic is the same shape: 225 exercises, multi-turn, at local decode speeds, is plausibly an 8-15 hour undertaking per full run. If this course later compares roughly 8 different configurations (quantization levels, offload strategies, context sizes — whatever later modules introduce), 8 full runs at even the low end of that range is well over 60 hours of pure inference time, before you’ve accounted for any reruns needed to also measure *their* noise floors. That schedule doesn’t survive contact with reality unless the subset from 3.2 is doing almost all of the iteration work, with full 225-exercise runs held back for the final, headline comparisons only.

**Steps:**
1. Using your own 3.3 wall-clock time for the ~30-exercise subset, extrapolate a rough full-225 estimate (scale linearly by exercise count as a first approximation, acknowledging multi-turn retries may not scale perfectly linearly).
2. Multiply that estimate by the number of configurations you expect to compare across the rest of this course.
3. Write down the resulting total, and confirm for yourself that it is not something you can casually re-run on a whim — this is the number that justifies treating the subset as your default tool and the full set as a reserved, occasional one.

> [!success] Done when
> You have your own extrapolated full-225 wall-clock estimate, and a written total across your expected number of configuration comparisons, confirming — with your own numbers, not borrowed ones — that the subset from 3.2 has to carry the iteration workload for the rest of this course.

> [!question]- It’s not working
> 1. Does your subset’s wall-clock time seem implausibly fast compared to the 8-15 hour full-set estimate even after scaling? Multi-turn retry behavior means full-run time doesn’t always scale perfectly linearly with exercise count — treat your extrapolation as a planning estimate, not a guarantee, and revisit it once you’ve actually run a full 225-exercise pass later in the course.

---

## Check your understanding

> [!question]- 1. Why does this module insist on running the identical configuration three times before comparing anything?
> > [!success]- Answer
> > Because a single run’s pass rate is one sample from a noisy process — sampling nondeterminism, local server batching/scheduling variance, and other small run-to-run differences can shift the outcome even with nothing about the configuration changed. Running the same setup three times produces a standard deviation, which tells you how much the number can move on its own. Without that number, you have no way to tell a real config effect apart from ordinary run-to-run noise, which is exactly the failure mode most local-LLM benchmark posts fall into.

> [!question]- 2. Your three baseline runs come back at 60%, 63%, and 58% pass rate. A later config change moves the pass rate to 64%. Can you claim that change helped?
> > [!success]- Answer
> > Not on this evidence. A spread of 58-63% across three identical runs implies real run-to-run noise on the order of a few percentage points — a single new run at 64% falls well within that same noisy range, and could easily be indistinguishable from the baseline rather than a genuine improvement. You’d need either a difference clearly larger than the measured standard deviation, or multiple repeated runs of the new configuration (the same three-run treatment given to the baseline), before treating 64% as a real effect rather than noise.

> [!question]- 3. Why is the ~30-exercise stratified subset sampled proportionally by language rather than just picking the first 30 exercises alphabetically or by file order?
> > [!success]- Answer
> > Because the full 225-exercise set spans multiple programming languages, and picking exercises without regard to that structure risks a subset that accidentally over-represents one language’s difficulty profile (or omits some languages entirely) while under-representing others — an unrepresentative subset can make a config look better or worse than it really is simply because of which languages happened to get sampled. Proportional sampling keeps the subset’s language mix close to the full set’s, so a pass-rate difference on the subset is more likely to reflect a real difference in coding capability rather than an artifact of which exercises got included.

## What’s next

You now have a working local benchmark harness, a defined and reproducible subset, and — most importantly — a measured noise floor that every later comparison in this course has to clear before it counts as a real result. That’s the full toolkit this course set out to build: two measured memory pools, a calibrated throughput model, and an evaluation instrument with a known sigma. Later modules use all three together to compare real configuration choices — quantization levels, offload splits between the two pools, context-length trade-offs — against this same subset, and hold every claimed improvement to the bar this module just established.

<script src="/tutor.js" defer></script>
