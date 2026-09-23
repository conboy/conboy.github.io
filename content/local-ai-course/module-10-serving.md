---
title: "Module 10 — Serving and the Tool-Call Contract"
description: "A model-swapping reverse proxy measured for swap latency and VRAM reclaim, and a controlled shootout of three tool-call protocols — including a homemade failure-rate curve for a claim AMD made about small models."
tags:
  - local-ai
  - serving
  - tool-calling
  - gbnf
  - llama-swap
---

# Module 10 — Serving and the Tool-Call Contract

**9 hours** · Part of [[index|Two Pools]] · Prev: [[module-9-kv-quantization]] · Next: [[module-11-aider]]

> [!info] What this module is for
> Every module so far has run a single `llama-server` process, loaded once, serving one client. That’s an honest way to measure a roofline, but it isn’t how a real workstation gets used — you switch between a coding model and something smaller for quick questions, and whatever’s serving the request needs to actually understand the tool calls your agentic harness sends it. This module has two genuinely separate halves, and both end in a real measurement rather than an impression. The first half treats **the serving layer itself** as a variable: `llama-server` is strictly one model per process, and `llama-swap` sits in front of it as a reverse proxy that swaps backends automatically — worth measuring for swap latency and VRAM reclaim, both real costs in a multi-model workflow, not free conveniences. The second half is the harder practical problem underneath every agentic coding setup: **how does a tool call actually get from the model’s output into working code**, across three genuinely different protocols, and does AMD’s own published claim about which model sizes can be trusted with tool-calling hold up when you test it yourself instead of citing it.

---

## 10.1 The serving problem: one process, one model

> [!abstract] Goal
> Understand exactly what `llama-server` does and does not do when you need more than one model available.

**What’s going on:** `llama-server` loads one model into one process and serves that model until the process exits. If you want a second model — a smaller, faster one for quick non-coding questions, say — you either run a second process (and juggle the VRAM budget between two resident models yourself) or you kill the first process, load the second, and eat the full model-load time on every switch. Neither is what an interactive workflow actually wants: you want to point one endpoint at whichever model a given request needs, and have the right backend already running (or spun up on demand) without babysitting processes by hand.

**Steps:**
1. Confirm this limitation directly: start `llama-server` with one model loaded, then try pointing a second client request at a different model name on the same port — confirm it either errors or silently ignores the model field and serves whatever’s already loaded.
2. Note, from your own Module 2 model-load-time measurement, how expensive a manual "kill and reload" cycle would be if you did this by hand every time you switched models during a work session.

> [!success] Done when
> You’ve confirmed `llama-server`’s one-process-one-model behavior directly, and you have your own model-load-time figure as the baseline cost 10.2 is trying to reduce.

## 10.2 `llama-swap`: measuring swap latency and VRAM reclaim

> [!abstract] Goal
> Set up `llama-swap` in front of two or more backend configurations, and measure the two real costs of automatic model switching: swap latency and VRAM reclaim time.

**What’s going on:** `llama-swap` is an MIT-licensed, Go-based reverse proxy shipped as a single binary plus one YAML configuration file. It reads the `model` field out of an incoming OpenAI-format request, matches it against a configured backend, and starts (or restarts) the right `llama-server` process automatically — Ollama-style convenience layered on top of whatever backend you point it at, not tied to any one inference engine. That convenience is not free: swapping models means stopping one process, waiting for VRAM to actually come back, and starting the next process’s load sequence, and all of that shows up as real, measurable latency on whichever request triggers the swap.

**Steps:**
1. Install `llama-swap` and configure it with at least two backend entries — your usual Qwen3-Coder-30B-A3B configuration, and a second, smaller model you can load quickly (a good use for the dense baseline model from [[module-2-calibration]]).
2. Send a request for model A, then immediately a request for model B, and time the full round-trip of the second request — this includes the swap, not just inference, so expect it to look nothing like your usual first-token latency.
3. Watch VRAM usage (via a system monitor, or `rocm-smi`-equivalent tooling) through the swap window specifically — measure how long it takes from "process A killed" to "VRAM fully reclaimed and available for process B’s allocation," since these are not always the same instant.
4. Repeat the A→B→A cycle several times and report mean and stddev swap latency — a single sample here is not enough to trust, the same discipline as every other benchmark in this course.

> [!success] Done when
> You have a measured mean ± stddev swap latency across several A↔B cycles, and a separate measured figure for how long VRAM reclaim actually takes relative to process termination.

> [!bug] Gotcha
> Behind an `nginx` reverse proxy in front of `llama-swap` itself, you must **disable response buffering**, or streaming (SSE) responses break — `nginx` buffers the entire response before forwarding it by default, which defeats token-by-token streaming and can make a working setup look like it’s hanging. Check `proxy_buffering off` (or the equivalent for your proxy) before concluding a streaming failure is `llama-swap`’s fault.

## 10.3 Native Anthropic Messages API support

> [!abstract] Goal
> Know that `llama-server`, Ollama, and LM Studio now speak the Anthropic Messages API directly, and why that matters for later modules.

**What’s going on:** `llama-server`, Ollama (0.14 and later), and LM Studio (0.4.1 and later) now expose a native `/v1/messages` endpoint speaking the **Anthropic Messages API** format directly — not translated through an OpenAI-compatible shim, but a first-class implementation of the same request/response shape Claude’s own API uses. This matters specifically for [[module-13-claude-code]], where pointing Claude Code itself at a local model becomes a question of endpoint compatibility rather than protocol translation. It’s worth knowing about now, in the serving-layer module, since it’s a property of the serving stack you’re setting up here, not something you’ll need to retrofit later.

**Steps:**
1. Check your `llama-server`, Ollama, or LM Studio version against the versions above, and confirm whether `/v1/messages` is available on your installed build.
2. Send a minimal request directly to `/v1/messages` (not `/v1/chat/completions`) using the Anthropic Messages request shape, and confirm you get a well-formed Anthropic-style response back, without routing through any translation layer.

> [!success] Done when
> You’ve confirmed native `/v1/messages` support on at least one of your serving backends, with a real request/response round trip.

## 10.4 Three tool-call protocols, side by side

> [!abstract] Goal
> Understand the three fundamentally different ways an agentic harness gets a tool call out of a model’s output, before measuring which one is most reliable.

**What’s going on:** This is the hard practical problem underneath every agentic coding setup, and it’s easy to underestimate because "the model calls a tool" sounds like one mechanism when it’s actually three, with real reliability differences between them:

1. **Cline/Roo-style custom syntax** — an XML-ish tool-call format inlined directly in the prompt as instructed text, parsed by the harness itself. This is **not** native JSON function calling — the model is simply instructed, in its system prompt, to emit a specific tag-delimited format, and the harness’s own parser (not the model’s chat template) is responsible for extracting it.
2. **Native JSON tool calling**, via the model’s own chat template — the model emits a structured JSON tool call in the format its training and template define, and the inference server extracts it using template-aware parsing rather than a harness-side regex over freeform text.
3. **GBNF-constrained JSON** — `llama.cpp` masks the logit distribution *before* the softmax at each generation step, so that any token which would violate the supplied grammar is structurally unreachable, not just discouraged. A JSON Schema can be automatically converted to the equivalent GBNF grammar, and "JSON healing" can repair a truncated JSON structure mid-stream if generation is cut off before the object closes.

**Steps:**
1. Confirm, from your agentic harness’s documentation or source, which of these three protocols it actually uses by default — don’t assume; Cline and Roo, specifically, use protocol 1, and that’s worth confirming directly rather than by reputation.
2. Write, in your own words, the practical difference between protocol 2 and protocol 3 — both produce JSON, but only one of them makes malformed JSON structurally impossible to generate in the first place.

> [!success] Done when
> You can name all three protocols, correctly attribute Cline/Roo to protocol 1, and explain the structural difference between "the model was trained to usually emit valid JSON" and "invalid JSON is unreachable by construction."

## 10.5 GBNF grammars: syntax guaranteed, semantics not

> [!abstract] Goal
> Understand precisely what a GBNF grammar does and does not guarantee, so you don’t mistake syntactic validity for correctness.

**What’s going on:** GBNF-constrained decoding is a genuinely strong guarantee about **syntax** — a grammar-constrained model cannot emit a JSON object with a missing brace, an unquoted key, or a value of the wrong basic type, because those tokens are masked out of the logit distribution before sampling ever happens. That’s a real, structural guarantee, not a probabilistic nudge.

It is **not** a guarantee about **semantics**. A grammar-constrained model can still produce perfectly valid JSON that calls the wrong tool, passes the right tool the wrong arguments, or omits an argument the tool actually needs — all while satisfying the grammar completely, because the grammar only knows about JSON structure, not about which tool call is *correct* for the current task. **Grammar guarantees syntax, not semantics** — keep this distinction sharp going into 10.8, where you’ll measure both kinds of failure separately rather than lumping them into one "tool call failed" bucket.

**Steps:**
1. Construct (or find) one example of a syntactically valid but semantically wrong tool call — a well-formed JSON object calling a tool that doesn’t exist, or passing an argument type the tool doesn’t accept even though the JSON itself parses cleanly.
2. State, in one sentence, why a malformed-call-rate metric alone (as used in 10.8) is not a complete measure of tool-calling reliability, and what a second metric would need to check for.

> [!success] Done when
> You can produce or point to a real example of syntactically valid but semantically wrong tool call output, and explain why GBNF alone doesn’t prevent it.

## 10.6 The autoparser and the `--jinja` requirement

> [!abstract] Goal
> Understand how `llama.cpp` extracts tool calls and reasoning text from a model’s raw output automatically, and why `--jinja` is not an optional flag for many current models.

**What’s going on:** `llama.cpp` uses what’s referred to as an **autoparser**: it analyzes a model’s Jinja chat template through differential analysis and automatically generates a PEG parser capable of separating a model’s raw output into its text, reasoning, and tool-call components. This is what lets the server hand back a clean `tool_calls` field in its API response instead of a client having to regex a model’s raw text output by hand.

The flag this depends on is **`--jinja`**, and it is **required**, not optional, for many current models — including Qwen3-Coder-class models. Without `--jinja`, the raw `<tool_call>` XML tags and `</think>` reasoning-closing tags **leak directly into the visible output** instead of being parsed out and structured — a failure mode that looks like a broken model when it’s actually a missing flag.

**Steps:**
1. Start the server once with `--jinja` omitted and once with it explicitly set, sending the same tool-calling prompt to both, and confirm you can see raw `<tool_call>` or `</think>` tags leaking into the response in the first case but not the second.
2. Check your model’s actual chat template (from its `tokenizer_config.json` or equivalent) for tool-call and reasoning tag syntax, so you recognize what a leak looks like for your specific model rather than expecting exactly the tags named above.

> [!success] Done when
> You’ve directly reproduced the tag-leak failure with `--jinja` omitted, and confirmed it disappears with `--jinja` set, on your own model and build.

> [!bug] Gotcha
> If your agentic harness is receiving raw `<tool_call>` XML instead of a structured tool-call response, check `--jinja` before assuming the model itself is broken or the wrong protocol is configured — a missing `--jinja` flag is a much more common cause of this exact symptom than a genuinely incompatible model.

## 10.7 Test AMD’s claim yourself

> [!abstract] Goal
> Reproduce, on your own hardware and your own task battery, AMD’s published claim about which model sizes fail at agentic tool-calling — and produce your own quantitative failure curve instead of citing a qualitative claim secondhand.

**What’s going on:** AMD’s own Cline-partnered testing publicly reported that models smaller than Qwen3-Coder-30B "consistently fail" at agentic tool calling. That’s a citable, public claim — and it’s exactly the kind of claim this course’s ground rules ask you not to take on faith. Turning someone else’s qualitative statement into your own quantitative curve, across a real size range, is one of the highest-value things you can do in this module: a genuinely new data point, not a repeated assertion.

**Steps:**
1. Assemble a small model-size matrix: at minimum, a 7B-class model, a 14B-class model, and Qwen3-Coder-30B-A3B itself, all runnable on this workstation.
2. Build (or reuse from [[module-9-kv-quantization|9.6]]) a tool-calling task battery of at least 30–50 tasks spanning single-turn and multi-turn tool use.
3. Run the identical battery against every model in your size matrix, using the same protocol (native JSON tool calling is the fairest baseline comparison, since it isolates model capability rather than harness-specific prompt engineering) and the same `--jinja`-enabled configuration from 10.6.
4. Record, per model size: malformed-call rate (syntax failures) and semantically-wrong-call rate (valid JSON, wrong tool or arguments) separately, per the 10.5 distinction.
5. Plot failure rate against model size, and state explicitly whether your own curve confirms, partially confirms, or refutes AMD’s "consistently fail below 30B" claim — a partial confirmation (e.g., 14B degrades but doesn’t collapse, 7B collapses outright) is a perfectly good, reportable outcome.

> [!success] Done when
> You have your own measured tool-call failure-rate curve across at least three model sizes, with malformed and semantically-wrong failures reported separately, and an explicit statement of how your curve compares to AMD’s cited claim.

> [!question]- It’s not working
> 1. The smaller models fail at a *much* higher rate than expected, even on trivial single-turn tasks? Confirm `--jinja` is enabled and the model’s chat template actually defines a tool-calling format at all — some smaller instruction-tuned models were never trained with structured tool-calling in mind, which would make this a capability gap rather than a parsing failure.
> 2. All three model sizes perform similarly well? That would be a genuine refutation of the cited claim on your specific task battery — report it as such rather than assuming your test battery must be too easy; consider whether your battery actually stresses multi-turn tool-call closure, which is where smaller models are more likely to degrade than on single-turn calls.

## 10.8 Run the full matrix

> [!abstract] Goal
> Combine protocol choice, quant level, and the 10.7 model-size sweep into one measurement: malformed-call rate per protocol per quant level, plus the token overhead each protocol’s preamble costs.

**What’s going on:** This is the module’s actual deliverable, pulling together everything above into a single comparison table instead of leaving the protocol question, the quant question, and the model-size question as three separate, disconnected results.

**Steps:**
1. Using Qwen3-Coder-30B-A3B at two quant levels from [[module-8-quantization]] (Q4_K_M and one other, e.g. Q6_K), run your tool-calling battery under all three protocols from 10.4: Cline/Roo-style inlined XML syntax, native JSON tool calling, and GBNF-constrained JSON.
2. For each protocol, measure the **token overhead of its preamble** — the extra system-prompt and instruction tokens the protocol requires before the actual task content, since a bulkier tool-call preamble eats directly into your effective context budget from earlier modules.
3. Record malformed-call rate and semantically-wrong-call rate (per the 10.5 distinction) for every protocol × quant combination.
4. Build the final table: rows are protocol × quant combinations, columns are malformed-call rate, semantically-wrong-call rate, and preamble token overhead.
5. Write one paragraph stating which protocol you’d actually recommend for this model and workstation, and under what condition (if any) you’d choose a different one — e.g., GBNF for reliability-critical automation despite its preamble cost, versus the harness’s native protocol when task variety matters more than syntactic guarantees.

> [!success] Done when
> You have a complete table of malformed-call rate, semantically-wrong-call rate, and preamble token overhead across all three protocols at two quant levels, plus a stated recommendation with its reasoning.

---

## Check your understanding

> [!question]- 1. A GBNF-constrained model produces a perfectly valid JSON tool call, but it calls the wrong function. Did GBNF fail?
> > [!success]- Answer
> > No — GBNF did exactly what it guarantees: the output is syntactically valid JSON, because the grammar made any syntax violation structurally unreachable. GBNF says nothing about semantics — it has no way to know which tool call is contextually correct, only that whatever tool call is emitted follows valid JSON structure. Calling the wrong tool with perfectly valid JSON is a semantic failure, a separate category this module tracks independently from malformed-call rate.

> [!question]- 2. Why does `llama.cpp` need the `--jinja` flag specifically, and what does its absence look like in practice?
> > [!success]- Answer
> > `--jinja` enables the autoparser, which analyzes a model’s Jinja chat template and generates a parser that separates raw model output into text, reasoning, and tool-call components. Without it, for many current models including Qwen3-Coder-class models, raw `<tool_call>` XML tags and `</think>` reasoning-closing tags leak directly into the visible output instead of being extracted into a structured API response — a symptom that looks like a broken model but is actually a missing flag.

> [!question]- 3. Your own 10.7 measurement shows a 14B model degrading moderately on tool-calling tasks rather than "consistently failing" the way AMD’s cited claim describes. Is your measurement wrong?
> > [!success]- Answer
> > Not necessarily — a partial confirmation is a legitimate and useful outcome, not evidence of a measurement error. AMD’s claim was made on its own task battery and its own model selection; your own battery, model choice, and protocol may reasonably produce a different shape of result. The value of running this yourself is precisely to find out whether the cited claim’s exact boundary holds on your setup — a moderate-degradation result at 14B, rather than outright collapse, is itself a real, reportable finding about where the boundary actually sits for your configuration.

## What’s next

You now have a measured swap-latency and VRAM-reclaim figure for `llama-swap`, confirmation of native Anthropic Messages API support across your serving stack, a working `--jinja`-enabled autoparser setup, your own quantitative reproduction of AMD’s small-model tool-calling claim, and a full malformed-call-rate table across three tool-call protocols and two quant levels. Every module so far has treated the model and its serving stack as the subject under test. [[module-11-aider]] flips the lens onto the **harness** itself — measuring tokens per turn and repo-map hit rate as the size of a real codebase grows, which is the first place in this course where the client-side tooling, not the model or the GPU, becomes the bottleneck worth measuring.

<script src="/tutor.js" defer></script>
