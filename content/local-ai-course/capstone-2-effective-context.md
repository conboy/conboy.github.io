---
title: "Capstone 2 — Your Effective Context"
description: "Two independent walls — the VRAM ceiling you can afford and the retrieval-and-use accuracy that actually survives long context — converging on your own effective context length, measured against a code-shaped NoLiMa-style probe."
tags:
  - local-ai
  - capstone
  - kv-cache
  - nolima
  - effective-context
---

# Capstone 2 — Your Effective Context

**2 weeks** · Part of [[index|Two Pools]] · Prev: [[capstone-1-predicted-machine]]

> [!info] What this capstone is for
> Capstone 1 answered "how fast," and it answered it well — a predicted machine, checked against measurement, with the gaps explained rather than hidden. This capstone answers a colder question the rest of the course has been circling since [[index|the very first page]]: when you set `-c 262144` or reach for the marketed 1M-token window, how much of that context can your model actually *use*? Two walls converge on the answer, and they are genuinely independent of each other — one is a hard physical limit on what fits in 16 GB, the other is a soft, model-behavior limit on what the model actually retrieves and correctly uses once it’s in there. Neither wall is the number on the model’s card. This capstone builds a code-shaped retrieval probe in the spirit of NoLiMa, runs it across your own quant levels and context lengths, and produces the one number this entire course was ultimately building toward: your own effective context length, stated plainly against both the model’s 262,144-token native maximum and its marketed 1,000,000-token extension.

---

## C2.1 Wall 1 — what you can afford

> [!abstract] Goal
> Establish the hard VRAM ceiling on context length before touching a single retrieval-quality question, and draw it as a fixed vertical line you’ll never be able to move past on this card.

**What’s going on:** [[module-4-vram-budget|Module 4]] already gave you the number, and it’s worth restating starkly here because this capstone is where its consequence actually lands: at this model’s native `262,144`-token context, the KV cache alone — at FP16, the format that preserves the most quality — costs **25.77 GB**. The RX 9070 XT has **16 GB** of VRAM, total, before a single model weight is loaded. The KV cache at native context doesn’t just compete with the weights for space — it exceeds the entire card by nearly 10 GB on its own. **You physically cannot reach this model’s native context length with an FP16 KV cache on this GPU, full stop, independent of quantization level, `n_cpu_moe` setting, or anything else this course has taught you to tune.**

[[module-9-kv-quantization|Module 9]] offers a lever: quantizing the KV cache to `q4_0`/`q4_0` cuts its footprint by roughly 72%, bringing 262,144-token context down to somewhere near **7.2 GB** — a number that finally fits alongside the model’s weights. But Module 9’s own thesis was specific and uncomfortable: **`q4_0` KV-cache quantization is exactly the lever most likely to break agentic tool-calling at long context**, and it’s strongly model-dependent — safe on one model, a near-total collapse on another, with no way to know which you have without testing it yourself. The VRAM wall doesn’t just tell you what fits. It tells you that the *only* way to physically reach anywhere near native context on this card runs directly through the one quantization setting this course already flagged as the highest-risk lever available.

**Steps:**
1. Recompute the KV-cache size at 262,144 tokens (native) and at 1,000,000 tokens (marketed) using the [[module-4-vram-budget|Module 4]] formula and your own model’s figures — confirm 25.77 GB and roughly 103 GB respectively, independently rather than copying these numbers.
2. Recompute the `q4_0`/`q4_0` footprint at native context using your [[module-9-kv-quantization|Module 9.2]] reduction figure, and confirm it lands near 7.2 GB.
3. Subtract your model’s non-expert weight size and your Module 4.4 overhead constant from 16 GB, and confirm how much of the remaining budget a 7.2 GB quantized KV cache at native context actually leaves for resident expert layers — state explicitly whether *any* `n_cpu_moe` value makes this configuration fit at all, or whether native context at any KV precision forces every single expert layer off the card.

> [!success] Done when
> You’ve independently confirmed the FP16 KV-cache size at native and marketed context, the `q4_0`/`q4_0` reduced size at native context, and stated plainly whether or how a legal `n_cpu_moe` configuration exists that fits native context on this card at all.

> [!bug] Gotcha
> Don’t let "it technically fits at `q4_0`/`q4_0`" read as "so native context is fine." Fitting in VRAM and producing a model that still works correctly at that context length are two different claims, and this capstone’s whole second half exists because the first one doesn’t imply the second.

## C2.2 Wall 2 — what actually works

> [!abstract] Goal
> Understand NoLiMa’s definition of effective length precisely, and know its headline finding well enough to explain why a context-length number on a model card is close to meaningless on its own.

**What’s going on:** NoLiMa (arXiv 2502.05167, ICML 2025) defines **effective length** as the longest context at which a model retains **at least 85%** of its own short-context baseline retrieval score. That’s a deliberately relative definition — it doesn’t ask "can the model find the needle at all," it asks "how much has performance already degraded by the point you’re testing," which is the more honest question for anyone actually planning to rely on a given context length.

NoLiMa’s headline result is worth holding onto precisely because of how large the gap it exposes is: **Llama 4 Scout’s effective length measured 1,000 tokens, against a marketed 10,000,000** — a base short-context score of 81.7, falling to 72.3 at 1K, 61.8 at 2K, 50.8 at 4K, 35.5 at 8K, 26.9 at 16K, and 21.6 at 32K. By 32K tokens — 0.32% of the marketed window — the model retains barely a quarter of its own short-context capability. Across the 22 models NoLiMa tested, **10 of the 12 models marketed for long context fell below 50% of their own base score by 32K tokens.** This is not a one-model anomaly; it’s the modal outcome for long-context-marketed models as a class.

This is **someone else’s measurement**, on general-purpose retrieval tasks, not coding tasks, and not this course’s model. That gap — a general-purpose finding with no code-specific equivalent — is exactly what the rest of this capstone fills for your own setup.

**Steps:**
1. Restate NoLiMa’s effective-length definition from memory — the 85%-of-base threshold specifically, not "can it still answer at all."
2. Write down the Llama 4 Scout numbers from memory (or from your notes) and state, in your own words, why "1,000 tokens effective against a marketed 10,000,000" is a more damning finding than "1,000 tokens effective" reported alone, without the marketed number beside it.
3. Note explicitly that no published NoLiMa-style number exists yet for this course’s model, at your quant and context settings — that gap is what C2.3 through C2.5 exist to close.

> [!success] Done when
> You can state NoLiMa’s 85%-of-base definition and the Llama 4 Scout numbers from memory, correctly labeled as another team’s measurement on a different model and task family, and you can explain precisely what’s missing that your own probe needs to supply.

## C2.3 Design a code-shaped needle

> [!abstract] Goal
> Build a retrieval-and-use probe specific to coding, since NoLiMa itself is general-purpose and no published code-shaped equivalent exists — and design it to score *use*, not just retrieval.

**What’s going on:** A generic needle-in-haystack test plants an arbitrary fact ("the special magic number is 4721") and asks the model to recite it back later. That measures retrieval, and retrieval alone is already informative — but it understates what a coding agent actually needs, because a coding agent doesn’t just need to *recall* a fact from earlier in context, it needs to *correctly use* it: call a function with the right argument order, respect a type signature, apply a config value in the generated code rather than just quoting it back in prose.

Design your needle around exactly that gap:

- **Plant** a real-looking artifact N tokens back in a long, realistic distractor context — a plausible codebase or a long conversation transcript with genuine surrounding content, not an obviously artificial wall of filler text. Good candidates: a function definition with an unusual parameter order or an uncommon default value, a type signature with a specific, arbitrary generic parameter, or a config constant with a distinctive, non-guessable value.
- **Query**, near the end of the context, in a way that requires the model to do something with the planted artifact, not just repeat it: write a call site that correctly invokes the planted function with its actual signature, or generate code that correctly uses the planted config value, or extend a class using the planted type signature correctly.
- **Score two things separately, not one combined pass/fail:**
  1. **Retrieval accuracy** — did the model’s answer correctly state the planted fact (the right parameter names, the right default value, the right constant) when asked to recall it directly?
  2. **Use accuracy** — independently, did the model’s *generated code*, when it had to actually apply the fact rather than recite it, get it right? A model can pass retrieval and fail use (it can quote the signature back correctly but still write a call site that gets an argument order wrong), and that distinction is exactly the kind of failure a plain needle-in-haystack test would miss entirely.

**Steps:**
1. Build at least 15–20 needle instances, each with a genuinely distinct planted artifact (different function names, different parameter shapes, different constants) — reusing the same needle repeatedly risks the model pattern-matching on the test’s own structure rather than genuinely retrieving from context.
2. For each instance, write both a direct-recall query (measuring retrieval) and a use-requiring query (measuring use) against the same planted artifact.
3. Pilot the probe at a short context length (e.g. 2K–4K tokens) first, to confirm your scoring rubric actually distinguishes correct from incorrect answers cleanly before you scale it up to long context — a probe that’s ambiguous to score at short context will only get harder to score reliably as context grows.

> [!success] Done when
> You have a working code-shaped needle probe — at least 15–20 distinct instances, each scored separately for retrieval accuracy and use accuracy — validated for scoring clarity at a short pilot context length before scaling up.

> [!question]- It’s not working
> 1. The model scores well on retrieval but your use-accuracy scoring feels inconsistent or ambiguous? Tighten the use-query so there’s an unambiguous, mechanically checkable correct answer — e.g., code that either compiles/type-checks against the planted signature or doesn’t, rather than a judgment call about code "quality."
> 2. Short-context pilot scores are already low, before you’ve even introduced long-context degradation? Fix your baseline task difficulty first — a probe that’s already hard at 2K tokens can’t cleanly show degradation as context grows, since you won’t be able to tell a real long-context effect apart from noise in an already-unreliable baseline.

## C2.4 Run the sweep: context length × KV quantization

> [!abstract] Goal
> Run your code-shaped needle probe across a range of context lengths and the same KV-cache quantization levels from [[module-9-kv-quantization|Module 9]], producing retrieval and use accuracy curves for each.

**What’s going on:** This is the measurement Module 9.6 built a smaller version of for generic tool-calling; this capstone’s version is code-specific and scored on the retrieval/use split from C2.3. Run it across enough context lengths to actually see a curve, not just two endpoints.

**Steps:**
1. Choose a spread of context lengths that spans from clearly-short to as close to your practical ceiling as C2.1’s VRAM wall allows — for example 2K, 8K, 32K, 64K, and 128K, plus whatever your quantized-KV configuration lets you reach closer to native context.
2. Run the full C2.3 probe battery at each context length, at four KV-cache configurations matching [[module-9-kv-quantization|Module 9.6]]: `f16`/`f16` baseline, `q8_0`/`q8_0`, `q4_0`/`q4_0`, and K-only `q8_0` with V left at `f16`.
3. Record retrieval accuracy and use accuracy **separately** at every context length × KV-configuration cell — do not collapse them into one score, since the whole point of C2.3’s design is to see them diverge.
4. Compute, for each KV configuration, the context length at which **both** retrieval and use accuracy first drop below 85% of their own short-context (e.g. 2K) baseline score — this is your NoLiMa-style effective length, one per KV configuration, and it may differ meaningfully between retrieval and use.

> [!success] Done when
> You have retrieval-accuracy and use-accuracy curves across your full range of context lengths, at all four KV-cache configurations, with an explicit 85%-of-baseline effective length computed for each configuration — and for retrieval and use separately if they diverge.

> [!question]- It’s not working
> 1. Use accuracy drops off far earlier than retrieval accuracy at every configuration? That’s a real, informative result, not a bug — it would mean the model can still find the fact in a degraded long context but is losing the ability to correctly apply it, which is exactly the gap a generic (non-code) NoLiMa-style probe would never surface. Report it as a genuine finding.
> 2. `q4_0`/`q4_0` shows little to no degradation beyond what `f16`/`f16` shows at the same context length, contradicting the model-dependent-collapse framing from Module 9.4? Report your own number as your own finding, the same way Module 9.6’s troubleshooting told you to — "must be benchmarked per-model" cuts both ways, including toward a better-than-expected result.

## C2.5 One figure, both walls

> [!abstract] Goal
> Combine the C2.1 VRAM ceiling and the C2.4 accuracy-degradation curves into a single figure that makes the entire capstone legible at a glance.

**What’s going on:** Neither wall alone tells the full story. The VRAM ceiling says "you can’t get there physically" past a certain point, at a given KV precision. The accuracy curve says "even where you can get there physically, the model may have already stopped being useful" well before you hit that physical limit. Putting both on one figure is what makes the honest answer to "can I run this at 1M context" visually unavoidable rather than something that has to be explained in a paragraph.

**Steps:**
1. Plot context length on the x-axis (log scale is likely necessary, given the span from 2K to 1M) and accuracy (as a fraction of short-context baseline) on the y-axis.
2. Plot retrieval-accuracy and use-accuracy curves for at least your `f16`/`f16` baseline and your most memory-efficient legal configuration from C2.1.
3. Draw a horizontal line at 85% — the NoLiMa effective-length threshold — and mark where each curve crosses it.
4. Draw a **vertical line** at the VRAM ceiling from C2.1: the longest context length actually reachable on this card at a given KV precision (mark both the FP16 ceiling and the `q4_0`/`q4_0` ceiling, since they’re very different numbers).
5. Mark two more vertical reference lines: this model’s native 262,144-token maximum, and its marketed 1,000,000-token extension.

> [!success] Done when
> You have one figure showing accuracy-vs-context-length curves, the 85% threshold line, at least two VRAM-ceiling vertical lines (FP16 and your best quantized configuration), and the native/marketed context reference lines — all on the same axes.

## C2.6 State your number

> [!abstract] Goal
> Say, plainly and specifically, what your own effective context length is — against both the native and the marketed maximums — with no hedging left in the final statement.

**What’s going on:** This is the capstone’s actual deliverable, and it deserves to be stated as a single, clean sentence rather than buried in a table: *at my measured KV-cache configuration, my model’s effective context length — the point where retrieval-and-use accuracy first drops below 85% of its short-context baseline — is approximately N tokens, against a native maximum of 262,144 and a marketed extension of 1,000,000.*

**Steps:**
1. Write that sentence, with your own real number substituted in, for your best (most usable) KV-cache configuration.
2. Compute the ratio of your effective length to both the native maximum and the marketed maximum, and state both percentages explicitly.
3. State which wall — VRAM capacity or accuracy degradation — is the tighter constraint *for your specific setup*. This may not be the same answer for every reader of this course; say which one binds first for you, specifically, and why.

> [!success] Done when
> You have a single, specific, unhedged statement of your own effective context length against both the 262,144 native maximum and the 1,000,000 marketed maximum, with the ratio to each stated as a percentage, and an explicit answer to which wall binds first on your own configuration.

---

## Check your understanding

> [!question]- 1. Why does NoLiMa define effective length relative to a model’s own short-context score (≥85% of baseline), rather than as an absolute retrieval-accuracy threshold?
> > [!success]- Answer
> > Because an absolute threshold conflates two different things: how good a model is at the task in general, and how much that specific capability degrades as context grows. A relative definition isolates the second question — it asks "how much of *this model’s own* baseline capability survives at this context length," which is exactly what someone deciding whether to trust a given context length needs to know, independent of whether the model was strong or weak to begin with.

> [!question]- 2. Your code-shaped probe shows retrieval accuracy holding up well past the point where use accuracy collapses. What does this tell you that a generic (non-code) NoLiMa-style probe never could?
> > [!success]- Answer
> > It tells you that "the model can still find the fact" and "the model can still correctly act on the fact" are separate capabilities that degrade at different rates under long context — and for a coding agent, the second one is the one that actually matters, since a coding assistant that can recite a function signature but call it incorrectly is not meaningfully more useful than one that forgot the signature entirely. A generic retrieval-only probe, by only ever asking the model to recite the planted fact, would report the retrieval curve alone and miss this gap completely — exactly the kind of code-specific finding this capstone’s C2.3 design exists to surface.

> [!question]- 3. The model technically fits a 262,144-token context in 16 GB of VRAM once the KV cache is quantized to q4_0/q4_0. Does that mean native context is a legitimate, usable configuration?
> > [!success]- Answer
> > Not on its own — fitting in VRAM answers Wall 1, not Wall 2. Module 9 already established that q4_0 KV-cache quantization is the highest-risk lever available for agentic tool-calling, and this capstone’s own C2.4 measurement is what determines whether retrieval-and-use accuracy has already collapsed below the 85% threshold well before 262,144 tokens, regardless of whether the bytes physically fit. A configuration that fits in VRAM but has already fallen below its own effective length is not usable in practice — it’s merely loadable, which is a different and much weaker claim.

## Closing: what you actually know now

You started this course with a fair question: can a single desktop run a serious coding model, locally, well enough to trust what comes back? You now have a genuinely earned answer, built from your own measurements rather than borrowed from anyone’s marketing copy or anyone’s forum post.

Concretely, and specifically — not abstractly — you now know: your own sustained DDR5 and VRAM bandwidth, and the calibration factor between what the spec sheet promised and what your machine actually delivers. Your own roofline curve for an RX 9070 XT running MoE expert offload, likely the first one published for this GPU. An Aider Polyglot harness with a known run-to-run sigma, so you can tell a real result from noise instead of guessing. A VRAM budget allocator that turns a target context length into the right `--n-cpu-moe` value without redoing the arithmetic by hand. A measured prefill/decode crossover and a backend decision rule, both stated as plain, falsifiable sentences someone else could apply without re-running your sweep. A quantization Pareto front with a genuine floor, not a guess, and a KV-cache quantization danger you reproduced yourself rather than took on faith. Tool-call failure rates across three real protocols, and a measured seconds-of-TTFT tax for two different coding harnesses’ architectural choices. A predicted machine, checked against three configurations you had never run before you ran them — including two models roughly four times this course’s own main model — with the prediction error stated honestly either way. And now, finally, your own effective context length, against both the number on the model’s spec page and the number in its marketing copy, with both walls that produced it drawn on the same figure.

That last one is worth sitting with, because it’s the honest answer to the question that started this whole project. Can you have a 1M-token context, locally, on this workstation? No. You cannot, and you now know exactly why, in two independent ways: the KV cache alone would need more memory than the card contains before a single weight loads, and — likely far before that ceiling would even matter — the model’s own ability to retrieve and correctly use a fact planted deep in that context has almost certainly already collapsed well below a threshold you’d trust for real work. Neither of those is a criticism of this specific model, or of local inference generally. Every context-window number on every model card, hosted or local, is measured the vendor’s way, on the vendor’s chosen task, and this course’s whole final lesson is that the gap between that number and your own reality is not a rounding error — it’s usually the entire story. You didn’t take anyone’s word for where that gap sits on your machine. You measured it.

<script src="/tutor.js" defer></script>
