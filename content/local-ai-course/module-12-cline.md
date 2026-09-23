---
title: "Module 12 — Cline and Roo: The Verbose Harness"
description: "Measuring the token overhead of Cline’s inlined XML tool-call syntax against Aider’s diff format, converted into a seconds-of-TTFT tax on a machine where prefill is expensive."
tags:
  - local-ai
  - cline
  - roo
  - tool-calling
  - harness
---

# Module 12 — Cline and Roo: The Verbose Harness

**7 hours** · Part of [[index|Two Pools]] · Prev: [[module-11-aider]] · Next: [[module-13-claude-code]]

> [!info] What this module is for
> Cline and Roo (Roo is a maintained fork of Cline) are two of the most widely used agentic coding extensions, and both make the architectural choice Module 11 measured Aider avoiding: they get a tool call out of the model by inlining a custom, XML-ish syntax directly into the system prompt as instructed text, then parse the model’s raw output for those tags themselves. That single choice is the main reason Cline and Roo misbehave on local models that weren’t fine-tuned on that exact in-context format — the model has to learn the tag syntax from a few examples in the system prompt alone, every session, with nothing baked into its weights the way a model trained on a native tool-calling format would have. The second thing this module measures is blunter and easier to miss: both harnesses are simply **verbose**. Ten to forty thousand tokens per turn is normal, not a misconfiguration. On a machine where prefill costs real, measured seconds — as [[module-6-prefill-decode|Module 6]] already proved — that verbosity is a direct, quantifiable tax, and this module reuses Module 11’s instrumentation rig to put an exact number on it.

---

## 12.1 The protocol: inlined XML, not native tool calling

> [!abstract] Goal
> Confirm directly, from your harness’s own behavior, that Cline/Roo use protocol 1 from [[module-10-serving|Module 10.4]] — not native JSON function calling — and understand why that’s the harness’s choice, not a limitation of the model you happen to be running.

**What’s going on:** Cline’s system prompt instructs the model to emit tool invocations as tag-delimited text — something in the shape of `<read_file><path>src/main.rs</path></read_file>` — as part of its ordinary text output. The model’s chat template and the server’s function-calling machinery are never involved: this is the harness’s own regex-and-string-parsing layer reading the model’s raw completion text, exactly the same relationship Aider has to a diff block, except the thing being parsed is a made-up XML dialect the model has to imitate correctly rather than a widely-represented format like a unified diff. Roo, as a Cline fork, inherits the same mechanism with its own prompt-template variations.

**Steps:**
1. Open your harness’s system prompt (Cline and Roo both let you view the actual prompt sent, either via a debug/verbose setting or by inspecting the request in your [[module-11-aider|11.3]] logging proxy) and find the literal tag syntax it instructs the model to use.
2. Confirm, from the same captured request, that no `tools` or `functions` field is present in the JSON payload sent to the server — the entire tool-call contract lives in the system-prompt text, not in a structured API field.
3. Compare this against the `tools`/`function_call` field you’d see in a native-JSON-calling request (you saw this shape already in [[module-10-serving|Module 10.4]]) — the absence is the confirmation.

> [!success] Done when
> You’ve inspected a real captured request from your harness and confirmed it carries the tool-call contract entirely as system-prompt text, with no structured tool-calling field in the request body.

## 12.2 Why this specifically breaks on local models

> [!abstract] Goal
> Understand the mechanical reason a model can fail at Cline/Roo’s protocol even when it’s perfectly capable of native JSON tool calling or Aider-style diffs.

**What’s going on:** A model’s chat template and any fine-tuning it received for tool use are built around *some* format — usually whatever the model’s own training data and template define, which for most current instruction-tuned models means a JSON-shaped function call, not this specific inlined XML dialect. Cline/Roo’s prompt has to teach the model this format from scratch, in-context, every single session, relying entirely on the model’s general instruction-following ability rather than anything it was actually trained to produce. A model that’s excellent at native JSON tool calling can still garble the made-up tag syntax, because "follow this XML-ish convention I just described to you in the system prompt" and "emit the tool-call JSON shape you were fine-tuned on" are different skills, and only the second one is something most current open models were explicitly trained for.

**Steps:**
1. Run one identical multi-step task through your harness at two different local models — Qwen3-Coder-30B-A3B and one smaller or differently-trained model you have available — and note whether the smaller model’s failure mode looks like "wrong decision" (chose the wrong file) or "malformed tag" (broke the XML syntax itself, or wrapped it in reasoning text the parser never recognized as a tool call at all).
2. State, in one sentence, why a model can be good at native JSON tool calling and still fail at this specific protocol, tying the answer back to "trained format" vs. "in-context-taught format."

> [!success] Done when
> You’ve reproduced at least one malformed-tag failure directly and can distinguish it, in your own transcript, from a correct-syntax-wrong-decision failure.

## 12.3 Measure verbosity: reuse the Module 11 instrumentation

> [!abstract] Goal
> Point your Module 11 logging proxy at Cline/Roo instead of Aider, and confirm the 10–40K-tokens-per-turn figure directly on your own setup rather than taking it on faith.

**What’s going on:** The logging proxy built in [[module-11-aider|11.3]] doesn’t know or care which client is talking to it — it logs `prompt_tokens`/`completion_tokens` for every request regardless of source. Point Cline or Roo’s OpenAI-compatible base URL setting at the same proxy and the same rig starts logging harness turns instead of Aider turns, with zero changes to the proxy itself.

**Steps:**
1. In Cline/Roo’s settings, set the API provider to "OpenAI Compatible," point the base URL at your Module 11 proxy, and set a placeholder API key — the same shape of configuration as pointing Aider at it, just a different client.
2. Run the same fixed task set (or as close an equivalent as the two harnesses’ different interaction models allow) that you ran through Aider in Module 11.
3. Pull `prompt_tokens` per turn from the log and confirm it lands in, or near, the stated 10–40K range for this class of harness — note explicitly if your numbers land meaningfully outside that range, since that’s itself worth reporting rather than silently adjusting to match the expected figure.

> [!success] Done when
> You have a measured tokens-per-turn distribution for Cline/Roo on your own fixed task set, using the unmodified Module 11 logging rig, confirmed or refuted against the 10–40K-token expectation.

## 12.4 Practical setup notes

> [!abstract] Goal
> Get past the specific configuration failure modes that look like model or harness bugs but are actually setup mistakes, before they cost you debugging time.

**What’s going on:** A short, concrete list, because these are the failures that eat hours if you don’t know to look for them specifically:

- Both Cline and Roo need an **OpenAI-compatible base URL** pointed at your server (or your logging proxy in front of it) — the exact settings-panel field names have shifted across releases, so check your installed version’s current settings UI rather than assuming a specific label from an older guide.
- A **wrong or missing chat template** shows up as "the model can’t edit or read files" — not as an obvious template error. If the model’s raw completion never gets recognized as containing a valid tool tag (garbled tags, or tags wrapped in text the parser doesn’t expect), the harness reports it as an inability to use tools at all, which reads exactly like a capability gap even when it’s a template mismatch. Check `--jinja` and your model’s actual template — the same diagnostic [[module-10-serving|Module 10.6]] walked through for the autoparser applies here too.
- You may have to **manually flag tool-calling capability** for a model in the harness’s settings, since an OpenAI-compatible custom endpoint often can’t auto-detect what the model actually supports the way a named, first-party provider integration can. Check your version’s settings for a model-capability toggle before assuming a failure is the model’s fault.
- **Reasoning models can stall the loop.** A model that emits a long `<think>`-style reasoning block before its tool call can trip a harness that expects the tool tag near the start of the response, or that times out waiting for output it interprets as "no action taken" — producing a retry loop that looks like the model refusing to act, when it’s actually still finishing a reasoning pass the harness didn’t budget time for.

**Steps:**
1. Reproduce the "cannot edit/read files" symptom deliberately — misconfigure `--jinja` or point the harness at a mismatched template — and confirm your harness reports it exactly as a capability failure, not a template error.
2. If you have a reasoning-tuned model available, run one task through it under Cline/Roo and watch for retry-loop behavior tied to a long reasoning block preceding the tool tag.

> [!success] Done when
> You’ve deliberately reproduced the template-mismatch symptom once, on your own setup, and can describe exactly what it looks like from the harness’s side versus what’s actually wrong underneath.

> [!question]- It’s not working
> 1. The harness silently ignores every tool call the model emits, even though the raw text looks correctly tagged to you? Check for a subtle mismatch between the exact tag names/attributes the harness’s parser expects and what the model actually produced — a single misspelled attribute name is enough to make the parser miss the whole block.
> 2. Requests reach the server but responses never make it back to the harness UI? Confirm streaming isn’t being buffered somewhere in your logging proxy or a reverse proxy in front of it, the same failure mode [[module-10-serving|Module 10.2]] flagged for `nginx`.

## 12.5 Convert the token delta into seconds of TTFT

> [!abstract] Goal
> Take the tokens-per-turn gap between Cline/Roo and Aider on the same task, and convert it into a wall-clock TTFT difference using the [[module-6-prefill-decode|Module 6]] prefill curve — this is the module’s real deliverable.

**What’s going on:** A token-count comparison alone ("Cline sends 3x more tokens than Aider per turn") is suggestive but not the number that matters to a person sitting at the keyboard. What matters is: how many *extra seconds* does that verbosity cost before the first response token appears, on this specific machine, at this specific configuration. Nobody instruments this, because it requires exactly the two things this course spent ten modules building: a real prefill-tok/s-vs-depth curve, and a real per-harness token count on an identical task.

**Steps:**
1. From your Module 11 and Module 12 logs, compute the mean `prompt_tokens` per turn for Aider and for Cline/Roo, on the same or an equivalent fixed task set.
2. Using the [[module-6-prefill-decode|Module 6.3]] prefill-tok/s-vs-depth table at your chosen-optimum `n_cpu_moe` configuration, convert each harness’s mean token count into seconds: `TTFT_seconds ≈ prompt_tokens / prefill_tok_s_at_that_depth`.
3. Report the **delta** directly: `TTFT_cline − TTFT_aider`, in seconds, at a representative turn depth (e.g., a mid-conversation turn with several prior turns of history already accumulated, not just the very first turn of a session).
4. State this delta as the architecture tax: this is what an inlined-XML harness costs you, in wall-clock seconds, purely from being more verbose than a diff-based one, on this exact workstation — independent of anything about model quality or task difficulty.

> [!success] Done when
> You have a stated seconds-of-TTFT delta between Cline/Roo and Aider on an equivalent task, derived from your own Module 6 prefill curve and your own Module 11/12 token logs — not an estimate, a computed number with both inputs shown.

> [!bug] Gotcha
> Make sure the two harnesses’ turns you’re comparing are at comparable conversation depth — a first turn (short accumulated history) against a tenth turn (long accumulated history) will show a token gap driven mostly by turn position, not harness architecture. Compare turns at matched depth, or average across a matched range of depths for both harnesses.

---

## Check your understanding

> [!question]- 1. Why can a model that reliably produces valid native JSON tool calls still fail at Cline/Roo’s tool-call protocol?
> > [!success]- Answer
> > Because Cline/Roo’s tag syntax isn’t something the model was trained to produce — it’s taught entirely in-context, in the system prompt, every session. Native JSON tool calling is typically baked into a model’s fine-tuning and chat template; the inlined XML-ish dialect is not. A model can be excellent at the trained format and still garble a format it only saw described a few lines earlier in its own context window, because those are two different skills, and only one of them benefited from training.

> [!question]- 2. Why does this module convert a token-count difference into a seconds-of-TTFT figure, instead of just reporting "Cline uses N more tokens per turn than Aider"?
> > [!success]- Answer
> > Because a raw token count doesn’t tell you what a person waiting at the keyboard actually experiences, and on this machine prefill is expensive enough that the conversion matters. Using the measured Module 6 prefill-tok/s-vs-depth curve turns an abstract token gap into a concrete number of extra seconds spent staring at a spinner before the first response token appears — the number that actually determines whether the harness feels responsive, not just which one is "more efficient" in the abstract.

> [!question]- 3. A reasoning-tuned model appears to loop retries under Cline even though its final tool call, once it arrives, is correctly formatted. What’s the likely cause, and is this the model’s fault?
> > [!success]- Answer
> > The likely cause is a long reasoning block (`<think>`-style output) preceding the tool tag, which can trip a harness that expects the tool call to appear promptly, or that interprets the delay as "no action taken" and retries. It’s not purely the model’s fault — the harness’s timing assumptions were built around models that don’t reason at length before acting, and a reasoning model’s correct behavior (think first, then act) collides with those assumptions. This is a harness/model interaction issue, not a formatting defect in either piece alone.

## What’s next

You now have Cline/Roo’s tool-call protocol confirmed directly from a captured request, a measured tokens-per-turn distribution on your own fixed task set, at least one deliberately reproduced template-mismatch failure, and a computed seconds-of-TTFT tax for this harness’s verbosity relative to Aider’s diff format — a number nobody else has published for this specific machine. [[module-13-claude-code]] takes the opposite extreme: a harness that speaks Anthropic’s own Messages API natively, built around a specific frontier model’s behavior, pointed at a local server that was never the intended target — and measures exactly which of Claude Code’s own features survive that swap.

<script src="/tutor.js" defer></script>
