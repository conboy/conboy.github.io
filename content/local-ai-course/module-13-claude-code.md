---
title: "Module 13 — Claude Code Against a Local Model"
description: "Pointing Claude Code at a local server via ANTHROPIC_BASE_URL, and measuring exactly what breaks — extended thinking, Plan Mode, prompt caching across model switches — with a real prefill-tax and turns-to-completion comparison against the API."
tags:
  - local-ai
  - claude-code
  - anthropic-api
  - tool-calling
  - harness
---

# Module 13 — Claude Code Against a Local Model

**6 hours** · Part of [[index|Two Pools]] · Prev: [[module-12-cline]] · Next: [[capstone-1-predicted-machine]]

> [!info] What this module is for
> Claude Code doesn’t speak an OpenAI-compatible wire format at all — it speaks the **Anthropic Messages API** directly, and it can be redirected to any server that answers on that same shape via two environment variables. That redirection is real, documented, and genuinely usable for an experiment on this workstation. It is also, by Anthropic’s own account, **not something Anthropic supports, endorses, or audits** — you are point ing a tool built around one specific model family’s behavior at a target it was never built for, and this module’s job is to measure exactly what survives that mismatch and what doesn’t, rather than either dismissing the idea or overselling it. Five concrete things break or degrade, each with a measured or directly observable consequence, and the module ends the way every module in this course does: with a real number, not an impression, comparing this local setup against the same task run through the real API.

---

## 13.1 Pointing Claude Code at a local server

> [!abstract] Goal
> Understand the two environment variables that redirect Claude Code, which one is correct for a local server, and when you need a translation shim versus a server with native support.

**What’s going on:** Claude Code reads `ANTHROPIC_BASE_URL` to decide which server to talk to, and requires *some* credential to be present alongside it. Two variables can supply that credential, and they are not interchangeable: `ANTHROPIC_API_KEY` is sent as an `x-api-key` header, while `ANTHROPIC_AUTH_TOKEN` is sent as a bearer-style `Authorization` header. For a local server that doesn’t validate credentials at all, either can work mechanically, but `ANTHROPIC_AUTH_TOKEN` is the one worth using deliberately here — it’s explicitly a bearer-token slot, and a placeholder value in it reads clearly, to anyone looking at your setup later, as "present because required, not because it’s checked."

Getting a request to actually land correctly still depends on what’s on the other end. `llama-server`, and recent versions of Ollama and LM Studio, reportedly implement the Anthropic Messages API directly at a `/v1/messages` endpoint — meaning no translation layer sits between Claude Code and the model. **Anthropic’s own documentation does not itself name which third-party servers implement this**, so treat specific version numbers for native support as something to verify against your own build’s current release notes and changelog before relying on them, not as a fixed fact this course can hand you. On an older build without native support, a translation shim — `claude-code-router` or `LiteLLM` are the commonly cited options, and `vLLM` exposes its own Anthropic-compatible route — sits between Claude Code and the model, translating Messages-API requests into whatever format the backend actually understands.

**Steps:**
1. Check your installed `llama-server`, Ollama, or LM Studio version’s own changelog or release notes for `/v1/messages` support, rather than assuming a specific version number from this or any other guide is current — this is a fast-moving compatibility surface.
2. Point Claude Code at your local server:
   ```bash
   export ANTHROPIC_BASE_URL=http://localhost:8080
   export ANTHROPIC_AUTH_TOKEN=sk-local-placeholder
   claude
   ```
3. Send one trivial prompt and confirm you get a coherent response routed through your local model, not an error indicating Claude Code is still trying to reach Anthropic’s real API — a wrong `ANTHROPIC_BASE_URL` value or a missing trailing detail in the URL is the most common reason this silently fails to redirect.
4. If your server doesn’t natively support `/v1/messages`, install and configure a translation shim, and repeat the same trivial-prompt check through it instead.

> [!success] Done when
> You have a real request/response round trip through `ANTHROPIC_BASE_URL` against your local server (native or shimmed), with a coherent response confirming you’re actually talking to your own model, not Anthropic’s.

> [!question]- It’s not working
> 1. Claude Code appears to hang or time out on startup, before you’ve sent anything? Check the count-tokens behavior in 13.3 first — this is a documented sharp edge on some local servers.
> 2. Requests seem to reach the server (visible in server logs) but Claude Code reports a malformed-response error? Confirm the server’s `/v1/messages` implementation is actually returning Anthropic’s response shape and not a translated-but-incomplete approximation of it — a partial shim implementation is a common source of this exact failure.

## 13.2 What breaks, and the measured consequence of each

> [!abstract] Goal
> Know the five concrete things that change when Claude Code is pointed at a non-Anthropic model, and — critically — attach a real, observed or measured consequence to each rather than listing them as abstract caveats.

**What’s going on:**

1. **Extended thinking degrades or disappears.** Extended thinking is enabled through an Anthropic-specific beta header with no equivalent request-body field for other model families — it’s an Anthropic-model feature, not a wire-protocol feature, so it’s silently unavailable once the header stops meaning anything to the model on the other end. **Consequence:** any workflow you’ve built around watching or steering Claude’s visible reasoning trace simply has nothing to show — not a degraded version of it, an absence.
2. **Plan Mode may not translate.** Plan Mode’s behavior is built around Claude’s own specific instruction-following and planning tendencies. There is no official documentation describing what happens when Plan Mode is invoked against an arbitrary local model — this is a genuine gap, not a claim this course can cite. **Consequence you should measure yourself:** run one task in Plan Mode against your local model and record, plainly, whether it produces a usable plan, a degraded one, or effectively ignores the mode’s intent — and report whichever you actually observe, since nobody has published this for you to check against.
3. **Prompt caching breaks across model switches, and this one is directly measurable.** Anthropic’s prompt-caching mechanism keys its cache per backend model; switching which model answers a conversation invalidates the cache for that conversation entirely. On a workstation where DDR5 delivers roughly 57.6 GB/s theoretical, a full-conversation re-prefill after a model switch is not a rounding error — it’s the exact prefill cost [[module-6-prefill-decode|Module 6]] already taught you how to measure, applied to your *entire* accumulated conversation instead of one turn’s new content. **Consequence, measured directly in 13.4:** the seconds of stall the next request after a model switch actually costs you.
4. **Tool-calling reliability becomes a property of whichever model you chose**, not of Claude Code itself. Claude Code’s own tool-calling machinery is only as good as the model executing it — this is exactly [[module-10-serving|Module 10.7]]’s measured failure-rate curve, re-applied here. If you’re running a model below the size AMD’s own cited testing flagged as unreliable for agentic tool calling, expect that same unreliability inside Claude Code, unchanged by which harness is asking.
5. **It is unofficial and unsupported.** Anthropic’s own position, stated plainly in its documentation: it does not endorse, maintain, or audit third-party gateway products, and does not support routing Claude Code to non-Claude models through any gateway. This isn’t a minor caveat to footnote — it means there is no support channel, no compatibility guarantee across Claude Code releases, and no expectation that a working setup today stays working after the next update.

**Steps:**
1. For each of the five items, write down which ones you can measure directly on this workstation (2, 3, 4) versus which ones require you to simply observe and report what happens, absent any published baseline to check against (1, 2 again for its qualitative half, 5).
2. Run one task under Plan Mode against your local model specifically, and record your own honest observation per item 2 — resist the temptation to either force a positive result or assume it must fail; report what actually happened.

> [!success] Done when
> You can state all five items from memory, each attached to its actual consequence rather than as a bare list, and you have your own direct Plan Mode observation recorded rather than a borrowed claim about it.

> [!bug] Gotcha
> Don’t conflate "extended thinking is unavailable" with "the model isn’t reasoning at all." A capable local model can still produce a strong final answer without an exposed thinking trace — the absence is about *visibility and control* of the reasoning process, not necessarily about the model’s underlying capability on the task.

## 13.3 The count-tokens hang

> [!abstract] Goal
> Know about a documented failure specific to at least one local server, and treat it as a live, build-dependent bug to check against the current issue tracker rather than a fixed fact.

**What’s going on:** Anthropic’s documentation describes token-counting (`count_tokens`) as an optional endpoint that Claude Code falls back away from gracefully — using a character-based estimate — when the endpoint is absent. In practice, at least one report describes a local server hanging or behaving badly specifically around this endpoint, rather than cleanly returning "not implemented" and letting Claude Code fall back the way the graceful-degradation design intends. Related reports describe concurrency issues on single-slot local servers under Claude Code’s request pattern more broadly. Treat all of this as **build-dependent and worth checking fresh**, not as a fixed defect — check the current issue trackers for both Claude Code and whichever server you’re running before assuming your own hang (if you hit one) is this exact bug.

**Steps:**
1. Before your first real session, send a request that would trigger Claude Code’s token-counting call (or simply start a normal session and watch server-side logs at startup) and confirm whether your server responds, errors cleanly, or hangs.
2. If you observe a hang or an unreasonably long stall specifically tied to a count-tokens request, check your server’s own issue tracker and the Claude Code issue tracker for existing reports before spending time debugging it as if it were novel.
3. If your server supports running multiple concurrent request slots, confirm it’s configured for at least a small number of parallel slots — a single-slot server serving a harness that may issue more than one request in flight is a plausible source of stalls that look like this bug but aren’t.

> [!success] Done when
> You’ve confirmed, on your own server and build, whether the count-tokens path behaves cleanly or hangs, and you know where to check (both projects’ issue trackers) if it does.

## 13.4 Measure: prefill tax and turns-to-completion, local vs. API

> [!abstract] Goal
> Measure the system-prompt-plus-tool-definition prefill cost in tokens and seconds, and compare turns-to-completion on a fixed task between your local setup and the real Anthropic API.

**What’s going on:** Every Claude Code session opens with a substantial system prompt plus a full set of tool definitions before your actual task content even begins — that fixed overhead is paid on every cold-started conversation, local or hosted, but it costs categorically different wall-clock time depending on which side of the 11x bandwidth gap in [[workstation-specs]] is doing the prefilling.

**Steps:**
1. Using your [[module-11-aider|11.3]]-style logging proxy in front of your local server, start a fresh Claude Code session (no prior cache) and capture `prompt_tokens` on the very first request — this figure is dominated by the system prompt and tool definitions, since your own task content hasn’t been added yet.
2. Convert that token count to seconds using the [[module-6-prefill-decode|Module 6]] prefill-tok/s curve at your current `n_cpu_moe` configuration, the same conversion method from [[module-12-cline|Module 12.5]].
3. Pick one fixed, concrete coding task and run it to completion twice: once through Claude Code against your local server, once through Claude Code against the real Anthropic API (a normal, unmodified `ANTHROPIC_BASE_URL`/no override). Record the number of turns each run took to reach a working result.
4. Report three numbers together: the local system-prompt-plus-tools prefill tax in tokens, its converted seconds figure, and the turns-to-completion comparison between local and API on the identical task.

> [!success] Done when
> You have a measured prefill-tax figure (tokens and seconds) for the fixed system-prompt-plus-tool-definition overhead on your local setup, and a turns-to-completion comparison on one identical task, local versus the real Anthropic API.

> [!question]- It’s not working
> 1. Your local run’s turns-to-completion is dramatically worse than the API run’s, even accounting for prefill tax? Check whether the gap is actually tool-calling reliability (item 4 in 13.2) rather than raw speed — a model that fails and retries tool calls will take more turns regardless of how fast each individual turn runs.
> 2. Prefill-tax token count looks suspiciously small for a full Claude Code system prompt? Confirm you captured the very first request of a genuinely fresh session — any caching (local or otherwise) from a prior run would undercount this figure.

## 13.5 Frame this honestly

> [!abstract] Goal
> State, in one paragraph, what this module has actually established — a viable, measured experiment, not a daily-driver recommendation.

**What’s going on:** Everything measured in this module is real: a genuine request/response round trip, a genuine prefill-tax figure, a genuine turns-to-completion comparison. None of it changes Anthropic’s own stated position that this configuration is unsupported, unaudited, and not guaranteed to keep working release over release. The honest framing sits between two wrong ones: "this obviously doesn’t work" (it does, mechanically, today) and "this is a real replacement for the API" (extended thinking is gone, Plan Mode is unverified, prompt caching resets on every model switch, and tool-calling reliability is capped by whatever local model you chose). Call it what it is — a genuinely viable **experiment**, worth running and worth measuring precisely because so little of it is published elsewhere — and not yet a reliable daily-driver replacement.

**Steps:**
1. Write your own one-paragraph summary of this module’s results, explicitly separating what you measured from what you’re recommending — these are not the same statement.

> [!success] Done when
> You have a written, honest one-paragraph summary distinguishing "what I measured" from "what I’d recommend running day to day," backed by your own 13.4 numbers.

---

## Check your understanding

> [!question]- 1. What’s the difference between `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN`, and which is the more deliberate choice for a local server that doesn’t validate credentials?
> > [!success]- Answer
> > `ANTHROPIC_API_KEY` is sent as an `x-api-key` header; `ANTHROPIC_AUTH_TOKEN` is sent as a bearer-style `Authorization` header. Either can work mechanically against a local server that ignores credentials entirely, but `ANTHROPIC_AUTH_TOKEN` is the more deliberate choice here — its name and header shape signal "a token that must be present," which matches the actual local-server situation more honestly than reusing the API-key slot meant for real authentication.

> [!question]- 2. Why does a model switch mid-conversation cost so much more on this workstation than it would on a hosted, high-bandwidth API backend?
> > [!success]- Answer
> > Because Anthropic’s prompt cache is keyed per backend model, so switching models invalidates the cache for the whole conversation and forces a full re-prefill of everything accumulated so far — not just the new content, all of it. On a hosted API backend with very high aggregate bandwidth, that re-prefill is comparatively cheap. On this workstation, prefill throughput is governed by the same DDR5/VRAM split [[module-6-prefill-decode]] already measured, so a full-conversation re-prefill after a model switch is a genuine, multi-second (or longer, at long context) stall — not a rounding error.

> [!question]- 3. Your local Claude Code setup takes noticeably more turns to complete a task than the same task run against the real API. Is that necessarily a speed problem?
> > [!success]- Answer
> > Not necessarily — it could just as easily be a tool-calling reliability problem, which is a completely different thing to fix. A model that occasionally emits a malformed or semantically wrong tool call will cost extra turns recovering from that failure, independent of how many tokens per second it can produce. Separating "this took longer because each turn was slower" from "this took longer because it needed more turns to recover from failures" is exactly why this module measures prefill tax and turns-to-completion as two distinct numbers rather than folding them into one.

## What’s next

You now have a real, measured local Claude Code setup: a confirmed request/response round trip through `ANTHROPIC_BASE_URL`, a prefill-tax figure in both tokens and seconds for the fixed system-prompt-plus-tool-definition overhead, a turns-to-completion comparison against the real API on an identical task, and an honest accounting of which Claude-specific features survive the swap and which don’t. Every module up to here has measured one piece of this workstation in isolation — a bandwidth number, a placement curve, a quant ladder, a harness’s token cost. [[capstone-1-predicted-machine]] is where all of it gets assembled into a single predictive model and tested against three configurations you have never actually run — including two models roughly four times this model’s size — with the predictions written down before a single benchmark executes.

<script src="/tutor.js" defer></script>
