---
title: "Module 11 — Aider: Diffs, Repo Maps, Cost per Turn"
description: "Instrumenting Aider’s local endpoint to measure tokens per turn and the repo map’s real cost against its benefit, and why sidestepping JSON tool calling makes Aider more robust on weaker local models."
tags:
  - local-ai
  - aider
  - repo-map
  - tool-calling
  - harness
---

# Module 11 — Aider: Diffs, Repo Maps, Cost per Turn

**7 hours** · Part of [[index|Two Pools]] · Prev: [[module-10-serving]] · Next: [[module-12-cline]]

> [!info] What this module is for
> [[module-10-serving|Module 10]] measured three tool-call protocols in the abstract, on a synthetic battery. This module and the next two put a real harness on top of the server you’ve spent ten modules tuning, and measure what actually happens when a coding agent — not a benchmark script — sends real turns against it. Aider goes first because its whole design is a bet against the assumption every other harness in this course makes: that the model needs to emit well-formed JSON to get anything done. Aider doesn’t ask for JSON tool calls at all. It asks the model to write a diff, or rewrite a whole file, as plain text, and parses that text itself. That single choice is why Aider tends to be the most forgiving harness on a local model that hasn’t been fine-tuned on a specific function-calling format — and it’s worth understanding mechanically, not just accepting as a reputation. This module also introduces the instrumentation rig — a token-logging layer between harness and server — that Module 12 reuses unchanged, so the Aider-vs-Cline comparison in Module 12 is apples to apples.

---

## 11.1 The edit format: diffs instead of tool calls

> [!abstract] Goal
> State precisely what Aider asks the model to emit for a code edit, and why that sidesteps the JSON-tool-calling reliability problem entirely rather than solving it.

**What’s going on:** Every protocol in [[module-10-serving|Module 10.4]] — inline XML, native JSON, GBNF-constrained JSON — exists to get a *structured* request out of a model’s output. Aider’s edit formats (`diff`, `diff-fenced`, `whole`, `udiff`, depending on model and configuration) don’t ask for structure at all for the actual code change. The model writes ordinary prose containing a fenced code block that looks like a unified diff or a full file body, and Aider’s own parser — regular text processing, not a JSON deserializer, not a grammar — extracts the edit from that block. There is no JSON to malform. A model that reliably writes "change this line to that line" in near-diff syntax succeeds at Aider edits even if that same model reliably breaks well-formed JSON schemas under GBNF-free decoding, which is exactly the failure mode [[module-10-serving|Module 10.7]] measured across smaller models.

This is a real architectural bet, not a cosmetic difference: **Aider is robust to weak instruction-following on the *structure* of the output, because it doesn’t require structure — only a recognizable diff shape.** That doesn’t mean Aider is immune to model weakness; a model that can’t reliably identify *which* lines to change will still fail, just further upstream, in the actual editing decision rather than in output formatting.

**Steps:**
1. Install Aider and confirm the current recommended install path from its own documentation before copying any specific command here — the packaging story has changed release to release. As of this writing, `python -m pip install aider-install && aider-install` is the documented path; verify against `aider --version` and the project’s own install page before trusting it on your build.
2. Point Aider at your local server the same way you’d point any LiteLLM-backed client at an OpenAI-compatible endpoint:
   ```bash
   export OPENAI_API_BASE=http://localhost:8080/v1
   export OPENAI_API_KEY=sk-local-placeholder
   aider --model openai/qwen3-coder-30b-a3b --verbose
   ```
   The `openai/` prefix tells Aider’s underlying LiteLLM layer to treat this as a generic OpenAI-compatible target rather than trying to route to Anthropic or another named provider.
3. Give Aider one small, real edit task in a scratch repo, and watch the terminal output for the literal diff block the model produced before Aider applied it — confirm with your own eyes that no JSON ever appears in that exchange.

> [!success] Done when
> You’ve run one real edit through Aider against your local server and can point to the exact diff-shaped text block the model produced, with no JSON tool call anywhere in the exchange.

> [!question]- It’s not working
> 1. Aider reports it can’t parse the model’s edit, even though the model’s output looks like a reasonable diff to you? Check which edit format Aider auto-selected for your model (`--edit-format` overrides the auto-detection) — a model that writes clean whole-file output may be getting scored against a stricter diff-format parser than it’s actually attempting to satisfy.
> 2. Requests seem to reach the server but nothing comes back? Confirm `OPENAI_API_BASE` includes the `/v1` suffix your server expects — a bare host:port without the path prefix is a common miss.

## 11.2 The repo map: tree-sitter and PageRank, not embeddings

> [!abstract] Goal
> Understand precisely how Aider decides which parts of a codebase to show the model, and rule out the wrong mental model — this is not retrieval-augmented generation.

**What’s going on:** As a codebase grows past what fits in context, Aider builds a **repo map**: a compressed outline of classes and function signatures across the repository, budgeted to a token limit (`--map-tokens`, default chosen based on the model’s context size unless set explicitly). The map is built two ways, and it’s worth being exact about both, because the wrong mental model — "it’s doing semantic search over the codebase" — leads to wrong predictions about what it will and won’t surface:

1. **Extraction** uses **tree-sitter** to parse every source file into a concrete syntax tree and pull out identifiers: class names, function signatures, and the call/reference relationships between them. This is a syntactic parse, not a semantic embedding — it knows that `foo()` calls `bar()` because the parse tree says so, not because a vector database found `bar` "close to" `foo` in embedding space.
2. **Ranking** treats the whole repository as a graph — files (or symbols) as nodes, references between them as edges — and runs a **PageRank-style algorithm** over that graph to decide which symbols are most "important" given which files are currently in the chat context. A file heavily referenced by files you’re actively editing ranks higher than an equally large file nobody currently in context touches.

**There is no embedding model anywhere in this pipeline.** No vector index, no nearest-neighbor search, no semantic similarity score. If you’ve used embedding-based RAG code assistants before, resist importing that mental model here — Aider’s repo map will surface a heavily-referenced utility function even if its *name* has nothing semantically in common with your current task, and it will *miss* a semantically-relevant file that happens to be structurally isolated (no inbound or outbound references from anything currently in view).

**Steps:**
1. Run `aider --show-repo-map` (or the equivalent current flag — check `aider --help`) against a real multi-file repo, and read the emitted map: confirm it’s signatures and structure, not prose summaries or semantic descriptions.
2. Pick one file you know is heavily imported elsewhere in the repo and one that’s structurally isolated (no imports in or out); confirm the heavily-referenced one ranks higher in the map even without opening it in the chat.
3. State, in your own words, one concrete scenario where PageRank-over-references would miss something an embedding search would catch — you’ll need this distinction sharp for 11.6.

> [!success] Done when
> You can explain repo-map construction as "tree-sitter for extraction, PageRank-style graph ranking for selection" without reaching for the word "embedding," and you’ve demonstrated the reference-count effect on your own repo.

> [!bug] Gotcha
> `--map-tokens 0` disables the repo map entirely — useful as a clean A/B baseline in 11.5, not just an obscure edge case. Don’t confuse "map disabled" with "map empty because nothing ranked" when you’re reading logs; the flag makes the difference unambiguous.

## 11.3 Instrument the endpoint: log tokens per request

> [!abstract] Goal
> Build a thin logging layer between Aider and your local server so every request’s prompt and completion token counts are recorded, without changing Aider’s or the server’s behavior.

**What’s going on:** `llama-server`’s OpenAI-compatible endpoint returns a `usage` object (`prompt_tokens`, `completion_tokens`, `total_tokens`) on non-streaming requests, and via `stream_options: {"include_usage": true}` on streaming ones — check your build’s current behavior, since streaming usage reporting has been a moving target across `llama.cpp` releases. Rather than modifying Aider itself, put a small logging reverse proxy in front of the server and point Aider at the proxy instead of the server directly. This is the same rig Module 12 reuses, so build it once, generically, keyed only on "log every request/response pair," not on anything Aider-specific.

**Steps:**
1. Write (or adapt) a minimal HTTP reverse proxy — a short Python script using `http.server` or a small `mitmproxy` addon both work — that forwards every request to `http://localhost:8080` unchanged, and on the way back, parses the response’s `usage` field and appends `{timestamp, prompt_tokens, completion_tokens, total_tokens}` to a CSV or JSONL log file.
2. Point Aider’s `OPENAI_API_BASE` at the proxy’s port instead of the server’s, and confirm end-to-end that edits still work identically — the proxy must be transparent, or every measurement downstream is comparing a different thing than production use.
3. Run a handful of real edit turns and confirm the log file is actually accumulating one row per request, with plausible non-zero token counts.

> [!success] Done when
> You have a working logging proxy that records prompt/completion tokens for every real Aider request without altering Aider’s behavior, verified against at least ten real turns.

> [!question]- It’s not working
> 1. `usage` field is missing or zeroed on streamed responses? Confirm you’ve set the `include_usage` stream option on the client side (Aider’s LiteLLM layer may or may not expose this directly — check whether disabling streaming for the instrumentation run is simpler than chasing the flag through LiteLLM).
> 2. Proxy silently breaks streaming (Aider seems to hang mid-response)? Make sure your proxy forwards chunks as they arrive rather than buffering the full response before relaying it — the same streaming-buffering trap [[module-10-serving|Module 10.2]] flagged for `nginx` applies here too.

## 11.4 Measure tokens-sent-per-turn and convert to prefill seconds

> [!abstract] Goal
> Turn the raw token log into a distribution of tokens-per-turn, then convert that distribution into wall-clock prefill seconds using the [[module-6-prefill-decode|Module 6]] prefill-tok/s curve, so "tokens" becomes "time you actually waited."

**What’s going on:** A token count on its own doesn’t tell you what a session felt like. [[module-6-prefill-decode|Module 6]] already measured prefill tok/s as a function of prompt depth at your chosen `n_cpu_moe` configuration — reuse that curve directly here instead of re-measuring prefill from scratch.

**Steps:**
1. Run a fixed task set — reuse a Module 3/8-style Aider Polyglot subset, or a handful of real multi-file edit sessions in a repo you know well — through your logged proxy, and pull `prompt_tokens` per turn out of the log.
2. Plot the tokens-sent-per-turn distribution across the whole task set: mean, median, and spread. Note whether it grows turn-over-turn within a single multi-turn session (expected, since Aider resends accumulated context) or resets (would suggest the repo map or chat history is being trimmed somewhere you didn’t expect).
3. For each turn’s `prompt_tokens` figure, look up (or interpolate) the corresponding prefill tok/s from your Module 6 table at the nearest measured depth, and divide: `prefill_seconds = prompt_tokens / prefill_tok_s_at_that_depth`. This converts the raw token count into the actual seconds of wall-clock time that turn’s prefill cost you.
4. Report both distributions side by side: tokens-per-turn, and prefill-seconds-per-turn.

> [!success] Done when
> You have a tokens-per-turn distribution and its converted prefill-seconds-per-turn distribution across a fixed task set, using your own Module 6 prefill curve for the conversion.

## 11.5 Measure the repo map’s token cost against its benefit

> [!abstract] Goal
> Run the same fixed task set with the repo map on and off, and measure both sides of the trade-off directly: how many tokens the map costs, and whether task success actually improves because of it.

**What’s going on:** The repo map isn’t free — it’s a chunk of every prompt, paid on every turn, whether or not the model ends up using the information in it. Whether that cost is worth paying is an empirical question this section answers directly, not an assumption.

**Steps:**
1. Run your fixed task set twice: once with `--map-tokens 0` (map disabled) and once with your normal map budget, everything else held constant.
2. From your 11.3 log, compute the mean token overhead the repo map adds per turn — this should show up as a consistent gap in `prompt_tokens` between the two runs on otherwise-identical turns.
3. Score task success (pass/fail against your fixed task set’s own success criterion — a passing diff, a passing test, whatever your task set defines) for both runs.
4. Report the pairing explicitly: token cost per turn with the map on, and the resulting change (positive, negative, or within noise) in task success rate.

> [!success] Done when
> You have a measured mean token cost for the repo map on your fixed task set, and a measured task-success delta between map-on and map-off — reported together, not as two disconnected numbers.

> [!question]- It’s not working
> 1. Success rate looks identical with the map on or off? That’s a legitimate result on a small or single-file task set — the repo map’s value scales with codebase size and file count, so a trivial task set may simply not be large enough to need it. Note the task set’s size explicitly rather than treating a null result as a bug.
> 2. Token overhead varies wildly turn to turn even with the map budget fixed? Check `--map-refresh` — Aider recomputes the map on a schedule (`auto`, `always`, `files`, or manual, depending on version), and a map recomputed mid-session against a different chat-file set will legitimately produce a different-sized map turn to turn.

## 11.6 The honest gap: no public repo-map-vs-RAG-vs-long-context comparison exists

> [!abstract] Goal
> State clearly what this module has and hasn’t proven, and why your own 11.4/11.5 numbers are a genuine contribution rather than a repeat of published work.

**What’s going on:** It would be reasonable to assume someone has already run a controlled three-way comparison — tree-sitter repo map vs. embedding-based RAG vs. just stuffing more of the codebase into a longer context window — for a coding task. As far as this course’s research turned up, **no such controlled comparison exists publicly for a local coding model against a real repository.** The closest adjacent result is a study comparing a tree-sitter-based code knowledge graph against a file-exploration coding agent, which reported the graph-based approach reaching roughly 83% answer quality against the exploration agent’s 92%, at roughly 10x fewer tokens and 2.1x fewer tool calls — a real efficiency-vs-quality trade-off, measured. But that study’s own authors flagged the comparison against embedding-based RAG specifically as future work, not something they themselves measured. Nobody has closed that gap publicly.

That means your 11.4 and 11.5 numbers — tokens-per-turn, prefill-seconds-per-turn, and the repo map’s own cost-vs-benefit on your fixed task set — are not a rerun of an existing result. They’re a genuine, if narrow, data point in a comparison space that’s still mostly unmeasured. Say so plainly if you write this up: "I don’t have a RAG or long-context-stuffing comparison arm, and neither, as far as I could find, does anyone else, publicly, for a local coding model against a real repo."

**Steps:**
1. Write down, in one paragraph, exactly what your 11.4/11.5 measurements do and do not establish — specifically, that they say nothing about how a RAG-based or long-context-stuffing alternative would have performed on the same task set, because you didn’t build either arm.
2. If you have time beyond this module’s budget, the single highest-value extension is building a crude long-context-stuffing baseline — concatenate the whole repo into the prompt up to your context limit, skip the repo map entirely — and compare its token cost and task success against both of your 11.5 runs. This is explicitly optional and outside this module’s 7-hour budget; note it as a possible future module rather than attempting it here.

> [!success] Done when
> You can state precisely what comparison arm is missing from your own measurement, correctly attribute the 83%/92%/10x/2.1x figures as a different (adjacent, not identical) study’s finding, and explain why that study’s authors — not this course — are the ones who called the RAG comparison future work.

---

## Check your understanding

> [!question]- 1. Why does Aider’s diff-based edit format tend to be more robust on a local model than a harness that requires native JSON tool calling?
> > [!success]- Answer
> > Because Aider never asks the model to produce structured JSON for the edit itself — it asks for a diff or whole-file rewrite in plain text, which its own parser extracts. A model that reliably writes near-diff-shaped text can succeed even if that same model is unreliable at emitting well-formed JSON under a stricter tool-calling protocol. It doesn’t fix a model that can’t identify the right edit — it just removes one entire layer of possible failure (output-format validity) that other protocols depend on.

> [!question]- 2. Why is "PageRank over a reference graph" the correct mental model for Aider’s repo map, rather than "semantic search over the codebase"?
> > [!success]- Answer
> > Because there’s no embedding model or vector index anywhere in the pipeline. Tree-sitter extracts symbols and their reference relationships from a syntactic parse, and a PageRank-style algorithm ranks symbols by how heavily-referenced they are relative to whatever’s currently in the chat context — importance by graph centrality, not by semantic similarity. A file can rank highly because it’s heavily imported even if its name and contents share no semantic overlap with the current task, and a semantically relevant but structurally isolated file can be missed entirely — the opposite failure mode an embedding-based system would have.

> [!question]- 3. Your 11.6 write-up cites a study finding 83% vs 92% answer quality between a tree-sitter graph and a file-exploration agent, at 10x fewer tokens. Does that number validate your own 11.5 repo-map measurement?
> > [!success]- Answer
> > Not directly — it’s a different system (a knowledge-graph QA setup, not Aider specifically), a different task (general codebase question-answering, not code editing), and critically, it has no RAG comparison arm either; its own authors named that as future work. It’s useful as the closest adjacent shape of result — efficiency-quality trade-offs in this space do seem to run in the tens-of-percent-quality-for-order-of-magnitude-fewer-tokens range — but treating it as validation of your specific Aider measurement would be citing an adjacent finding as if it were a direct replication, which it isn’t.

## What’s next

You now have a working token-logging rig, a tokens-per-turn and prefill-seconds-per-turn distribution for Aider on your local model, a measured repo-map cost-vs-benefit pairing, and an honest account of the comparison this field still hasn’t published. [[module-12-cline]] reuses the exact same instrumentation against a structurally different harness — Cline and Roo inline their entire tool-call protocol as custom XML syntax in the prompt itself, which should cost meaningfully more tokens per turn than Aider’s diff format, and this time you have the rig built to put an actual number, in seconds, on that architectural difference.

<script src="/tutor.js" defer></script>
