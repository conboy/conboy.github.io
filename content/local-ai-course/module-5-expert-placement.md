---
title: "Module 5 — Placing the Experts"
description: "Predicting the tok/s knee across n_cpu_moe before sweeping it, deriving a VRAM-budget-constrained optimum, and building the allocator script that picks it for you."
tags:
  - local-ai
  - moe
  - vram
  - llama.cpp
  - roofline
---

# Module 5 — Placing the Experts

**12 hours** · Part of [[index|Two Pools]] · Prev: [[module-4-vram-budget]] · Next: [[module-6-prefill-decode]]

> [!info] What this module is for
> This is the centrepiece of the course, and its twelve-hour budget is not padding. Everything before this module built the tools: a measured DDR5 bandwidth number, a measured VRAM bandwidth number and their ratio, a calibration constant for the gap between theory and reality, and a complete VRAM budget equation that tells you exactly how much room is left once weights, KV cache, and overhead have taken their share of 16 GB. This module spends all of that at once. You’ll derive — on paper, from the model’s own published architecture, before running a single benchmark — a predicted curve of tokens-per-second against how many expert layers you push off the GPU. Then you’ll find out where the VRAM budget from [[module-4-vram-budget]] forces a *minimum* amount of offload at any given context length, which turns "more VRAM headroom is better" into a genuine constrained optimization, not just a knob to turn until it feels fast. Then you measure the real curve and see how well the prediction survived contact with your card. No public `--n-cpu-moe` benchmark exists for the RX 9070 XT — whatever curve you produce here is very likely the first one anyone has published for this GPU on this model.

---

## 5.1 The flags: `n-cpu-moe`, `cpu-moe`, and `override-tensor`

> [!abstract] Goal
> Know exactly what each expert-placement flag does and which one to reach for in which situation, before touching any of them.

**What’s going on:** `llama.cpp` gives you three related but distinct mechanisms for controlling where [[concepts#MoE|MoE]] expert weights live:

- **`--n-cpu-moe N`** (short form `-ncmoe N`) keeps the routed-expert FFN weights of the *first N transformer layers* resident in CPU RAM, offloading the rest to GPU as usual. This is the primary tool for this module — a single integer that sweeps smoothly from "everything on GPU" (`N=0`) to "as much as possible on CPU" (`N=48`, the full layer count for this model).
- **`--cpu-moe`** (short form `-cmoe`) is the extreme case of the same idea: keep *all* expert weights in CPU RAM, unconditionally. It’s `--n-cpu-moe 48` spelled as a fixed flag rather than a swept parameter — useful as a known reference point, not something you’ll sweep.
- **`-ot` / `--override-tensor`**, given a regex, is the older and strictly more general mechanism: match tensor names by pattern and force each match’s placement individually. `--n-cpu-moe` is convenience syntax built on top of the same underlying capability, restricted to a contiguous "first N layers" range. `-ot` predates it — it landed in PR #11397 — and it’s still the tool you need whenever your placement scheme isn’t a clean contiguous prefix, most importantly for **multi-GPU splits**, where you might want layers 0–10 and 30–40 on GPU 0, 11–29 on GPU 1, and some non-contiguous subset on CPU — a shape `--n-cpu-moe`’s single integer simply cannot express.

`--n-cpu-moe` itself is recent: it landed in PR #15077 (August 4, 2025). If you’re reading an older guide or forum post about MoE offload on `llama.cpp` and it only mentions `-ot`, that’s why — the convenience flag didn’t exist yet when it was written.

**Steps:**
1. Run `llama-server --help` (or `llama-cli --help`) and confirm `--n-cpu-moe` / `-ncmoe` and `--cpu-moe` / `-cmoe` are present in your build — if they’re missing, your `llama.cpp` checkout predates PR #15077 and needs updating before this module works as written.
2. Load the model once with `--n-cpu-moe 0` and once with `--cpu-moe`, and confirm from the startup logs that the second one reports every expert-FFN tensor placed on CPU, while the first reports none.
3. Read the `-ot` section of the `llama.cpp` documentation or `--help` output and note the regex syntax it expects — you’ll write real patterns with it in 5.6.

> [!success] Done when
> You can state, without checking, which of the three flags to reach for: a single-GPU contiguous sweep (`--n-cpu-moe`), the all-CPU reference point (`--cpu-moe`), and a non-contiguous or multi-GPU placement (`-ot`).

> [!question]- It’s not working
> 1. `--n-cpu-moe` not recognized at all? Check your `llama.cpp` build date against August 4, 2025 (PR #15077) — you likely need to pull and rebuild.
> 2. Model loads but startup logs don’t show any tensors moved to CPU even with a nonzero `--n-cpu-moe`? Confirm the model is actually MoE — a dense model has no expert tensors for this flag to act on, and it will silently do nothing.

## 5.2 Why placement works: read frequency, not size

> [!abstract] Goal
> Understand the actual mechanism that makes `--n-cpu-moe` a good idea for this model, and confirm — precisely — the one condition under which it stops being a good idea at all.

**What’s going on:** The instinct "put the big stuff on the slow pool, keep the small stuff on the fast pool" is close, but it’s the wrong frame — capacity alone doesn’t tell you what to do, *read frequency per token* does. Attention weights (Q/K/V/O projections), normalization layers, the token embedding table, and the MoE router itself are all read on **every single token**, regardless of model size or routing decisions — they’re a small fraction of total bytes, and they belong in VRAM because the fast pool’s bandwidth advantage compounds every time something is re-read.

Routed expert FFN weights are the overwhelming majority of this model’s bytes — 17.41 GB out of 18.30 GB total, from the split in 5.3 below — but only the **top-8 of 128** experts fire for any given token. The *effective* bytes streamed per token from the expert pool is a small fraction of the pool’s total size, even though the pool itself is huge. That’s the entire justification for `--n-cpu-moe`: you’re not choosing to read less data by moving experts to DDR5, you’re choosing to pay DDR5’s lower bandwidth **only for the fraction of expert weights that actually get read on a given token**, while everything read unconditionally every token stays on the fast pool.

**This mechanism only exists because the model is MoE.** A dense model has no routing — every weight in every layer is read on every token, full stop. There is no "effective bytes per token is smaller than total bytes" trick available, because there’s no sparsity to exploit. Moving a dense model’s weights to CPU RAM just means paying DDR5 bandwidth for the *entire* weight set, every token, with nothing to offset it — which is exactly the "dense-equivalent baseline" you’ll compute in 5.4 and which turns out to be catastrophically slow by comparison. Keep this distinction sharp: `--n-cpu-moe` is a MoE-specific trick, not a general "how to run big models on small GPUs" trick.

**Steps:**
1. Name, from memory, the four categories of weights that should stay in VRAM regardless of `--n-cpu-moe` setting (attention Q/K/V/O, norms, embeddings, router) and state why each is read every token.
2. State, in one sentence, why "17.41 GB of expert weights" and "1.088 GB of *effective* expert reads per token" are both true numbers about the same model, and are not in tension with each other.
3. Explain, without looking it up, why running a dense (non-MoE) model with `--n-cpu-moe`-style offload would not give you the same win — tie this back to the "every weight, every token" property of dense models.

> [!success] Done when
> You can explain the read-frequency mechanism, in your own words, to a satisfaction that would survive being asked "then why doesn’t this help a dense model?"

## 5.3 Predict the curve — the weight split and the active-bytes model

> [!abstract] Goal
> Derive the full predicted tok/s-vs-`n_cpu_moe` curve from the model’s own weight split and your Module 0/1 bandwidth figures, before running a single sweep.

**What’s going on:** This is the course’s signature derivation, and the discipline matters: **predict first, sweep second.** A blind sweep just produces a scatter of numbers with no model behind them — you’d be able to report *what* happened but not *why*, and you’d have no way to know if your measured curve is behaving sensibly or hiding a bug.

Start from the weight split at Q4_K_M quantization, from Module 2’s numbers:

- Non-expert weights: 1.49 B params → **0.89 GB**
- Expert weights: 29.01 B params → **17.41 GB**
- Total: **18.30 GB**

Only 8 of 128 experts fire per token, so the *active* expert bytes read per token, if every one of them were resident somewhere:

```
active_expert_bytes = (8/128) × 17.41 GB = 1.088 GB
```

Now split those active bytes between the two pools as a function of `n` (the `--n-cpu-moe` value — number of layers whose experts sit in CPU RAM), assuming active expert reads are spread evenly across the model’s 48 layers:

```
D(n) = (n/48)      × 1.088 GB     — active expert bytes read from DDR5
V(n) = 0.89 GB + ((48−n)/48) × 1.088 GB   — everything read from VRAM
```

`V(n)` includes the fixed 0.89 GB of always-resident non-expert weights *plus* whatever fraction of the active expert bytes still lives on the GPU-resident layers. Then the per-token time is the sum of the two pools’ individual costs — this is the same `t_token(n)` equation from the course abstract, made concrete for this model:

```
t_token(n) = V(n)/BW_vram + D(n)/BW_ddr
```

Using the **theoretical** figures — 640 GB/s VRAM, 57.6 GB/s DDR5 — not yet your Module 0/1 *measured* numbers, gives this predicted table:

| n | D(n) GB | V(n) GB | t_token (ms) | tok/s |
|---|---|---|---|---|
| 0 | 0.000 | 1.980 | 3.09 | 323 |
| 8 | 0.181 | 1.799 | 5.96 | 168 |
| 16 | 0.363 | 1.617 | 8.82 | 113 |
| 18 | 0.408 | 1.572 | 9.54 | 105 |
| 24 | 0.544 | 1.436 | 11.69 | 86 |
| 32 | 0.725 | 1.255 | 14.55 | 69 |
| 48 | 1.088 | 0.892 | 20.28 | 49 |

**These are theoretical-bandwidth predictions, not measurements — treat every number in this table as a hypothesis until 5.5.** Your real curve will very likely sit below this one everywhere, the same way η_model in Module 2 sat below the naive VRAM-only prediction, because this table still ignores kernel launch overhead, attention compute, and the same real-world inefficiencies η_model was invented to absorb.

**For contrast — the dense-equivalent baseline:** if this were a dense model of the same 18.30 GB total size, running fully from DDR5 (no VRAM residency at all, the worst case), every one of the 18.30 GB of weights has to stream per token:

```
t_dense = 18.30 GB / 57.6 GB/s = 317.7 ms → 3.15 tok/s
```

Compare that to the `n=18` row above — **~105 tok/s** — a roughly **33× win** from MoE sparsity plus placement, over a same-sized dense model with no placement options at all. That single number is the reason this module — and arguably this course — exists.

**Steps:**
1. Recompute `D(n)` and `V(n)` for at least three rows of the table yourself, from the formulas, not by copying the table.
2. Recompute `t_dense` and confirm the ~33× figure independently.
3. Sketch the predicted tok/s-vs-`n` curve on paper (or in a spreadsheet) — note its shape: monotonically decreasing, steepest at low `n`, flattening out toward `n=48`. That shape is itself a prediction you’ll check against the real sweep.

> [!success] Done when
> You have your own copy of the predicted table, independently recomputed, plus the ~33× dense-comparison figure, and a sketched curve shape you can compare the real sweep against.

> [!bug] Gotcha
> Every number in this section uses the *theoretical* 640 GB/s and 57.6 GB/s bandwidth figures from [[workstation-specs]] — not your Module 0/1 *measured* bandwidth or your Module 2 η_model correction factor. That’s deliberate: this is the naive, first-pass prediction. Don’t quietly substitute your measured numbers in here and call the result "predicted" — keep this table as the theoretical baseline, and build a second, corrected prediction using your own measured bandwidths and η_model if you want a tighter target to check the real sweep against.

## 5.4 VRAM capacity dictates a minimum n

> [!abstract] Goal
> Combine this module’s active-bytes model with Module 4’s budget equation to find the *smallest* `n_cpu_moe` that actually fits in 16 GB at a given context length — because the fastest row of the 5.3 table (`n=0`) is not always a legal configuration.

**What’s going on:** The 5.3 table makes `n=0` look best — lowest `t_token`, highest tok/s, every time. But `n` is not a free parameter you tune purely for speed; it’s constrained from below by how much VRAM is left after weights, KV cache, and overhead have taken their share, using the exact budget equation from [[module-4-vram-budget]]:

```
budget = 16 GB − 0.89 GB (non-expert weights) − KV_bytes(ctx) − ~1.0 GB (compute buffers, from your 4.4 measurement)
```

Each resident expert layer costs `17.41 GB / 48 layers = 0.363 GB`. The number of layers you can afford to keep fully resident is `floor(budget / 0.363)`, and `n_cpu_moe` is whatever’s left: `48 − resident_layers`.

Working through three context lengths, using the 4.1 KV figures:

- **8K context:** KV = 0.81 GB → budget = 16 − 0.89 − 0.81 − 1.0 = **13.30 GB** → 36 layers resident → **`n_cpu_moe = 12`** → predicted **~135 tok/s** (interpolating the 5.3 table)
- **32K context:** KV = 3.22 GB → budget = 16 − 0.89 − 3.22 − 1.0 = **10.89 GB** → 30 layers resident → **`n_cpu_moe = 18`** → predicted **~105 tok/s**
- **64K context:** KV = 6.44 GB → budget = 16 − 0.89 − 6.44 − 1.0 = **7.67 GB** → 21 layers resident → **`n_cpu_moe = 27`** → predicted **~78 tok/s**

Notice the shape of this result: **longer context doesn’t just cost you KV cache VRAM — it indirectly forces slower decode, too**, because the KV cache eats into the same budget that would otherwise hold experts on the fast pool. A context-length decision and an expert-placement decision are not independent choices; the budget equation is what couples them.

**Steps:**
1. Using your own 4.4 measured overhead constant (not necessarily the ~1.0 GB placeholder above, if your measurement gave you a different number), recompute the budget and minimum `n_cpu_moe` for 8K, 32K, and 64K context.
2. Pick one additional context length not covered above (e.g., 16K or 96K) and work through the same calculation yourself.
3. State, in one sentence, why `n=0` — the fastest row in the 5.3 table — is not a legal configuration at any of these context lengths once KV cache is accounted for (check: does `resident_layers` at `n=0`, i.e. all 48 layers resident, actually fit in your computed budget at 8K? If not, you’ve just shown `n=0` is infeasible even at the shortest context you’re testing).

> [!success] Done when
> You have your own minimum-`n_cpu_moe` figure at three context lengths, using your own measured overhead constant, and a written statement of whether `n=0` is ever actually achievable given real KV cache and overhead costs.

## 5.5 Run the real sweep

> [!abstract] Goal
> Measure tok/s across a real `--n-cpu-moe` sweep and lay the measured curve directly against the 5.3 prediction.

**What’s going on:** Everything up to this point has been on paper. This is where the model meets the GPU.

**Steps:**
1. Pick a fixed context length to hold constant across the whole sweep — 32K is a reasonable default, since it has a clean minimum `n_cpu_moe` from 5.4 (18) to check against.
2. Sweep `--n-cpu-moe` across a range that spans below and above your computed minimum for that context — e.g., 0, 8, 12, 16, 18, 24, 32, 40, 48 — running each with `llama-bench`:
   ```bash
   ./build-hip/bin/llama-bench -m qwen3-coder-30b-a3b-q4_k_m.gguf \
       -c 32768 -ncmoe <N> -t <your-elbow-thread-count> \
       -p 512 -n 128 -r 5 -o md
   ```
3. For any `N` below your computed minimum (12 layers less than 48, i.e. `n < 18` at 32K context in the 5.4 example), **expect and record an allocation failure or a silent CPU spillover** rather than a clean run — this is the budget constraint from 5.4 showing up as a real failure, not a bug in your sweep script.
4. Tabulate measured `tg128` tok/s (avg ± stddev across the `-r 5` repetitions) against your `n` values, and against the predicted values from the 5.3 table.
5. Plot both curves (predicted and measured) on the same axes — `n_cpu_moe` on x, tok/s on y — and visually locate the knee: the point where increasing `n` further stops buying you much more VRAM headroom relative to the tok/s you’re giving up.

> [!success] Done when
> You have a measured tok/s-vs-`n_cpu_moe` curve, plotted against the 5.3 prediction, spanning at least the range from your 5.4-computed minimum down through `n=48`, with the real knee identified and compared to where the prediction placed it.

> [!question]- It’s not working
> 1. Measured tok/s is *higher* than predicted at every `n`? Double-check you’re not accidentally hitting a cached KV state from a previous run — restart the server between sweep points.
> 2. Measured tok/s collapses far below prediction at low `n`? Check the startup log for silent partial CPU spillover of non-expert weights or KV cache itself — that would mean your VRAM budget assumption from 5.4 was too optimistic (your overhead constant may need revising).
> 3. Stddev unusually large at particular `n` values? MoE routing means the *actual* experts read per token vary slightly by input — this is expected variance, not a bug, but note it, since it means the active-bytes model in 5.3 is an average-case prediction, not an exact one.

> [!tip] A tie is a valid, publishable outcome
> If your measured curve shows `--n-cpu-moe` already placing tensors close to optimally — i.e., hand-tuning in 5.6 barely beats it — **that is not a failed module.** It’s a genuine, useful negative result: it means the convenience flag’s simple "first N layers" heuristic is already good enough for this model shape, and nobody publishing `-ot` regexes for this model needed to. Plan for this outcome in advance so it doesn’t feel like the module didn’t work.

## 5.6 Hand-tune with `-ot` regexes

> [!abstract] Goal
> Write explicit `--override-tensor` regexes targeting the same placement `--n-cpu-moe` gives you, and see whether hand-tuning beats the convenience flag.

**What’s going on:** `--n-cpu-moe N` places the *first N layers’* experts on CPU — a specific, contiguous heuristic. `-ot` lets you express any placement a regex can describe, including things `--n-cpu-moe` cannot: gate-only offload, up/down-only offload, or a non-contiguous layer selection. One important note specific to this model: **Qwen3-Coder-30B-A3B has no shared experts** (unlike DeepSeek-V2/V3, which route every token through both a shared expert and a set of routed experts). Guides written for DeepSeek-family models sometimes use a bare `exps=CPU` shortcut that relies on shared-expert tensor naming conventions — that shortcut does not map identically onto this model’s tensor names, so use the explicit `ffn_(up|gate|down)_exps` pattern below instead of copying a DeepSeek-oriented regex verbatim.

Useful patterns:

```bash
# All routed-expert FFN tensors to CPU (roughly equivalent to --cpu-moe)
-ot ".ffn_.*_exps.=CPU"

# Only up/down projections to CPU; gate stays on GPU
-ot ".ffn_(up|down)_exps.=CPU"

# Layers 19 and up (of 48) to CPU — hand-written equivalent of --n-cpu-moe 29
-ot "blk\.(19|[2-9][0-9])\.ffn_(up|gate|down)_exps\.weight=CPU"
```

**Steps:**
1. Write an `-ot` regex that reproduces your 5.4-computed minimum `n_cpu_moe` for your chosen context length exactly (i.e., the same layer range `--n-cpu-moe` would have selected), and confirm via startup logs that the set of CPU-placed tensors matches.
2. Benchmark that hand-written regex with the identical `llama-bench` invocation from 5.5, and compare tok/s directly against the `--n-cpu-moe` run at the same `n`.
3. Try at least one placement `--n-cpu-moe` cannot express — e.g., gate tensors staying on GPU while up/down move to CPU — and measure whether it beats, ties, or loses to the plain `--n-cpu-moe` baseline.
4. Record the delta, in tok/s, between your best hand-tuned `-ot` configuration and the `--n-cpu-moe` baseline at the same effective placement.

> [!success] Done when
> You have a measured tok/s delta between hand-tuned `-ot` and the equivalent `--n-cpu-moe` setting — whether that delta favors hand-tuning, favors the convenience flag, or is within your Module 3-style noise floor.

> [!question]- It’s not working
> 1. Regex matches zero tensors (startup log shows nothing moved)? Print the model’s actual tensor names (`llama-cli` verbose loading output, or `gguf-dump` if available) and match your regex against real names rather than guessing the naming convention.
> 2. Regex matches *more* than intended (e.g., it also catches attention tensors)? Tighten the pattern — `ffn_.*_exps` is deliberately scoped to expert FFN tensors; a looser pattern like a bare `.*` will over-match.

## 5.7 Build the VRAM budget allocator

> [!abstract] Goal
> Turn the 5.4 budget arithmetic into a small script: given 16 GB of VRAM, a model’s weight split, and a target context length, it should emit the right `--n-cpu-moe` value directly.

**What’s going on:** This is the deliverable worth sharing out of this module. Everything in 5.4 was arithmetic you did by hand for three context lengths — the actual useful artifact is a script that does that arithmetic for *any* context length, and for any model whose weight split and layer count you can supply, so you (or anyone else reading this course) never has to redo the by-hand version again.

**Steps:**
1. Encode the constants: total VRAM (16 GB), non-expert weight size (0.89 GB), expert weight size (17.41 GB), layer count (48), your measured overhead constant from 4.4, and the KV-bytes-per-token figure (98,304) from 4.1.
2. Write a function that takes a target context length and returns: KV cache size, remaining budget, resident-layer count, and the resulting `n_cpu_moe`.
3. Have it print the exact `--n-cpu-moe N` (or full `llama-server` invocation) a user should run for that target context.
4. Validate it against your own three hand-worked examples from 5.4 — it should reproduce `n_cpu_moe = 12, 18, 27` at 8K/32K/64K exactly.

```python
#!/usr/bin/env python3
"""VRAM budget allocator for Qwen3-Coder-30B-A3B on a 16GB card.
Given a target context length, print the minimum --n-cpu-moe that fits.
"""

VRAM_TOTAL_GB     = 16.0
NON_EXPERT_GB     = 0.89
EXPERT_TOTAL_GB   = 17.41
NUM_LAYERS        = 48
OVERHEAD_GB       = 1.0        # replace with your own 4.4 measured constant
KV_BYTES_PER_TOK  = 98_304     # from module 4.1 -- recompute for other models

BYTES_PER_GB = 1_000_000_000
EXPERT_GB_PER_LAYER = EXPERT_TOTAL_GB / NUM_LAYERS

def kv_gb(context_tokens: int) -> float:
    return (KV_BYTES_PER_TOK * context_tokens) / BYTES_PER_GB

def recommend_n_cpu_moe(context_tokens: int) -> dict:
    kv = kv_gb(context_tokens)
    budget = VRAM_TOTAL_GB - NON_EXPERT_GB - kv - OVERHEAD_GB
    if budget <= 0:
        raise ValueError(
            f"context {context_tokens} alone needs more than fits in "
            f"{VRAM_TOTAL_GB} GB after weights and overhead -- reduce context "
            f"or use KV quantization (module 9)."
        )
    resident_layers = min(NUM_LAYERS, int(budget // EXPERT_GB_PER_LAYER))
    n_cpu_moe = NUM_LAYERS - resident_layers
    return {
        "context": context_tokens,
        "kv_gb": round(kv, 2),
        "budget_gb": round(budget, 2),
        "resident_layers": resident_layers,
        "n_cpu_moe": n_cpu_moe,
    }

if __name__ == "__main__":
    for ctx in (8192, 32768, 65536):
        result = recommend_n_cpu_moe(ctx)
        print(
            f"ctx={result['context']:>6}  KV={result['kv_gb']:>5} GB  "
            f"budget={result['budget_gb']:>5} GB  "
            f"resident_layers={result['resident_layers']:>2}  "
            f"--n-cpu-moe {result['n_cpu_moe']}"
        )
```

> [!success] Done when
> The script reproduces `n_cpu_moe = 12, 18, 27` at 8K/32K/64K context using your own measured overhead constant, and can be pointed at a different target context length to emit a usable `--n-cpu-moe` value without any by-hand arithmetic.

---

## Check your understanding

> [!question]- 1. Why does `--n-cpu-moe` help a MoE model’s decode speed but do nothing useful for a dense model?
> > [!success]- Answer
> > `--n-cpu-moe` works by exploiting the gap between total expert weight size and the much smaller *effective* bytes read per token, because only a handful of experts (8 of 128, here) actually fire for any given token — the rest of the 17.41 GB of expert weights simply aren’t touched that step. A dense model has no such gap: every weight in every layer is read on every token, full stop, so moving weights to CPU RAM just means paying DDR5’s lower bandwidth for the *entire* weight set every token, with no sparsity to offset the cost. There’s no analogous "only a fraction gets read" property to exploit.

> [!question]- 2. At 64K context, why is `n_cpu_moe=27` the *minimum* rather than a target you’d only reach for extra headroom?
> > [!success]- Answer
> > Because at 64K context, the KV cache alone costs 6.44 GB, and after subtracting the fixed 0.89 GB of non-expert weights and roughly 1.0 GB of compute-buffer overhead from 16 GB, only 7.67 GB remains for resident expert layers — enough for 21 of 48 layers, no more. Any `n_cpu_moe` smaller than 27 (i.e., trying to keep more than 21 layers resident) would require more VRAM than is actually available at that context length, and would fail to allocate or silently spill elsewhere rather than run faster.

> [!question]- 3. Your measured sweep shows hand-tuned `-ot` regexes landing within your Module 3-style noise floor of `--n-cpu-moe` at the equivalent placement. Did this module fail?
> > [!success]- Answer
> > No — this is one of the explicitly anticipated outcomes, not a failure. It’s a genuine, publishable negative result: it means `--n-cpu-moe`’s simple "first N contiguous layers" heuristic is already close to optimal for this model’s shape, and the extra complexity of hand-written `-ot` regexes buys nothing measurable here. That’s useful information for anyone deciding whether hand-tuning is worth their time on this model — the module’s job was to measure the delta, not to guarantee hand-tuning would win.

## What’s next

You now have a predicted curve derived from the model’s own weight split, a VRAM-budget-constrained minimum for `n_cpu_moe` at several context lengths, a real measured sweep checked against both, and a script that does the budget arithmetic for you. But everything measured so far has been pure decode — `tg128`, one token at a time, after whatever prompt was already sitting in context. [[module-6-prefill-decode]] introduces the other half of a real coding session: the prefill pass that processes a 10–40K-token prompt *before* decode even starts, and asks whether the placement scheme that just won you decode speed might be quietly costing you far more at the start of every turn.

<script src="/tutor.js" defer></script>
