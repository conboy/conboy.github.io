---
title: "Module 4 — The VRAM Budget"
description: "Computing KV cache from the formula, measuring what llama.cpp actually allocates, and deriving the overhead constant that closes the gap."
tags:
  - local-ai
  - vram
  - kv-cache
  - memory-budget
---

# Module 4 — The VRAM Budget

**6 hours** · Part of [[index|Two Pools]] · Prev: [[module-3-eval-instrument]] · Next: [[module-5-expert-placement]]

> [!info] What this module is for
> Module 2 watched Qwen3-Coder-30B-A3B fail to load in 16 GB and left it at "weights alone exceed capacity." That was true, but it was also the easy half of the story — weights are a fixed, known number, and the rest of your VRAM budget is context-dependent, and largely invisible until you go looking for it. This module builds the piece that was missing: the [[concepts#KV cache|KV cache]] formula, applied to this exact model, at the context lengths you actually care about. Then it does the thing the formula alone cannot do — it measures what `llama.cpp` really allocates on the card, because the formula only accounts for the KV tensors themselves, and a running server also holds compute buffers, allocator padding, and other overhead that no back-of-envelope calculation captures. The gap between predicted KV bytes and measured VRAM growth gets a name here too, the same way η_model did in Module 2, and it becomes the last piece of a complete VRAM budget equation — the equation Module 5 spends twelve hours exploiting.

---

## 4.1 The KV cache formula, applied to Qwen3-Coder-30B-A3B

> [!abstract] Goal
> Derive the exact bytes-per-token figure for this model from its architecture config, before touching a GPU.

**What’s going on:** The KV cache stores the key and value projections for every token, for every attention layer, so that generating token *N+1* doesn’t require recomputing attention over tokens *1…N* from scratch. The formula is:

```
KV_bytes = 2 × layers × kv_heads × head_dim × context × bytes_per_element
```

The leading `2` is for the two tensors being cached — keys and values — not a fudge factor. `bytes_per_element` is 2 for FP16, 1 for Q8_0 KV quantization (a Module 9 topic, not this one — assume FP16 here).

Qwen3-Coder-30B-A3B’s config gives you: 48 transformer layers, 4 KV heads, head_dim 128. Plug those in:

```
2 × 48 × 4 × 128 × 2 = 98,304 bytes/token
```

That’s the number this whole module and the next one are built on. Scaled to real context lengths (decimal GB, 1e9 bytes):

| Context | KV bytes | KV size |
|---|---|---|
| 8K | 805,306,368 | 0.81 GB |
| 32K | 3,221,225,472 | 3.22 GB |
| 128K | 12,884,901,888 | 12.88 GB |
| 256K | 25,769,803,776 | 25.77 GB |
| 1M | 103,079,215,104 | 103.08 GB |

Sit with the 256K row. 262,144 tokens is this model’s own advertised *native* maximum context — not some artificially inflated marketing figure — and the KV cache alone at that length is **25.77 GB, more than your entire 16 GB card, before a single weight is loaded.** That tension — a model whose native max context cannot possibly run on this hardware, no matter how you place the weights — is deliberately left open here. It’s Capstone 2’s problem, not this module’s, but you should feel the weight of it now.

**Steps:**
1. Find Qwen3-Coder-30B-A3B’s `config.json` (or the equivalent metadata printed by `llama-cli` at load time) and confirm `num_hidden_layers`, `num_key_value_heads`, and `head_dim` (or derive `head_dim` from `hidden_size / num_attention_heads` if it isn’t listed directly).
2. Recompute the 98,304 bytes/token figure yourself from those numbers — don’t just copy it down.
3. Fill in the table above for at least 8K, 32K, and 128K by hand, in GB, before you check them against the ones printed here.

> [!success] Done when
> You’ve derived 98,304 bytes/token from the model’s own config values, independently, and produced your own version of the table above for at least three context lengths.

> [!question]- It’s not working
> 1. Getting a much larger number? You’re almost certainly using the 32 *attention* heads instead of the 4 *KV* heads — see 4.2, this is the single most common mistake with this formula.
> 2. Numbers off by exactly 2×? Check whether you’re double-counting the leading `2` (K and V) somewhere else in your arithmetic, or forgetting it entirely.
> 3. Config file shows an unfamiliar field name for `head_dim`? Some model configs only list `hidden_size` and `num_attention_heads` — divide the former by the latter to recover it.

## 4.2 GQA: why 4, not 32

> [!abstract] Goal
> Understand *why* the KV cache formula uses 4 KV heads instead of the 32 attention heads Qwen3-Coder-30B-A3B actually has, and quantify the saving.

**What’s going on:** [[concepts#GQA|Grouped-Query Attention]] (GQA) is the single detail most likely to make you compute a KV cache figure that’s 8× too large. In standard multi-head attention, every one of the 32 attention heads has its own K and V projection, and the KV cache has to store all 32. GQA changes this: multiple query heads *share* a smaller number of K/V head groups. Qwen3-Coder-30B-A3B has 32 query heads but only 4 KV heads — each KV head is shared across 8 query heads (32 ÷ 4 = 8). The queries are still full-resolution; only the keys and values being cached are shrunk.

The saving is direct and multiplicative: using 4 KV heads instead of 32 makes the KV cache **8× smaller** at every context length in the table above, for identical attention quality within the range GQA was tuned for. If you’d used 32 in the formula, the 128K row would read **103.08 GB** instead of 12.88 GB — a number that makes context planning look catastrophically worse than it actually is, for no real reason.

This is worth internalizing precisely because it’s a recurring category of mistake: **skimming a model card for "attention heads" and plugging that number into a KV cache formula that wants "KV heads."** They are only the same number in old-style multi-head attention (no GQA) — anything from the last few years’ worth of serious open models almost certainly uses GQA or a variant, and the two numbers will differ.

**Steps:**
1. Confirm, from the config, that `num_attention_heads` (32) and `num_key_value_heads` (4) are genuinely different fields, not a typo or your own misreading.
2. Recompute the 128K KV cache figure using 32 instead of 4 in the formula, and confirm you get 103.08 GB — the same order of magnitude as the 1M-context figure from 4.1, which should feel wrong the moment you see it, because it is.
3. State, in one sentence, the query-head-to-KV-head ratio (8:1) and what it means physically: 8 query heads read from the same cached K/V pair.

> [!success] Done when
> You can state the 8:1 ratio from memory, explain why it produces an 8× KV cache saving, and you’ve confirmed by direct substitution that using 32 instead of 4 would have given you the wrong 128K figure by exactly that factor.

> [!bug] Gotcha
> If a model card or benchmark script only reports "heads" without specifying which kind, do not guess. Open the actual `config.json`. `num_attention_heads` and `num_key_value_heads` are both real, both present, and only equal to each other in the absence of GQA — assuming they’re the same field is the single most common way to get a KV budget wrong by a clean integer multiple.

## 4.3 MoE does not change the KV cache — and what MLA does instead

> [!abstract] Goal
> Separate what routing affects (FFN compute) from what it doesn’t (attention, and therefore KV cache), and note the one architecture that changes the KV cache formula itself.

**What’s going on:** It’s tempting to assume a Mixture-of-Experts model’s KV cache is somehow smaller, because so much else about it is — after all, "only 8 of 128 experts fire per token" sounds like the kind of sparsity that should show up everywhere. It doesn’t show up here. [[concepts#MoE|MoE]] routing operates on the **feed-forward (FFN) sublayer** — deciding which expert’s FFN weights process a given token. Attention — the Q/K/V/O projections, and therefore the KV cache — is a completely separate sublayer, computed identically regardless of which experts get routed to afterward. Every token still produces a full K and V vector at every layer, cached in full, whether the model is dense or MoE. The 98,304 bytes/token figure from 4.1 would be *identical* for a dense model with the same layer count, KV head count, and head_dim — MoE-ness is invisible to this formula entirely.

There is exactly one architectural change that *does* alter the KV cache formula, and it’s worth knowing about even though Qwen3-Coder-30B-A3B doesn’t use it: **[[concepts#MLA|Multi-head Latent Attention]] (MLA)**, introduced in DeepSeek-V2. Instead of caching full-size K and V tensors per head, MLA caches a single compressed low-rank latent vector per token, and reconstructs per-head K/V from it on the fly at attention time. DeepSeek reported a **93.3% KV cache reduction** compared to their own prior dense multi-head attention baseline — a change of the same order of magnitude as GQA’s 8× saving, achieved by a completely different mechanism (compression instead of head-sharing). If you ever see a KV-heavy budget problem on a DeepSeek-family model and the 4.1 formula doesn’t match reality, MLA is very likely why — the formula in this module assumes GQA/MHA-style caching and does not apply as written to an MLA model.

**Steps:**
1. State in one sentence why MoE routing has no effect on the KV cache formula — name the sublayer routing touches (FFN) and the sublayer the formula describes (attention).
2. Look up which sublayer(s) MLA compresses, and confirm it’s the same attention sublayer GQA modifies — MLA and GQA are two different answers to the same problem (attention KV cache is expensive), not layered on top of each other in most model families.
3. Write down the 93.3% figure and which model pair (DeepSeek-V2 vs. its dense predecessor) it was measured against — this is a reported figure from DeepSeek’s own published comparison, not something to treat as universal.

> [!success] Done when
> You can state, without hedging, that Qwen3-Coder-30B-A3B’s MoE-ness has zero effect on its KV cache size — and that if it used MLA instead of GQA, the 4.1 formula would need to be replaced, not just re-parameterized.

## 4.4 Measure what `llama.cpp` actually allocates

> [!abstract] Goal
> Load the model at several context lengths and measure real VRAM growth with `rocm-smi`, to find out how much the 4.1 formula under-counts.

**What’s going on:** The formula in 4.1 computes the size of the KV *tensors themselves*. It says nothing about the compute buffers `llama.cpp` allocates for intermediate activations during a forward pass, allocator alignment padding, or any per-context-size scratch space the backend reserves up front. All of that is real VRAM, it scales somewhat with context length, and no formula in this module predicts it — which is exactly why you’re about to measure it instead of trusting arithmetic.

**Steps:**
1. Record idle VRAM usage before loading anything:
   ```bash
   rocm-smi --showmeminfo vram
   ```
2. Load Qwen3-Coder-30B-A3B with a small context and full offload where it fits, or a fixed, noted partial-offload configuration if it doesn’t — the point here is holding everything except context length constant across repeats, not achieving full residency:
   ```bash
   ./build-hip/bin/llama-server -m qwen3-coder-30b-a3b-q4_k_m.gguf \
       -ngl 99 -c 8192 --host 0.0.0.0 --port 8080
   ```
3. With the server up and idle (no requests in flight), record VRAM again:
   ```bash
   rocm-smi --showmeminfo vram
   ```
4. Compute `measured_delta = vram_after - vram_before`, and separately compute `predicted_kv = 98,304 bytes × context` from 4.1 for the same context length.
5. Repeat steps 2–4 at 32K and 128K context (`-c 32768`, `-c 131072`), restarting the server cleanly between runs so allocator state doesn’t carry over.
6. Tabulate `measured_delta`, `predicted_kv`, and `overhead = measured_delta - predicted_kv` for all three context lengths.

> [!success] Done when
> You have a table of measured VRAM delta vs. predicted KV bytes at 8K, 32K, and 128K context, with the overhead column computed at each row.

> [!question]- It’s not working
> 1. Is `rocm-smi --showmeminfo vram` reporting total card memory instead of your process’s allocation? Cross-check against a second tool (e.g., `rocm-smi` without flags, or `nvtop`-equivalent for AMD if installed) if the numbers look like the full 16 GB regardless of what’s loaded.
> 2. Is the delta *negative* or near-zero at larger contexts? You may be spilling to CPU RAM instead of allocating on the GPU — check the server’s startup log for a partial-offload warning, the same one you learned to watch for in Module 2.
> 3. Getting wildly different overhead numbers between runs of the *same* context length? Confirm you’re measuring at true idle (no in-flight request, no warm-up generation still running) both before and after.

> [!bug] Gotcha
> `-ngl 99` forcing full GPU offload of weights does not mean the KV cache and compute buffers "just fit" — they are allocated *in addition to* the weights, and a config that loads cleanly at 8K context can fail to allocate at 128K purely from KV and overhead growth, even though the weights themselves haven’t changed size at all.

## 4.5 The complete budget equation

> [!abstract] Goal
> Fold the measured overhead from 4.4 into a single equation you can use to predict, not just explain, how much VRAM a given context length will actually consume.

**What’s going on:** The formula from 4.1 alone is not the budget — it’s the *predictable* term. 4.4 gave you the correction term. Put together:

```
VRAM_required(ctx) = weights_resident + KV_bytes(ctx) + overhead(ctx)
```

where `weights_resident` is whatever fraction of the 18.30 GB total you’ve placed on GPU (all of it, if it fits; a partial amount once Module 5’s placement scheme is in play), `KV_bytes(ctx)` is the 4.1 formula, and `overhead(ctx)` is whatever constant or slowly-growing term your 4.4 measurements actually showed — state it as a flat constant if your three data points were roughly context-independent, or as a term with its own rough per-context growth if they clearly weren’t.

This equation is the one Module 5 spends its entire 12 hours exploiting — the "budget" in that module’s minimum-`n_cpu_moe` calculation is exactly this equation, solved for how many expert layers can stay resident once KV and overhead have taken their cut of 16 GB.

**Steps:**
1. From your 4.4 table, decide whether `overhead(ctx)` is closer to a flat constant across the three context lengths, or grows noticeably with context — write down which, and the actual number(s).
2. Write the complete equation for your own hardware, with real numbers substituted for `weights_resident` (18.30 GB fully resident, or a smaller number if you tested a partial-offload configuration) and your measured `overhead`.
3. Use the equation to predict VRAM usage at a context length you did *not* directly measure (e.g., 64K), then briefly sanity-check that prediction against a real load at that context length.

> [!success] Done when
> You have a single written equation, with your own measured overhead constant substituted in, and one held-out context length where the equation’s prediction was checked against a real measurement.

---

## Check your understanding

> [!question]- 1. Why doesn’t `llama.cpp`’s memory usage match the 4.1 formula exactly?
> > [!success]- Answer
> > The 4.1 formula only accounts for the KV cache tensors — the actual K and V values stored per token, per layer, per KV head. A running server also allocates compute buffers for intermediate activations during the forward pass, plus whatever padding or alignment overhead the memory allocator adds. None of that is captured by a formula that only describes the KV tensors themselves, which is exactly why 4.4 measures it instead of computing it.

> [!question]- 2. A model has 32 attention heads and 8 KV heads. What’s the KV cache saving from GQA, and how many query heads share each KV head?
> > [!success]- Answer
> > 32 ÷ 8 = 4 query heads share each KV head, giving a 4× KV cache saving compared to caching all 32 heads’ worth of K/V independently — smaller than Qwen3-Coder-30B-A3B’s 8× (4 KV heads for 32 query heads), but the same mechanism: fewer cached K/V groups, shared across more query heads each.

> [!question]- 3. Why doesn’t Qwen3-Coder-30B-A3B’s Mixture-of-Experts architecture reduce its KV cache size, given that only 8 of 128 experts fire per token?
> > [!success]- Answer
> > Because MoE routing operates on the FFN sublayer, not the attention sublayer. Every token still computes a full attention pass — full Q, K, V, O projections at every layer — regardless of which expert FFN processes it afterward. The KV cache stores the output of the attention sublayer, which is untouched by expert routing; the "8 of 128" sparsity only ever applies to which FFN weights get read, never to the K/V tensors being cached.

## What’s next

You now have a KV cache formula anchored to this exact model, a measured overhead constant that closes the gap between predicted and real VRAM, and a complete budget equation. [[module-5-expert-placement]] uses that equation to answer the question this module was building toward: given a 16 GB card and a target context length, how many expert layers can you afford to keep on GPU, and how much does moving the rest to DDR5 actually cost you per token?

<script src="/tutor.js" defer></script>
