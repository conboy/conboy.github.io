---
title: "Phase 1 — A Baseline LLM on the ARM Core"
description: "Porting llama2.c to bare-metal ARM, profiling it, quantizing to int8, and freezing the golden model."
tags:
  - npu
  - hardware
---

# Phase 1 — Build the thing you're going to accelerate

**3 weeks** · Part of [[index|Zero to NPU]] · Prev: [[phase-0-toolchain]] · Next: [[phase-2-numerics]]

> [!info] What this phase is for
> By the end of this phase you will have a real language model — not a toy, not a demo, an actual transformer — generating short stories on your Blackboard, running entirely on the ARM Cortex-A9, with zero custom hardware involved. That's the milestone. Everything from Phase 2 onward is about making this exact program faster by moving pieces of it into the FPGA fabric. This phase also produces two things you'll lean on for the rest of the course: a **profile** telling you exactly where the time goes, and a **golden reference** — known-correct inputs and outputs you'll check every future RTL module against. Nothing here is throwaway work.

---

## Before you start: what a transformer actually does

You don't need a machine learning background for this course, but you do need one clear mental model, because everything you build for the next eight months is accelerating *this specific loop*. So let's walk through it once, plainly, with no math you don't need.

**The whole job of the model is: given the words so far, guess the next word.** That's it. Feed it "Once upon a", it guesses "time". Feed it "Once upon a time", it guesses something else. Run that loop over and over and you get a story. Everything below is just the machinery for making that one guess as good as possible.

The model doesn't work with English words directly — it works with **[[concepts#Token|tokens]]**, small chunks of text (could be a word, part of a word, or a punctuation mark). This model has a vocabulary of exactly 512 possible tokens — `vocab_size=512` — so every token is just an integer from 0 to 511. The first thing that happens to a token is it gets turned into a list of 64 numbers, called an **[[concepts#Embedding|embedding]]**. This is nothing clever: it's a lookup table. Token #37 always maps to the same 64 numbers. Those 64 numbers are the model's internal "meaning" of that token, and `dim=64` is why they're 64 long — that number is a config choice baked into this checkpoint.

That 64-number vector then flows through 5 identical **layers** (`n_layers=5`), stacked back-to-back like pipeline stages. Each layer does the same seven steps, in this order:

1. **[[concepts#RMSNorm|RMSNorm]]** — rescale the 64 numbers so their overall size stays in a sane range. Purely a normalization step, no learned "meaning" here beyond a scale factor.
2. **Attention: compute Q, K, V** — three separate matrix multiplies produce a "Query", "Key", and "Value" vector for the current token. Think of Query as "what am I looking for", Key as "what do I offer", Value as "what do I actually contribute".
3. **[[concepts#RoPE|RoPE]] (rotary position embedding)** — rotates the Q and K vectors by an amount that depends on *where* this token sits in the sequence, so the model can tell "the cat sat" apart from "sat the cat".
4. **Attention scores** — compare this token's Query against every previous token's Key (a dot product each) to get a raw "how relevant is that past token to me right now" score per past position.
5. **[[concepts#Softmax|Softmax]]** — squash those raw scores into probabilities that sum to 1, so they behave like weights.
6. **Weighted sum** — blend all the previous tokens' Value vectors together using those softmax weights. This is "attention" — the current token pulling in relevant context from earlier tokens.
7. **FFN ([[concepts#SwiGLU|SwiGLU]]) + residual add** — a small two-layer matrix network (`hidden_dim=172`) that further transforms the vector, then it gets added back onto what came into the layer (a "residual" connection, which is just addition — it keeps information from earlier layers from getting lost).

Step 4 is where the **[[concepts#KV cache|KV cache]]** matters. Naively, generating token #50 means recomputing Q/K/V for tokens #1 through #49 all over again — wasteful and it gets worse every token. Instead, you cache every previous token's K and V vectors the first time you compute them, so generating a new token only ever costs *one* new Q/K/V computation plus reusing the cache. This is a pure engineering optimization; it changes nothing about the math, only how much of it you redo.

After all 5 layers, you have one final 64-number vector for the current token. One last matrix multiply projects that into 512 numbers — one score per possible next token — called **[[concepts#Logits|logits]]**. Bigger logit means the model thinks that token is more likely to come next. You **sample** one token from those 512 scores (sometimes just picking the biggest, sometimes picking randomly weighted by the scores), append it to your sequence, and go back to the top of the loop:

```
 tokens so far ──▶ [ token ] ──▶ [ embedding: 64 nums ]
                                        │
                                        ▼
                     ┌─────────────────────────────────────┐
                     │  layer 1 .. layer 5 (each layer:     │
                     │  RMSNorm → Q,K,V → RoPE → scores →   │
                     │  softmax → weighted sum → FFN+add )  │
                     └─────────────────────────────────────┘
                                        │
                                        ▼
                          [ logits: 512 scores ] ──▶ sample
                                        │
                                        ▼
                              append new token ──▶ loop
```

Now the important part, the reason this whole course exists: **almost every one of those steps above is a matrix multiplied by a vector.** The Q/K/V projections, the attention weighted sum, the FFN, the final logits projection — all matmuls. RMSNorm, RoPE, and softmax are cheap by comparison; you'll measure exactly how cheap in exercise 1.2. A matrix-multiply engine, built in the FPGA fabric, is therefore the single highest-leverage piece of hardware you could design. That's what the rest of this course builds toward — this phase just makes sure you understand, in software, exactly what that hardware will need to do.

---

## Why this model, specifically

`stories260K` is Karpathy's smallest `llama2.c` checkpoint: `dim=64`, `hidden_dim=172`, `n_layers=5`, `n_heads=8`, `n_kv_heads=4`, `head_size=8`, `kv_dim=32`, `vocab_size=512`. That's roughly 260K parameters total — about 227K spread across the 5 transformer layers, plus 33K in the token embedding table. At 1 byte per parameter after quantization (Phase 1.3), that's under 260 KB, which fits comfortably inside a **225 KB on-chip BRAM budget** on the XC7Z007S with room for activations. It's also trained on TinyStories, a dataset of very simple children's stories, so even this tiny model produces genuinely coherent (if simple) English — you'll actually be able to tell if your port is working by reading its output. Later in the course you'll hear about `stories15M` (15 million parameters) as the "what if we had a bigger FPGA" thought experiment — it won't fit in BRAM here, and that contrast is the point: this course is about what you *can* accelerate on real, constrained silicon.

---

## 1.1 Get a real LLM running on the ARM core

> [!abstract] Goal
> Port `llama2.c` to bare-metal on the Zynq's A9 core and get it printing a story over UART.

**What's going on:** `llama2.c` was written for a Linux/Mac laptop, so it leans on OS features you don't have here: it `mmap()`s the checkpoint, uses `malloc()` freely, and calls `printf("%f", ...)`. Bare-metal Xilinx gives you none of that. 1.1 is pure plumbing, not math — same logic, same weights, minus the OS.

**Steps:**
1. Pull `llama2.c` and `stories260K.bin` (+ tokenizer file) from Karpathy's repo, and copy them onto the microSD card.
2. In Vitis, add the FatFs library so you can `f_open`/`f_read` the checkpoint off the SD card into a static buffer instead of `mmap()`-ing it.
3. Replace every `malloc()` in the loading path with a fixed-size static array — every dimension (`dim=64`, `hidden_dim=172`, ...) is already known.
4. Strip the `%f`/`%e` floating-point `printf` calls (bare-metal newlib-nano usually can't format floats); swap in an integer-scaled print helper, or skip printing floats for now.
5. Retarget one `putchar()`-style function so output goes out the UART instead of stdout.
6. Build, flash, and generate.

```c
/* ---- 1. Strip mmap: read the checkpoint into a static buffer ---- */
/* llama2.c originally does:
 *   fd = open(checkpoint, O_RDONLY);
 *   data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
 * There is no mmap() on bare metal. Replace with a plain read into
 * memory you already own. We know the file size ahead of time because
 * we know the checkpoint's dimensions.
 */
#define CHECKPOINT_MAX_BYTES (300 * 1024)      /* stories260K.bin is small */
static uint8_t checkpoint_buf[CHECKPOINT_MAX_BYTES];  /* static, not malloc'd */

int load_checkpoint_from_sd(const char *path) {
    FIL file;
    UINT bytes_read;
    if (f_open(&file, path, FA_READ) != FR_OK) return -1;
    if (f_read(&file, checkpoint_buf, CHECKPOINT_MAX_BYTES, &bytes_read) != FR_OK) {
        f_close(&file);
        return -1;
    }
    f_close(&file);
    /* checkpoint_buf now holds the raw weight file, same bytes mmap() would
     * have exposed. Point your TransformerWeights struct fields at offsets
     * into checkpoint_buf exactly like the original pointer-cast code does. */
    return 0;
}

/* ---- 2. Retarget output: give llama2.c a UART "printf" ---- */
/* Xilinx's standalone BSP already provides outbyte(char c) over UART if
 * you enabled the UART peripheral in your board support package. Newlib's
 * _write() syscall is the one hook that everything -- printf, putchar,
 * puts -- eventually calls down into. Retarget that single function and
 * every existing call site in llama2.c "just works" unmodified. */
int _write(int fd, const char *buf, int len) {
    (void)fd;
    for (int i = 0; i < len; i++) {
        outbyte(buf[i]);           /* BSP-provided: one byte out the UART */
    }
    return len;
}
```

> [!success] Done when
> The UART terminal (115200 baud, same setup as Phase 0) prints a short, readable story generated by `generate()` — not garbage characters, not a hang.

> [!question]- It's not working
> 1. Nothing prints — check `_write()` is actually linked in (not stripped as unused), and that UART baud matches your terminal.
> 2. Garbled characters — baud mismatch, or you're writing raw floats/structs instead of formatted text.
> 3. Crash or hang on load — check `CHECKPOINT_MAX_BYTES` is big enough, and that `f_read`'s return code and `bytes_read` match the real file size.
> 4. Crash inside `forward()` — you likely left a `malloc()` in the `RunState` path; every buffer size there is known ahead of time, so make it static too.
> 5. SD card not found — reformat as FAT32; the Zynq SD driver is picky about partition tables.

> [!bug] Gotcha
> Do this port with **zero changes to the model math**. Every stripped-out feature (mmap, malloc, float printf) is an OS convenience, not part of the transformer. If you find yourself editing `forward()` or `matmul()` in this step, stop — you've wandered outside the scope of 1.1.

---

## 1.2 Profile it

> [!abstract] Goal
> Measure, in cycles, exactly where `generate()` spends its time.

**What's going on:** Right now you have a working but unmeasured program. Before spending eight months building hardware to accelerate "the slow part," you need hard numbers proving what the slow part actually is. The A9 has a **global timer**, and Xilinx's standalone BSP exposes it through one clean API: `XTime_GetTime()` in `xtime_l.h`. Wrap each major operation with a start/stop timestamp pair and add up the cycles.

**Steps:**
1. `#include "xtime_l.h"` and confirm `XTime_GetTime()` is available in your BSP.
2. Write a `PROFILE_BEGIN`/`PROFILE_END` macro pair that reads the timer before/after a block and accumulates elapsed cycles into a named counter.
3. Wrap the matmul calls inside `forward()`, plus `rmsnorm()`, `softmax()`, RoPE, and `sample()`, each with its own counter.
4. Run a full generation and print the accumulated counters as percentages of total time.

```c
#include "xtime_l.h"

/* One running total per operation you care about. */
static XTime cycles_matmul  = 0;
static XTime cycles_rmsnorm = 0;
static XTime cycles_softmax = 0;
static XTime cycles_rope    = 0;
static XTime cycles_sample  = 0;

/* PROFILE_BEGIN/END wrap a block and add its elapsed cycles into `counter`.
 * XTime_GetTime() reads the A9 global timer's free-running counter -- it
 * never stops or wraps in any run this program will do, so a plain
 * subtraction gives elapsed cycles directly. */
#define PROFILE_BEGIN(tvar)  XTime tvar##_start; XTime_GetTime(&tvar##_start)
#define PROFILE_END(tvar, counter) \
    do { \
        XTime tvar##_end; \
        XTime_GetTime(&tvar##_end); \
        counter += (tvar##_end - tvar##_start); \
    } while (0)

/* Example: wrapping one matmul call site inside forward() */
void profiled_matmul(float *out, float *x, float *w, int n, int d) {
    PROFILE_BEGIN(t);
    matmul(out, x, w, n, d);      /* the real llama2.c function, untouched */
    PROFILE_END(t, cycles_matmul);
}

/* After generation finishes, print a breakdown. COUNTS_PER_SECOND is a
 * BSP-provided constant for converting the global timer's raw counts into
 * real time, if you want seconds instead of raw cycles. */
void print_profile(XTime total_cycles) {
    xil_printf("matmul : %d%%\r\n", (int)(100 * cycles_matmul  / total_cycles));
    xil_printf("rmsnorm: %d%%\r\n", (int)(100 * cycles_rmsnorm / total_cycles));
    xil_printf("softmax: %d%%\r\n", (int)(100 * cycles_softmax / total_cycles));
    xil_printf("rope   : %d%%\r\n", (int)(100 * cycles_rope    / total_cycles));
    xil_printf("sample : %d%%\r\n", (int)(100 * cycles_sample  / total_cycles));
}
```

Your results table will look roughly like this shape (your exact numbers will differ):

| Operation | % of total time |
|---|---|
| matmul  | 85–95% |
| rmsnorm | low single digits |
| softmax | low single digits |
| RoPE    | low single digits |
| sample  | negligible |

> [!success] Done when
> You have a printed table of percentages that sums to ~100%, gathered from a real generation run, not a guess.

> [!question]- It's not working
> 1. Percentages don't sum near 100% — you're missing a counter somewhere; check you wrapped *every* matmul call site, not just one.
> 2. All counters read zero — the global timer peripheral may not be enabled in your Vivado hardware platform.
> 3. Numbers vary wildly run to run — on this single-core bare-metal setup, that's usually your own code (accidentally timing a UART print or SD read), not the hardware.
> 4. matmul isn't dominating as expected — double check you aren't timing something else inside the same counter by mistake.

> [!warning] Expect matmul at 85–95%
> Everything else is noise — *until you accelerate the matmul*, at which point it becomes your bottleneck. Remember this number. It's Amdahl's Law and it will ambush you in [[phase-5-vector-unit]].

---

## 1.3 Quantize to int8, in C

> [!abstract] Goal
> Convert the model's floating-point weights and activations to int8, entirely in software, before any hardware exists to consume them.

**What's going on:** Floating-point multiply-accumulate hardware is expensive to build in an FPGA; integer multiply-accumulate is cheap. Before designing any matrix engine, you need to prove the *model itself* still works with 8-bit integers instead of 32-bit floats. "Per-channel symmetric for weights" means each row of a weight matrix gets its own scale factor, symmetric around zero (no zero-point offset, so no correction term in the matmul). "Per-tensor for activations" means the whole activation vector shares one scale, recomputed each call since activations change every token. Accumulation happens in int32 so nothing overflows.

**Steps:**
1. For each row of a weight matrix, find its max absolute value and compute scale = max_abs / 127.
2. Quantize every value in that row: round(value / scale), clamped to [-127, 127].
3. For activations, do the same with one scale for the whole vector, recomputed per call.
4. Accumulate products in `int32_t`, then requantize back to int8 before the next operation.
5. Run `generate()` on both fp32 and int8 versions and compare output quality (perplexity, or just read the stories side by side).

```c
#include <stdint.h>
#include <math.h>

/* Compute a per-channel (per-row) symmetric scale for a weight matrix.
 * `w` is `rows x cols`, row-major. One scale per row -- this is what
 * "per-channel" means here: each output channel gets its own scale
 * instead of the whole matrix sharing one. */
void compute_per_channel_scales(const float *w, int rows, int cols, float *scales_out) {
    for (int r = 0; r < rows; r++) {
        float max_abs = 0.0f;
        for (int c = 0; c < cols; c++) {
            float v = fabsf(w[r * cols + c]);
            if (v > max_abs) max_abs = v;
        }
        /* Symmetric: map [-max_abs, +max_abs] onto [-127, 127]. Zero-point
         * is implicitly 0 -- that's what "symmetric" buys you: no offset
         * term to add back in during the matmul. */
        scales_out[r] = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    }
}

/* Quantize one row of floats to int8 using its scale. Round-to-nearest,
 * then clamp -- values right at the edge of the range can round outside
 * [-127, 127] and must be clipped, not wrapped. */
void quantize_row_int8(const float *row_f, int8_t *row_q, int cols, float scale) {
    for (int c = 0; c < cols; c++) {
        int32_t q = (int32_t)lroundf(row_f[c] / scale);   /* round to nearest */
        if (q > 127)  q = 127;                            /* clamp high */
        if (q < -127) q = -127;                            /* clamp low  */
        row_q[c] = (int8_t)q;
    }
}

/* int8 matmul, int32 accumulate. This is the operation your future
 * matrix-engine RTL will be reimplementing in hardware -- match its
 * behavior exactly here, because this function is what "correct" means
 * for the rest of the course. */
void matmul_int8(int32_t *out, const int8_t *x, const int8_t *w,
                  const float *w_scales, float x_scale, int n, int d) {
    for (int i = 0; i < d; i++) {
        int32_t acc = 0;
        for (int j = 0; j < n; j++) {
            acc += (int32_t)x[j] * (int32_t)w[i * n + j];   /* int8 * int8 -> int32, no overflow */
        }
        /* Dequantize back to a real magnitude by folding in both scales.
         * Downstream code either keeps this as int32 or requantizes it
         * back to int8 for the next layer -- your call, document which. */
        out[i] = (int32_t)(acc);   /* store raw int32; scale = w_scales[i] * x_scale */
    }
}
```

> [!success] Done when
> The int8 model produces stories you'd judge as similarly coherent to the fp32 version, and you've logged a measured perplexity delta of **under 1%**.

> [!question]- It's not working
> 1. Output is total gibberish — check you're rounding to nearest, not always rounding toward zero or always up.
> 2. Output degrades a lot but isn't gibberish — check your per-channel scale isn't accidentally per-tensor (one scale shared across all rows); this is the most common bug here.
> 3. Values silently overflow — confirm your accumulator is genuinely `int32_t`, not truncated back to `int8_t`/`int16_t` somewhere downstream.
> 4. Perplexity delta way over 1% — check your clamp bound is 127, not 128, which overflows `int8_t`'s range.
> 5. All outputs are zero — an all-zero row gives a scale of 0.0; guard against dividing by it during requantization.

> [!bug] Gotcha
> Use **symmetric** (zero-point = 0) quantization. Asymmetric adds a cross-term to every matmul and roughly doubles your hardware complexity for accuracy you don't need here.

---

## 1.4 Freeze the golden model

> [!abstract] Goal
> Produce a set of known-correct input/output files for individual operations, so every RTL module you build later can be checked against real data instead of hand-derived test vectors.

**What's going on:** From Phase 2 on you'll write RTL — fixed-point arithmetic, then a full matrix engine. Each module needs a ground truth, and "the story looks okay" can't debug a broken multiplier. Refactor your int8 code so each operation dumps its exact inputs and outputs to the SD card; a later testbench checks your hardware's output bit-for-bit against that file.

**Steps:**
1. Pick one simple, self-describing binary format — a small header (op name, shape, dtype) followed by raw bytes.
2. Write a generic `dump_tensor()` helper any operation can call for its inputs or outputs.
3. Call it around one matmul during a real `generate()` run to produce your first golden vector file.
4. Confirm you can read the file back and recover the exact same bytes.

```c
/* A minimal, self-describing binary dump format for one operation's data.
 * Kept deliberately simple: a fixed header so a host-side (or Verilator)
 * reader always knows how many bytes follow, then raw data with no
 * padding or alignment surprises. */
typedef struct {
    char     op_name[16];   /* e.g. "matmul_l2_q"  -- which op, which call site */
    uint32_t rows;
    uint32_t cols;
    uint8_t  dtype;         /* 0 = int8, 1 = int32, 2 = float32 */
    uint8_t  reserved[3];   /* pad to a round header size */
} dump_header_t;

/* Write one tensor (input or output) to the SD card as
 * "<op_name>_<in|out>.bin". Called around whichever operation you're
 * freezing as a golden reference for that week's RTL module. */
void dump_tensor(const char *op_name, const char *suffix,
                  const void *data, uint32_t rows, uint32_t cols, uint8_t dtype) {
    dump_header_t hdr = {0};
    strncpy(hdr.op_name, op_name, sizeof(hdr.op_name) - 1);
    hdr.rows = rows;
    hdr.cols = cols;
    hdr.dtype = dtype;

    char filename[32];
    snprintf(filename, sizeof(filename), "%s_%s.bin", op_name, suffix);

    FIL file;
    UINT written;
    f_open(&file, filename, FA_WRITE | FA_CREATE_ALWAYS);
    f_write(&file, &hdr, sizeof(hdr), &written);

    uint32_t elem_size = (dtype == 0) ? 1 : 4;   /* int8 = 1 byte, int32/float32 = 4 */
    f_write(&file, data, rows * cols * elem_size, &written);
    f_close(&file);
}

/* Example usage around one matmul call during generation: */
void golden_matmul_l2(int32_t *out, const int8_t *x, const int8_t *w, int n, int d) {
    dump_tensor("matmul_l2", "in_x", x, 1, n, 0);      /* input activation, int8 */
    dump_tensor("matmul_l2", "in_w", w, d, n, 0);       /* weight matrix, int8 */
    matmul_int8(out, x, w, /* scales omitted for brevity */ NULL, 1.0f, n, d);
    dump_tensor("matmul_l2", "out", out, 1, d, 1);      /* output, int32 */
}
```

The format is intentionally boring: a fixed header then raw bytes, no compression, no text parsing. A Verilator or SystemVerilog testbench needs to read this back with minimal ceremony, and "boring and unambiguous" beats "clever" for a golden reference you'll depend on all course.

> [!success] Done when
> You can generate a `_in`/`_out` file pair for at least one matmul call, read it back byte-for-byte identical to what you wrote, and describe its format well enough that someone else could write a parser for it without asking you a question.

> [!question]- It's not working
> 1. File is empty or truncated — check `f_write`'s return code and byte count; SD writes can silently short-write if you don't check `written`.
> 2. Bytes don't round-trip identically — your struct may be picking up compiler padding; try `__attribute__((packed))` on the header.
> 3. Wrong shape on read-back — you likely swapped `rows`/`cols` somewhere; stay consistent about row-major everywhere.
> 4. Can't tell which call site produced which file — make `op_name` unique per call site, e.g. include the layer number.

> [!bug] Gotcha
> Every RTL module from here on is validated against these dumps.
>
> This is standard design-verification practice: you don't check hardware by looking at it, you check it against a reference implementation you trust. Hobby FPGA projects usually skip this step, then spend weeks debugging a system where *any* of a dozen blocks could be the culprit. Build the oracle first — now, not once the first module is already half-built.

---

## Check your understanding

> [!question]- Why does the KV cache exist, and what would break without it?
> > [!success]- Answer
> > Without it, generating token N recomputes Q, K, and V for all N-1 previous tokens from scratch every step — cost grows every token instead of staying roughly constant. The cache just stores each token's K and V the first time they're computed so later steps reuse them. It changes nothing about the math or output, only how much redundant work you do.

> [!question]- Why does this course care so much about matmul specifically?
> > [!success]- Answer
> > Because exercise 1.2 will show matmul is 85-95% of runtime — every Q/K/V projection, the attention-weighted sum, the FFN, and the logits projection are matrix multiplies. RMSNorm, softmax, and RoPE are comparatively cheap. A hardware matrix engine is the one piece of custom silicon that can actually move total speed; everything else optimizes the 5-15% left over.

> [!question]- Why symmetric int8 quantization instead of asymmetric, for this project?
> > [!success]- Answer
> > Symmetric quantization has no zero-point offset, so every matmul stays a plain multiply-accumulate. Asymmetric adds a correction term to every multiply, roughly doubling the arithmetic and hardware needed — for accuracy you don't need here, where symmetric already measures under 1% perplexity cost.

---

## What's next

You now have a working, profiled, quantized model and a set of golden test vectors on your SD card. [[phase-2-numerics]] takes the int8 quantization scheme you just built in C and turns it into real fixed-point arithmetic rules you'll implement in Verilog — the last stop before any RTL touches this model's actual numbers.

<script src="/tutor.js" defer></script>
