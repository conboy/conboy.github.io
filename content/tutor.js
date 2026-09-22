/*
 * NPU course tutor — an in-browser Socratic study assistant.
 *
 * Design constraint #1: DO NOT AFFECT PAGE LOAD.
 * This file is loaded with `defer`, is a few KB, and on load does nothing
 * except draw a button and attach listeners. No network requests, no model,
 * no content index, no WebLLM library until the reader explicitly clicks.
 * The ~900 MB model download happens only on an explicit second confirmation.
 *
 * Everything runs on the visitor's own GPU via WebGPU. No API key, no backend,
 * no per-visitor cost, no rate limit, and nothing the reader types ever leaves
 * their machine.
 */
(() => {
  "use strict";

  const MODEL = "Llama-3.2-1B-Instruct-q4f16_1-MLC";
  const WEBLLM_ESM = "https://esm.run/@mlc-ai/web-llm";
  const INDEX_URL = "/static/contentIndex.json";
  const COURSE_PREFIX = "npu-course/";

  // ---------------------------------------------------------------- state
  let engine = null;       // WebLLM engine, created on demand
  let chunks = null;       // RAG corpus, fetched on demand
  let booting = false;
  let history = [];

  const SYSTEM = `You are a patient teaching assistant for a hardware engineering course about building an int8 NPU (neural processing unit) on a Real Digital Blackboard FPGA (Xilinx Zynq XC7Z007S).

RULES, in priority order:

1. ANSWER ONLY FROM THE COURSE EXCERPTS PROVIDED. The excerpts below are the
   course text. If the answer is not in them, say plainly: "The course doesn't
   cover that directly" and then say what it DOES cover that is closest. Never
   invent register names, part numbers, Vivado menu paths, API functions, or
   resource counts. A confident wrong number is much worse than "I don't know" —
   this reader is working on real hardware and will act on what you say.

2. BE SOCRATIC. This course is built on doing, not reading. Prefer a guiding
   question over a handed-over answer. If the reader asks "why is my timing
   failing?", ask what their WNS is and which path is worst before explaining.
   If they ask a direct factual question ("how many DSP slices?"), just answer
   it — Socratic does not mean evasive.

3. BE BRIEF. Two or three short paragraphs at most. Point them at the specific
   phase to read rather than reproducing it.

4. Use the reader's real numbers: 14,400 LUTs, ~60 DSP48E1 slices, 225 KB BRAM,
   single-core ARM Cortex-A9 at 666 MHz, 512 MB 16-bit DDR3 (~1.2 GB/s).`;

  // ------------------------------------------------------------ retrieval
  // Tiny lexical retriever. No embedding model — that would mean a second
  // download. Scores on rare-term overlap, which is plenty for 14 pages.
  const STOP = new Set(("the a an and or but if then than that this these those is are was were be been being of to in on at by for with from as it its i you your we they he she what why how when where which who do does did not no yes can could should would will just about into over under more most some any each"
  ).split(" "));

  const tokenize = (s) => (s.toLowerCase().match(/[a-z0-9_.+-]{2,}/g) || []).filter((w) => !STOP.has(w));

  async function loadChunks() {
    if (chunks) return chunks;
    const res = await fetch(INDEX_URL);
    if (!res.ok) throw new Error("Could not load the course index (" + res.status + ")");
    const idx = await res.json();
    const out = [];
    for (const [slug, page] of Object.entries(idx)) {
      if (!slug.startsWith(COURSE_PREFIX)) continue;
      const text = (page.content || "").replace(/\s+/g, " ").trim();
      if (!text) continue;
      // ~1100-char windows with overlap, so a definition is never cut in half.
      for (let i = 0; i < text.length; i += 900) {
        const body = text.slice(i, i + 1100);
        if (body.length < 120) continue;
        out.push({ slug, title: page.title || slug, body, terms: tokenize(body) });
      }
    }
    // Inverse document frequency, so "systolic" outweighs "the array".
    const df = new Map();
    for (const c of out) for (const t of new Set(c.terms)) df.set(t, (df.get(t) || 0) + 1);
    for (const c of out) {
      const tf = new Map();
      for (const t of c.terms) tf.set(t, (tf.get(t) || 0) + 1);
      c.vec = tf;
    }
    chunks = { list: out, df, N: out.length };
    return chunks;
  }

  function retrieve(query, k = 4) {
    const qs = tokenize(query);
    if (!qs.length) return [];
    const { list, df, N } = chunks;
    // Prefer the page the reader is actually on — a question asked while
    // reading Phase 3 almost always means Phase 3.
    const hereSlug = location.pathname.replace(/^\/|\/$/g, "");
    const isDefinitional = /^\s*(what('?s| is| are)|define|meaning of|what do(es)? .* mean)\b/i.test(query);
    const scored = list.map((c) => {
      let s = 0;
      for (const t of new Set(qs)) {
        const tf = c.vec.get(t);
        if (!tf) continue;
        s += (1 + Math.log(tf)) * Math.log(1 + N / (1 + (df.get(t) || 0)));
      }
      // Title hits are signal when the page name shares a query term
      // ("vector unit", "board specs", "command sequencer").
      const title = tokenize(c.title);
      for (const t of new Set(qs)) if (title.includes(t)) s *= 1.35;
      if (c.slug === hereSlug) s *= 1.25;
      // "What is X?" is a definition question — the glossary answers those best.
      // Without this, "what is a LUT" surfaces the vector unit's lookup tables.
      if (isDefinitional && c.slug === COURSE_PREFIX + "concepts") s *= 1.6;
      return { c, s };
    });
    scored.sort((a, b) => b.s - a.s);
    return scored.filter((x) => x.s > 0).slice(0, k).map((x) => x.c);
  }

  // -------------------------------------------------------------- styling
  const CSS = `
  #npu-tutor-fab{position:fixed;right:1.25rem;bottom:1.25rem;z-index:9998;border:0;border-radius:999px;
    padding:.7rem 1.1rem;font:600 .9rem/1 var(--bodyFont,system-ui),system-ui;cursor:pointer;
    background:var(--secondary,#284b63);color:#fff;box-shadow:0 6px 20px rgba(0,0,0,.25)}
  #npu-tutor-fab:hover{filter:brightness(1.12)}
  #npu-tutor{position:fixed;right:1.25rem;bottom:1.25rem;z-index:9999;width:min(30rem,calc(100vw - 2.5rem));
    max-height:min(40rem,calc(100vh - 2.5rem));display:none;flex-direction:column;border-radius:.8rem;
    overflow:hidden;background:var(--light,#faf8f8);color:var(--dark,#2b2b2b);
    border:1px solid var(--lightgray,#e5e5e5);box-shadow:0 12px 40px rgba(0,0,0,.3);
    font:400 .9rem/1.5 var(--bodyFont,system-ui),system-ui}
  #npu-tutor.open{display:flex}
  #npu-tutor header{display:flex;align-items:center;gap:.5rem;padding:.6rem .8rem;
    background:var(--secondary,#284b63);color:#fff;font-weight:600;font-size:.85rem}
  #npu-tutor header .sp{flex:1}
  #npu-tutor header button{background:transparent;border:0;color:#fff;cursor:pointer;font-size:1.1rem;line-height:1;padding:.15rem .35rem}
  #npu-tutor .body{flex:1;overflow-y:auto;padding:.8rem;display:flex;flex-direction:column;gap:.7rem}
  #npu-tutor .msg{padding:.55rem .75rem;border-radius:.6rem;white-space:pre-wrap;overflow-wrap:anywhere}
  #npu-tutor .msg.u{background:var(--secondary,#284b63);color:#fff;align-self:flex-end;max-width:85%}
  #npu-tutor .msg.a{background:var(--lightgray,#e5e5e5);color:var(--dark,#2b2b2b);align-self:flex-start;max-width:95%}
  #npu-tutor .msg.sys{background:transparent;border:1px dashed var(--gray,#b8b8b8);font-size:.8rem;color:var(--darkgray,#4e4e4e)}
  #npu-tutor .cites{font-size:.75rem;margin-top:.4rem;opacity:.85}
  #npu-tutor .cites a{color:var(--secondary,#284b63)}
  #npu-tutor form{display:flex;gap:.4rem;padding:.6rem;border-top:1px solid var(--lightgray,#e5e5e5)}
  #npu-tutor input{flex:1;padding:.5rem .6rem;border-radius:.45rem;border:1px solid var(--lightgray,#e5e5e5);
    background:var(--light,#fff);color:inherit;font:inherit}
  #npu-tutor button.send{border:0;border-radius:.45rem;padding:.5rem .8rem;cursor:pointer;
    background:var(--secondary,#284b63);color:#fff;font-weight:600}
  #npu-tutor button.send:disabled{opacity:.5;cursor:default}
  #npu-tutor .chips{display:flex;flex-wrap:wrap;gap:.35rem;padding:0 .6rem .6rem}
  #npu-tutor .chips button{font:inherit;font-size:.78rem;padding:.3rem .55rem;border-radius:999px;cursor:pointer;
    background:transparent;color:var(--secondary,#284b63);border:1px solid var(--lightgray,#e5e5e5)}
  #npu-tutor .start{padding:1rem;font-size:.85rem}
  #npu-tutor .start b{display:block;margin-bottom:.4rem;font-size:.95rem}
  #npu-tutor .start ul{margin:.5rem 0 .8rem;padding-left:1.1rem}
  #npu-tutor .start li{margin:.2rem 0}
  #npu-tutor progress{width:100%;height:.4rem}
  #npu-sel-btn{position:absolute;z-index:9997;border:0;border-radius:.4rem;padding:.35rem .6rem;cursor:pointer;
    font:600 .78rem/1 system-ui;background:var(--secondary,#284b63);color:#fff;box-shadow:0 4px 14px rgba(0,0,0,.28)}
  @media print{#npu-tutor,#npu-tutor-fab,#npu-sel-btn{display:none!important}}`;

  // ----------------------------------------------------------------- view
  let el = {};

  function mount() {
    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);

    const fab = document.createElement("button");
    fab.id = "npu-tutor-fab";
    fab.type = "button";
    fab.textContent = "Ask the tutor";
    fab.addEventListener("click", open);
    document.body.appendChild(fab);

    const panel = document.createElement("div");
    panel.id = "npu-tutor";
    panel.innerHTML = `
      <header><span>NPU course tutor</span><span class="sp"></span>
        <button type="button" data-act="close" aria-label="Close">&times;</button></header>
      <div class="body"></div>
      <div class="chips"></div>
      <form autocomplete="off"><input type="text" placeholder="Ask about the course…" aria-label="Ask a question"/>
        <button class="send" type="submit">Ask</button></form>`;
    document.body.appendChild(panel);

    el = {
      fab, panel,
      body: panel.querySelector(".body"),
      chips: panel.querySelector(".chips"),
      form: panel.querySelector("form"),
      input: panel.querySelector("input"),
      send: panel.querySelector("button.send"),
    };

    panel.querySelector('[data-act="close"]').addEventListener("click", close);
    el.form.addEventListener("submit", (e) => { e.preventDefault(); submit(el.input.value); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });

    showIntro();
    initSelection();
  }

  function open() { el.panel.classList.add("open"); el.fab.style.display = "none"; el.input.focus(); }
  function close() { el.panel.classList.remove("open"); el.fab.style.display = ""; }

  function add(cls, text) {
    const d = document.createElement("div");
    d.className = "msg " + cls;
    d.textContent = text;
    el.body.appendChild(d);
    el.body.scrollTop = el.body.scrollHeight;
    return d;
  }

  function showIntro() {
    const d = document.createElement("div");
    d.className = "start";
    const webgpu = "gpu" in navigator;
    d.innerHTML = webgpu
      ? `<b>A study assistant that runs on your own GPU</b>
         It answers <em>only</em> from this course's 14 pages, and prefers to ask you
         a guiding question rather than hand over the answer.
         <ul>
           <li>Runs entirely in your browser — nothing you type is sent anywhere</li>
           <li>No account, no API key, no usage limit</li>
           <li>One-time <b>~900&nbsp;MB</b> model download, then cached offline</li>
         </ul>
         <button class="send" type="button" data-act="boot">Download model &amp; start</button>`
      : `<b>Your browser can't run the tutor</b>
         It needs <b>WebGPU</b>, which this browser doesn't expose. Chrome or Edge 113+,
         Firefox 141+, or Safari on macOS&nbsp;Tahoe&nbsp;26 / iOS&nbsp;26 will work.
         <p style="margin:.6rem 0 0">Everything the tutor knows is just the course text —
         <a href="/npu-course/concepts">the glossary</a> and each phase's
         <em>Check your understanding</em> section cover the same ground.</p>`;
    el.body.appendChild(d);
    const b = d.querySelector('[data-act="boot"]');
    if (b) b.addEventListener("click", () => { d.remove(); boot(); });
  }

  // ------------------------------------------------------------- lifecycle
  async function boot() {
    if (engine || booting) return;
    booting = true;
    const status = add("sys", "Loading the course text…");
    const bar = document.createElement("progress");
    bar.max = 100; bar.value = 0;
    status.appendChild(bar);

    try {
      await loadChunks();
      status.firstChild.nodeValue = "Downloading the model (first visit only)… ";
      const webllm = await import(/* webpackIgnore: true */ WEBLLM_ESM);
      engine = await webllm.CreateMLCEngine(MODEL, {
        initProgressCallback: (p) => {
          status.firstChild.nodeValue = p.text || "Loading…";
          if (typeof p.progress === "number") bar.value = Math.round(p.progress * 100);
        },
      });
      status.remove();
      add("sys", "Ready. Ask anything about the course — or select text on the page and click “Explain this”.");
      showChips();
    } catch (err) {
      status.remove();
      add("sys", "The tutor failed to load: " + (err && err.message ? err.message : err) +
        "\nThe course text itself is unaffected — try the glossary at /npu-course/concepts.");
    } finally {
      booting = false;
    }
  }

  function showChips() {
    const here = document.title.replace(/\s*[|·].*$/, "").trim();
    const ideas = [
      "Quiz me on this page",
      "Why 8×4 and not 8×8?",
      "What is a systolic array?",
      "Explain the KV cache budget",
    ];
    if (location.pathname.includes("/npu-course/") && here) ideas[0] = `Quiz me on “${here}”`;
    el.chips.innerHTML = "";
    for (const t of ideas) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = t;
      b.addEventListener("click", () => submit(t));
      el.chips.appendChild(b);
    }
  }

  // ----------------------------------------------------------------- chat
  async function submit(text, prefix) {
    text = (text || "").trim();
    if (!text) return;
    if (!engine) { if (!booting) boot(); return; }
    el.input.value = "";
    el.send.disabled = true;
    el.chips.innerHTML = "";
    add("u", text);

    const hits = retrieve(prefix ? prefix + " " + text : text, 4);
    const context = hits.map((h, i) => `[${i + 1}] From "${h.title}":\n${h.body}`).join("\n\n");
    const pageHint = location.pathname.includes("/npu-course/")
      ? `\n\nThe reader is currently on the page: "${document.title}".` : "";

    const out = add("a", "");
    try {
      const messages = [
        { role: "system", content: SYSTEM + pageHint },
        ...history.slice(-4),
        { role: "user", content:
          `COURSE EXCERPTS:\n${context || "(no relevant excerpt found)"}\n\n---\nReader asks: ${text}` },
      ];
      const stream = await engine.chat.completions.create({
        messages, stream: true, temperature: 0.4, max_tokens: 420,
      });
      let acc = "";
      for await (const part of stream) {
        acc += part.choices?.[0]?.delta?.content || "";
        out.textContent = acc;
        el.body.scrollTop = el.body.scrollHeight;
      }
      history.push({ role: "user", content: text }, { role: "assistant", content: acc });

      const seen = [...new Set(hits.map((h) => h.slug))];
      if (seen.length) {
        const c = document.createElement("div");
        c.className = "cites";
        c.innerHTML = "Course pages: " + seen
          .map((s) => `<a href="/${s}">${s.replace(COURSE_PREFIX, "")}</a>`).join(" · ");
        out.appendChild(c);
      }
    } catch (err) {
      out.textContent = "Something went wrong: " + (err && err.message ? err.message : err);
    } finally {
      el.send.disabled = false;
      showChips();
      el.input.focus();
    }
  }

  // ------------------------------------------------- explain-this-selection
  function initSelection() {
    let btn = null;
    const kill = () => { if (btn) { btn.remove(); btn = null; } };

    document.addEventListener("mouseup", () => {
      setTimeout(() => {
        const sel = window.getSelection();
        const text = sel ? sel.toString().trim() : "";
        kill();
        if (!text || text.length < 12 || text.length > 1200) return;
        // Ignore selections inside the tutor itself.
        if (el.panel && sel.anchorNode && el.panel.contains(sel.anchorNode)) return;

        const rect = sel.getRangeAt(0).getBoundingClientRect();
        btn = document.createElement("button");
        btn.id = "npu-sel-btn";
        btn.type = "button";
        btn.textContent = "Explain this";
        btn.style.top = (window.scrollY + rect.bottom + 8) + "px";
        btn.style.left = (window.scrollX + rect.left) + "px";
        btn.addEventListener("mousedown", (e) => {
          e.preventDefault();
          const q = `Explain this excerpt from the course, in context:\n\n"""${text}"""`;
          kill();
          open();
          if (!engine) {
            add("sys", "Start the tutor above, then select the text again.");
            if (!booting) boot();
            return;
          }
          submit(q, text);
        });
        document.body.appendChild(btn);
      }, 10);
    });

    document.addEventListener("mousedown", (e) => { if (btn && e.target !== btn) kill(); });
    document.addEventListener("scroll", kill, { passive: true });
  }

  // Only mount on course pages, and only once the browser is otherwise idle.
  function start() {
    if (!location.pathname.includes("/npu-course")) return;
    if (document.getElementById("npu-tutor-fab")) return;
    mount();
  }
  const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 200));
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => idle(start));
  } else {
    idle(start);
  }
  // Quartz is an SPA; re-mount after client-side navigation.
  document.addEventListener("nav", () => idle(start));
})();
