# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Source for **https://conboy.dev** — Conrad Fernandez's portfolio and technical notebook, built with **Quartz v5** (a Markdown static-site generator that renders Obsidian syntax natively) and deployed to GitHub Pages at `github.com/conboy/conboy.github.io`.

Almost all work here is **authoring Markdown in `content/`**, not changing Quartz itself. `quartz/` is upstream framework code; treat it as vendored unless a change genuinely requires it.

## Commands

```sh
npm ci                      # Node 22+ required (engines field); v24 is installed
npx quartz build --serve    # preview at localhost:8080, live reload
npx quartz build            # build into public/
npx tsc --noEmit            # typecheck
npm test                    # tsx --test, runs quartz/**/*.test.ts
npm run check               # tsc --noEmit && prettier --check
npm run format              # prettier --write
```

Run a single test file: `npx tsx --test quartz/util/path.test.ts`

CI (`.github/workflows/deploy.yml`, on push to `main`) runs, in order: **guard -> typecheck -> test -> build -> deploy**. Reproduce the full gate locally with `npx tsc --noEmit && npm test && npx quartz build`.

`npm run prebuild` (auto-run on build) executes `install-plugins`, which fetches the `@quartz-community/*` plugins listed in `quartz.config.yaml`. Plugin enablement and page layout both live in that file, not in TypeScript.

## Content architecture

`content/` is the entire published surface. Filename maps to URL: `content/notebook.md` -> `conboy.dev/notebook`; `content/index.md` is the homepage.

- `content/index.md` — portfolio landing page. **Raw HTML, not Markdown prose**, styled by `quartz/styles/custom.scss` with classes scoped to the homepage (`.portfolio-hero`, `.project-card`, `.evidence-strip`). Editing the homepage means editing HTML and SCSS together.
- `content/notebook.md` — hub that links out to notes and courses.
- `content/projects/` — case studies; each cites public sources for its claims.
- `content/npu-course/` — the "Zero to NPU" course. **This is the template for any new course.**
- `content/CNAME` — must survive every deploy or GitHub Pages drops the custom domain.

### Course structure convention (follow this for new courses)

The NPU course establishes the pattern, and it is deliberate:

- `index.md` is the map of content: framing, a "who this is for", a phase table, ground rules, capstones.
- Numbered phase notes (`phase-0-*` through `phase-8-*`) carry the hands-on work.
- `concepts.md` is a glossary defining every term, each entry ending with a _Used in:_ link back to the phase where it does real work.
- `board-specs.md` and `reference-material.md` hold deep reference material that phases link into.
- `architecture.canvas` is a JSON Canvas diagram.

Two rules that make the courses work:

1. **Modules stay exercise-driven; depth lives in linked reference notes.** Every module ends in a _measurement_ — a testbench that passed, or a number that came out of the hardware — not a feeling. Convert every "read X" into "run X and check Y".
2. **Never fabricate.** Where source material does not cover something, say so explicitly rather than filling the gap. The NPU course carries explicit "not documented in these guides" markers.

Each phase note opens with frontmatter (`title`, quoted `description`, `tags`), then a nav line:
`**N weeks** · Part of [[index|Course Name]] · Prev: [[...]] · Next: [[...]]`

## The IP guards — read before writing content

Conrad is a graphics engineer at AMD. Two independent guards scan `content/` for proprietary markers and **will block commits and deploys**:

1. `.githooks/pre-commit` — local; requires `git config core.hooksPath .githooks` per clone (already set in this clone, lost on a fresh one).
2. The `guard` job in `.github/workflows/deploy.yml` — a server-side mirror that cannot be bypassed.

Both scan `*.md`, `*.markdown`, and `*.canvas` under `content/` (canvas node text is published verbatim, so it is scanned too). Blocked markers include `AMD Confidential`, `Advanced Micro Devices`, `//depot/`, `atlp4s13`, `amd.com`, internal codenames, and `SWDEV-` IDs. Some patterns are case-sensitive on purpose so ordinary prose does not trip them.

> The guards are regex safety nets, not permission. **They cannot catch framing.** When porting work-adjacent writing, manually strip "at work", "the chip you work on", and links into private notes, then re-anchor every claim to a public source (open-source drivers, papers, public datasheets). A previous subagent rewrite reintroduced "the DV discipline you're surrounded by at work" — no regex catches that.

## The Obsidian vault link

`Documents/Obsidian/conboy.dev` is a **Windows directory junction pointing at `conboy-site/content/`**. Edits made in Obsidian show up in `git status` here immediately; there is no copy step. `WRITING.md` still describes the older "never connected, always hand-copy" design in places — the junction supersedes it.

The junction targets `content/`, not the repo root, deliberately: it keeps `node_modules` out of Obsidian's indexer and makes `quartz.config.yaml` uneditable from inside the vault.

**Consequence:** the same vault holds internal-only folders (`GFX13.7 Course`, `SV_TESTBENCH_EXPLAINED`) sitting beside the publish folder. Moving a note into `conboy.dev/` _is_ a publish action.

If the junction needs recreating, use PowerShell `New-Item -ItemType Junction` — `cmd mklink /J` gets mangled by MSYS path conversion. Remove it with `rmdir` on the junction itself, never `rm -rf`, which may recurse into the target.

## Authoring gotchas

- Quartz renders `[[wikilinks]]`, callouts, LaTeX/KaTeX, Mermaid, checkboxes, `.canvas` pages, backlinks, and the graph view.
- **An unquoted YAML `description:` containing a colon breaks the build.** Quote them.
- **The canvas plugin does not resolve `[[wikilinks]]`** inside node text; they emit as literal brackets. Use absolute Markdown links (`[label](/path)`) in canvases.
- `draft: true` excludes a note from the build (the `remove-draft` plugin).
- Standalone non-Quartz pages such as `/npu-anim/` need `data-router-ignore` on links so the SPA router performs a full document load.
- On AMD's network, Google Fonts fetches fail during build, giving a harmless system-font fallback locally. CI fetches them fine.

## Deployment traps

- GitHub Pages `build_type` must be `workflow`, not `legacy`, or Actions output is ignored.
- A force-push that replaces repo contents drops the custom domain. Re-bind with:
  `gh api -X PUT repos/conboy/conboy.github.io/pages -f cname="conboy.dev"`
- The 2022 Chirpy portfolio is preserved on the `chirpy-portfolio` branch of the same repo. Do not delete it.
