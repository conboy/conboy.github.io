# Publishing notes to conboy.dev

## The rule

**Your Obsidian vault is never connected to this repo.** Nothing is published unless
you deliberately copy it into `content/`. There is no sync, no symlink, no automation
pointing at your vault.

Two vaults exist on this machine and neither is referenced here:

- `Documents/Obsidian`
- `OneDrive - Advanced Micro Devices Inc/Documents/Obsidian Vault`

The OneDrive one is corporate. Do not copy from it.

## Writing a post

1. Write the note in Obsidian, in whatever vault you like.
2. Copy the finished `.md` into `conboy-site/content/`.
3. Preview locally: `npx quartz build --serve` → `localhost:8080`
4. Commit and push. The Actions workflow builds and deploys.

Filename becomes the URL: `content/npu-notes.md` → `conboy.dev/npu-notes`.
`content/index.md` is the homepage.

### Attachments

Images go in `content/` alongside the notes. Obsidian embeds (`![[diagram.png]]`)
resolve correctly as long as the image is copied over too.

## What works

Quartz understands Obsidian syntax natively:

- `[[wikilinks]]` and `[[wikilinks|with aliases]]`, plus automatic backlinks
- Callouts (`> [!note]`, `> [!tip]`, `> [!warning]`)
- YAML frontmatter — `title`, `tags`, `draft: true`
- LaTeX, Mermaid diagrams, syntax-highlighted code blocks
- The graph view, built from your actual link structure

Set `draft: true` in frontmatter to keep a note out of the build.

## The guardrails

Three independent layers, because one is not enough:

1. **`.gitignore`** — excludes `.obsidian/`, `private/`, `.trash/`, `*.private.md`,
   `*.confidential.md`.
2. **`.githooks/pre-commit`** — scans staged Markdown for proprietary markers (AMD
   Confidential, `//depot/` paths, internal project codenames, `SWDEV-` ticket IDs) and
   aborts the commit. Tested and confirmed working. Bypass with `--no-verify` only if
   you are certain it is a false positive.
3. **`.github/workflows/deploy.yml`** — the `guard` job re-runs that scan in CI and
   fails the deploy. This one cannot be bypassed by a forgotten hook or fresh clone.

The hook lives in `.githooks/` and is wired up via `git config core.hooksPath .githooks`.
That setting is **local to this clone** — if you clone this repo elsewhere, re-run:

```bash
git config core.hooksPath .githooks
```

`quartz.config.yaml` also has `ignorePatterns` covering `private`, `templates`,
`.obsidian`, `.trash`, and `**/AMD/**`.

> [!warning] These are safety nets, not permission
> A regex cannot recognize every piece of proprietary information. The guards catch
> obvious markers. Judgment about what is safe to publish is still yours.

## Local commands

```bash
npx quartz build --serve   # preview at localhost:8080, live reload
npx quartz build           # build into public/
```

Note: on AMD's network, Google Fonts fetches fail during build and Quartz falls back to
a system font. Harmless locally — CI fetches them fine.
