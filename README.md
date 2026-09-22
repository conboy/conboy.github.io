# Conrad Fernandez — portfolio and technical notebook

Source for https://conboy.dev, built with Quartz v5 and deployed to GitHub Pages.

## Local development

Use Node.js 22 or newer:

```sh
npm ci
npx quartz build --serve
```

## Site structure

- `content/index.md`: portfolio landing page.
- `content/projects/`: project case studies and public evidence links.
- `content/notebook.md`: entry point for technical notes.
- `content/npu-course/`: NPU learning roadmap; capstones are labelled as targets.
- `content/assets/`: portfolio diagrams and the LoRa project photograph.
- `quartz/styles/custom.scss`: responsive portfolio styling, scoped to the homepage.
- `quartz.config.yaml`: notebook layout, homepage component visibility, and site links.

The LoRa photo comes from the public `conboy/LoRa-Wildfire-Detection-System` development log. Project case studies cite their source repositories and public announcements. Update measured results and project status when new evidence is available.

## Standalone animation

`https://conboy.dev/npu-anim/` is served by the separate `conboy/npu-anim` project. It is not a Quartz article. Demo links use `data-router-ignore` to request a full document load. The SPA router also falls back to normal navigation for HTML without Quartz's `body[data-slug]` marker, so standalone scripts initialize in their own document.

## Verification and deployment

```sh
npx tsc --noEmit
npm test
npx quartz build
```

The deployment workflow runs the content guard, typecheck, tests, build, and GitHub Pages deployment on pushes to `main`. Follow [WRITING.md](WRITING.md) when adding notes.
