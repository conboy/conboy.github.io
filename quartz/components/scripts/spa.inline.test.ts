import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import test from "node:test"
import vm from "node:vm"
import { transformSync } from "esbuild"

// Exercise the real router with a small DOM boundary. In particular, a
// standalone app must leave the SPA before cleanup or DOM morphing occurs.
function routerHarness(quartzPage: boolean, contentType = "text/html") {
  const calls = { assigned: [] as string[], morphed: 0, cleaned: 0, pushed: 0 }
  const element = () => ({
    style: {},
    dataset: {} as Record<string, string>,
    textContent: "",
    remove() {},
    appendChild() {},
    prepend() {},
    setAttribute() {},
    querySelectorAll: () => [],
    hasAttribute: (name: string) => quartzPage && name === "data-slug",
  })
  const parsed = {
    body: element(),
    head: element(),
    querySelector: () => ({ textContent: "Target" }),
  }
  const context = vm.createContext({
    URL,
    console,
    setTimeout: () => 0,
    HTMLElement: class {},
    CustomEvent: class {
      constructor(public type: string) {}
    },
    customElements: { get: () => true },
    DOMParser: class {
      parseFromString() {
        return parsed
      }
    },
    document: {
      createElement: element,
      body: element(),
      head: element(),
      querySelector: () => null,
      dispatchEvent() {},
      getElementById: () => null,
    },
    window: {
      location: {
        origin: "https://conboy.dev",
        pathname: "/",
        toString: () => "https://conboy.dev/",
        assign: (url: URL) => calls.assigned.push(String(url)),
      },
      addEventListener() {},
      scrollTo() {},
    },
    history: { pushState: () => calls.pushed++ },
    micromorph: () => calls.morphed++,
    getFullSlug: () => "index",
    normalizeRelativeURLs() {},
    fetchCanonical: async () => ({
      headers: { get: () => contentType },
      text: async () => "<html>target</html>",
    }),
  })
  const source = readFileSync(new URL("./spa.inline.ts", import.meta.url), "utf8").replace(
    /^import .*$/gm,
    "",
  )
  vm.runInContext(transformSync(source, { loader: "ts" }).code, context)
  context.window.addCleanup(() => calls.cleaned++)
  return { context, calls }
}

test("standalone HTML loads as a document before cleanup or morphing", async () => {
  const { context, calls } = routerHarness(false)
  await context.window.spaNavigate(new URL("https://conboy.dev/npu-anim/"))
  assert.deepEqual(calls, {
    assigned: ["https://conboy.dev/npu-anim/"],
    morphed: 0,
    cleaned: 0,
    pushed: 0,
  })
})

test("Quartz pages retain SPA navigation and cleanup", async () => {
  const { context, calls } = routerHarness(true)
  await context.window.spaNavigate(new URL("https://conboy.dev/notebook"))
  assert.deepEqual(calls, { assigned: [], morphed: 1, cleaned: 1, pushed: 1 })
})

test("non-HTML resources use full navigation", async () => {
  const { context, calls } = routerHarness(false, "application/pdf")
  await context.window.spaNavigate(new URL("https://conboy.dev/spec.pdf"))
  assert.equal(calls.assigned[0], "https://conboy.dev/spec.pdf")
  assert.equal(calls.morphed, 0)
  assert.equal(calls.cleaned, 0)
})

test("explicit standalone links bypass the router even when a child is clicked", () => {
  const { context } = routerHarness(false)
  context.target = {
    nodeType: 1,
    attributes: { getNamedItem: () => null },
    closest: () => ({ dataset: { routerIgnore: "" }, href: "https://conboy.dev/npu-anim/" }),
  }
  assert.equal(vm.runInContext("getOpts({target})", context), undefined)
})
