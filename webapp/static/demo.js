// Static-demo shim (GitHub Pages build only — scripts/build_static_demo.py
// injects it before app.js; the live server never loads it).
//
// There is no solver behind a static site, so this answers the editor's
// /api/* calls from pre-computed JSON under data/: the catalog, the examples,
// Verilog-A sources, and one captured /api/run result per example. A run is
// looked up by the SHA-256 of its exact request body (the build drives this
// same app.js headlessly, so an unedited example hashes identically in any
// browser). An edited circuit misses; it then shows the unedited example's
// result, labelled, with a pointer to the live simulator.
(() => {
  "use strict";
  const LIVE_URL = "https://codespaces.new/alexsludds/photonflux?quickstart=1";
  const LIVE_LINK = `<a href="${LIVE_URL}" target="_blank" rel="noopener">run it live in Codespaces</a>`;
  const realFetch = window.fetch.bind(window);
  const dataUrl = (p) => new URL("data/" + p, document.baseURI).href;
  // example most recently fetched this visit (a first visit auto-loads 01; a
  // returning visitor's tabs come from autosave, so this stays null)
  let lastExample = null;
  let index = null;

  const json = (obj, status = 200) => new Response(JSON.stringify(obj), {
    status, headers: { "Content-Type": "application/json" } });
  const file = async (p) => {
    const r = await realFetch(dataUrl(p));
    return r.ok ? r : json({ ok: false, error: "not in the demo build" }, 404);
  };
  const loadIndex = async () =>
    index || (index = await (await realFetch(dataUrl("runs/index.json"))).json());

  async function sha256(text) {
    const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
    return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
  }

  async function run(body) {
    const idx = await loadIndex();
    const hit = idx.hashes[await sha256(body)];
    if (hit) return file(`runs/${hit}.json`);
    // Edited circuit: fall back to the pre-computed result of the example it
    // came from, and say so in the log.
    if (lastExample && idx.examples.includes(lastExample)) {
      showBanner(`<b>Your changes were not simulated</b> — this demo only has results for ` +
        `the built-in examples, so the plots show the original example. To simulate your ` +
        `changes, ${LIVE_LINK}.`);
      const res = await (await realFetch(dataUrl(`runs/${lastExample}.json`))).json();
      res.log = [`<span class="err">Demo mode: this circuit was changed, so it was not simulated. ` +
        `Showing the pre-computed result for the original example instead. To simulate your ` +
        `changes, ${LIVE_LINK}.</span>`, ...(res.log || [])];
      return json(res);
    }
    showBanner(`<b>Not simulated</b> — this demo only has results for the built-in examples ` +
      `(pick one from <b>Examples</b>). To simulate this circuit, ${LIVE_LINK}.`);
    return json({ ok: false, error: "This demo only has pre-computed results for the built-in " +
      "examples. To simulate this circuit, run it live in Codespaces: " + LIVE_URL }, 422);
  }

  window.fetch = async (input, init = {}) => {
    const url = new URL(typeof input === "string" ? input : input.url, location.href);
    if (!url.pathname.startsWith("/api/")) return realFetch(input, init);
    const route = url.pathname;
    const method = (init.method || "GET").toUpperCase();
    if (method === "POST") {
      if (route === "/api/run") return run(String(init.body || ""));
      if (route === "/api/cancel") return json({ ok: true });
      if (route === "/api/eyemeasure")
        return json({ ok: false, error: `The stateye / TDECQ measurement needs the live simulator — ${LIVE_LINK}.` }, 422);
      if (route === "/api/schematic") return json({ ok: false, disabled: true }, 403);
      return json({ ok: false, error: `Not available in the demo — ${LIVE_LINK}.` }, 403);
    }
    if (route === "/api/components") return file("components.json");
    if (route === "/api/examples") return file("examples.json");
    if (route.startsWith("/api/examples/")) {
      lastExample = route.slice("/api/examples/".length);
      return file(`examples/${lastExample}.json`);
    }
    if (route === "/api/veriloga")
      return file(`veriloga/${encodeURIComponent(url.searchParams.get("type") || "")}.json`);
    if (route === "/api/progress")
      return json({ active: false, frac: 0, phase: "", runs: 0, run: 0 });
    if (route === "/api/eyemeasure/progress") return json({ active: false });
    if (route === "/api/schematic") return json({ ok: false, disabled: true }, 403);
    return json({ ok: false, error: "not found" }, 404);
  };

  // The notebook-bridge event stream has no server to talk to.
  window.EventSource = class { constructor() {} addEventListener() {} close() {} };

  function showBanner(html) {
    document.getElementById("demo-banner")?.remove();
    const bar = document.createElement("div");
    bar.id = "demo-banner";
    bar.innerHTML = html + ` <button type="button" aria-label="Dismiss">×</button>`;
    bar.querySelector("button").onclick = () => bar.remove();
    document.body.append(bar);
  }

  document.addEventListener("DOMContentLoaded", () => showBanner(
    `<b>Photonflux demo</b>: pick a circuit from <b>Examples</b> and press <b>Run</b>. ` +
    `Every example is pre-simulated, so results appear instantly. To simulate circuits ` +
    `you edit, ${LIVE_LINK} (free with a GitHub account).`));
})();
