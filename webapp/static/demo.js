// Static-demo shim (GitHub Pages build only — scripts/build_static_demo.py
// injects it before app.js; the live server never loads it).
//
// There is no solver behind a static site, so this answers the editor's
// /api/* calls from pre-computed JSON under data/: the catalog, the examples,
// Verilog-A sources, and one captured /api/run result per example. A run is
// looked up by the SHA-256 of its exact request body (the build drives this
// same app.js headlessly, so an unedited example hashes identically in any
// browser). An edited circuit misses and gets a "not simulated" message with a
// pointer to the live simulator.
(() => {
  "use strict";
  const LIVE_URL = "https://codespaces.new/alexsludds/photonflux?quickstart=1";
  const LIVE_LINK = `<a href="${LIVE_URL}" target="_blank" rel="noopener">run it live in Codespaces</a>`;
  const realFetch = window.fetch.bind(window);
  const dataUrl = (p) => new URL("data/" + p, document.baseURI).href;
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
    showBanner(`<b>Not simulated</b>: this demo only has results for the built-in examples ` +
      `as they ship (pick one from <b>Examples</b>). To simulate a circuit you changed, ` +
      `${LIVE_LINK}.`);
    return json({ ok: false, error: "This demo only has pre-computed results for the built-in " +
      "examples. To simulate this circuit, run it live in Codespaces: " + LIVE_URL }, 422);
  }

  window.fetch = async (input, init) => {
    const req = input instanceof Request ? input : null;
    const url = new URL(req ? req.url : String(input), location.href);
    if (!url.pathname.startsWith("/api/")) return realFetch(input, init);
    const route = url.pathname;
    const method = (init?.method || req?.method || "GET").toUpperCase();
    if (method === "POST") {
      if (route === "/api/run")
        return run(init?.body != null ? String(init.body) : req ? await req.text() : "");
      if (route === "/api/cancel") return json({ ok: true });
      if (route === "/api/eyemeasure")
        return json({ ok: false, error: `The stateye / TDECQ measurement needs the live simulator — ${LIVE_LINK}.` }, 422);
      if (route === "/api/schematic") return json({ ok: false, disabled: true }, 403);
      return json({ ok: false, error: `Not available in the demo — ${LIVE_LINK}.` }, 403);
    }
    if (route === "/api/components") return file("components.json");
    if (route === "/api/examples") return file("examples.json");
    if (route.startsWith("/api/examples/"))
      return file(`examples/${route.slice("/api/examples/".length)}.json`);
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
