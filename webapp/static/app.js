/* photonflux web schematic editor + simulation frontend.
 *
 * Vanilla JS + SVG. State is a plain JSON document (instances/wires/probes,
 * plus optional render-only `notes` text annotations) that maps 1:1 onto the
 * backend's /api/run schematic payload; the backend ignores `notes`. Rendering
 * is a full redraw of a handful of SVG layers (plenty fast at schematic scale).
 */
"use strict";

const GRID = 10;                 // snap pitch [px]
const PROBE_COLORS = ["#fbbf24", "#6ecbf5", "#4ade80", "#f472d0", "#f87171",
                      "#a78bfa", "#fb923c", "#34d399"];

const ID_PREFIX = {
  cw_laser: "LAS", mzm: "MZM", pulse_mod: "MOD", ring_mod: "RING",
  waveguide: "WG", splitter: "SPL", dir_coupler: "DC", photodiode: "PD",
  vdc: "V", vpulse: "VP", vsin: "VS", idc: "I", resistor: "R",
  capacitor: "C", inductor: "L", diode: "D", nmos: "MN", pmos: "MP",
  sky130_nfet: "XN", sky130_nfet_lvt: "XN", sky130_nfet_5v: "XN",
  sky130_nfet_nvt: "XN", sky130_pfet: "XP", sky130_pfet_lvt: "XP",
  sky130_pfet_hvt: "XP", sky130_pfet_5v: "XP",
  sky130_res_po: "R", sky130_res_nd: "R", sky130_res_high_po: "R",
  sky130_res_xhigh_po: "R", sky130_cap_mim: "C", sky130_cap_mim2: "C",
  opamp: "OA", ground: "GND",
};

// ---------------------------------------------------------------------------
// global state
// ---------------------------------------------------------------------------
let CAT = {};                    // backend catalog: type -> {ports, params, ...}
// `globals` holds schematic-wide parameters. `baud` (symbols/s) is the single
// source of truth for the signaling rate: the PRBS unit interval and the eye
// fold both derive UI = 1/baud from it, rather than each carrying their own.
const DEFAULT_BAUD = 10e9;       // 10 GBd -> 100 ps UI
let state = { instances: {}, wires: [], probes: [], notes: [],
              globals: { baud: DEFAULT_BAUD } };
let undoStack = [], redoStack = [];
let view = { x: 60, y: 40, k: 1 };

let mode = { kind: "idle" };     // idle | place | wire | probe | drag | pan
let selection = null;            // {kind:"inst"|"wire"|"probe", id}
let lastResult = null;
let plots = [];

const $ = (id) => document.getElementById(id);
const svg = $("canvas");
const layers = {
  notes: $("layer-notes"), wires: $("layer-wires"), comps: $("layer-comps"),
  probes: $("layer-probes"), tool: $("layer-tool"),
};

// ---------------------------------------------------------------------------
// geometry helpers
// ---------------------------------------------------------------------------
const snap = (v) => Math.round(v / GRID) * GRID;

function rotPt(px, py, w, h, rot) {
  const cx = w / 2, cy = h / 2, dx = px - cx, dy = py - cy;
  switch (((rot % 360) + 360) % 360) {
    case 90:  return [cx - dy, cy + dx];
    case 180: return [cx - dx, cy - dy];
    case 270: return [cx + dy, cy - dx];
    default:  return [px, py];
  }
}

// component-local point -> world: horizontal flip (mirror about the vertical
// centre line) is applied first, then rotation, then the instance origin.
// The SVG group transform below mirrors this order exactly.
function localToWorld(inst, sym, px, py) {
  if (inst.flip) px = sym.w - px;
  const [rx, ry] = rotPt(px, py, sym.w, sym.h, inst.rot || 0);
  return [inst.x + rx, inst.y + ry];
}

// full group transform string for an instance (flip composes with rotate to
// give all 8 orientations of the dihedral group)
function compTransform(inst, sym) {
  let tf = `translate(${inst.x},${inst.y}) `
         + `rotate(${inst.rot || 0} ${sym.w / 2} ${sym.h / 2})`;
  if (inst.flip) tf += ` translate(${sym.w},0) scale(-1,1)`;
  return tf;
}

function pinPos(inst, pin) {
  return localToWorld(inst, S[inst.type], ...S[inst.type].pins[pin]);
}

function portDomain(type, pin) {
  const e = CAT[type];
  const p = e && e.ports.find((p) => p.name === pin);
  return p ? p.domain : "electrical";
}

function toWorld(ev) {
  const r = svg.getBoundingClientRect();
  return [(ev.clientX - r.left - view.x) / view.k,
          (ev.clientY - r.top - view.y) / view.k];
}

// ---------------------------------------------------------------------------
// state mutation + undo
// ---------------------------------------------------------------------------
function commit(mutate) {
  undoStack.push(JSON.stringify(state));
  if (undoStack.length > 100) undoStack.shift();
  redoStack = [];
  mutate();
  autosave();
  render();
}

function undo() {
  if (!undoStack.length) return;
  redoStack.push(JSON.stringify(state));
  state = JSON.parse(undoStack.pop());
  selection = null; autosave(); render();
}
function redo() {
  if (!redoStack.length) return;
  undoStack.push(JSON.stringify(state));
  state = JSON.parse(redoStack.pop());
  selection = null; autosave(); render();
}

function autosave() {
  try { localStorage.setItem("photonflux_sch", JSON.stringify(state)); } catch {}
}

// ---------------------------------------------------------------------------
// global baud rate (schematic-wide). UI = 1/baud; the PRBS source and the eye
// diagram both pull from here instead of carrying a per-instance unit interval.
// ---------------------------------------------------------------------------
function globalBaud() {
  const b = state.globals && state.globals.baud;
  return b > 0 ? b : DEFAULT_BAUD;
}
function globalUI() { return 1 / globalBaud(); }

// Make `state.globals.baud` authoritative. On a schematic that predates the
// global (or a fresh load), seed it from the first PRBS source's unit interval
// so legacy circuits keep their rate, then strip the now-vestigial per-instance
// `ui` so the global stays the sole source of truth.
function adoptGlobals(st) {
  st.globals = st.globals || {};
  if (!(st.globals.baud > 0)) {
    let baud = DEFAULT_BAUD;
    for (const inst of Object.values(st.instances || {})) {
      const ui = inst.type === "prbs" ? (inst.settings || {}).ui : undefined;
      if (ui > 0) { baud = 1 / ui; break; }
    }
    st.globals.baud = baud;
  }
  for (const inst of Object.values(st.instances || {})) {
    if (inst.type === "prbs" && inst.settings) delete inst.settings.ui;
  }
}

// mirror state -> top-bar input (skip while the user is mid-edit). fmtNum, not
// fmtSI: baud is an editable field that must round-trip through parseSI, so a
// fractional lane rate (10.3125 GBd) isn't collapsed to "10.3G" and corrupted.
function syncGlobalsUI() {
  const el = document.getElementById("glob-baud");
  if (el && document.activeElement !== el) el.value = fmtNum(globalBaud());
}

function newId(type) {
  const prefix = ID_PREFIX[type] || "U";
  let n = 1;
  while (state.instances[prefix + n]) n++;
  return prefix + n;
}

function freshProbeName() {
  let n = 1;
  while (state.probes.some((p) => p.name === "probe" + n)) n++;
  return "probe" + n;
}

function deleteSelection() {
  if (!selection) return;
  const sel = selection;
  commit(() => {
    if (sel.kind === "inst") {
      delete state.instances[sel.id];
      state.wires = state.wires.filter(
        (w) => w.from.split(",")[0] !== sel.id && w.to.split(",")[0] !== sel.id);
      state.probes = state.probes.filter((p) => p.at.split(",")[0] !== sel.id);
    } else if (sel.kind === "wire") {
      state.wires.splice(sel.id, 1);
    } else if (sel.kind === "probe") {
      state.probes.splice(sel.id, 1);
    }
  });
  selection = null;
  renderInspector();
}

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------
function render() {
  syncGlobalsUI();
  layers.notes.innerHTML = "";
  layers.comps.innerHTML = "";
  layers.wires.innerHTML = "";
  layers.probes.innerHTML = "";
  $("viewport").setAttribute(
    "transform", `translate(${view.x},${view.y}) scale(${view.k})`);

  // --- notes (render-only text annotations, painted behind the circuit) -----
  for (const note of state.notes || []) renderNote(note);

  // wired-port + junction bookkeeping
  const useCount = {};
  for (const w of state.wires) {
    useCount[w.from] = (useCount[w.from] || 0) + 1;
    useCount[w.to] = (useCount[w.to] || 0) + 1;
  }

  // --- wires ---------------------------------------------------------------
  state.wires.forEach((w, i) => {
    const [x1, y1] = pinPos(state.instances[w.from.split(",")[0]], w.from.split(",")[1]);
    const [x2, y2] = pinPos(state.instances[w.to.split(",")[0]], w.to.split(",")[1]);
    const dom = portDomain(state.instances[w.from.split(",")[0]].type, w.from.split(",")[1]);
    const d = wirePath(x1, y1, x2, y2, dom);
    const g = document.createElementNS(svg.namespaceURI, "g");
    g.innerHTML = `
      <path class="wire-hit" d="${d}"/>
      <path class="wire ${dom} ${selection?.kind === "wire" && selection.id === i ? "selected" : ""}" d="${d}"/>`;
    g.addEventListener("mousedown", (ev) => {
      ev.stopPropagation();
      selection = { kind: "wire", id: i };
      renderInspector(); render();
    });
    layers.wires.appendChild(g);
  });

  // --- components ----------------------------------------------------------
  for (const [id, inst] of Object.entries(state.instances)) {
    const sym = S[inst.type];
    if (!sym) continue;
    const compClass = "comp" +
      (selection?.kind === "inst" && selection.id === id ? " selected" : "");
    const g = document.createElementNS(svg.namespaceURI, "g");
    g.setAttribute("class", compClass);
    g.setAttribute("transform", compTransform(inst, sym));
    g.innerHTML = sym.draw();

    // Text (refdes/value/pin labels) lives in a separate, un-transformed
    // overlay so it always renders upright and unmirrored, positioned by
    // transforming its anchor through the same flip+rotate as the body. It
    // carries the same `comp`/`selected` class as the body group so the
    // `.comp text` / `.comp .refdes` fills apply — without it the labels fall
    // back to SVG-default black and vanish against the dark canvas.
    const labels = document.createElementNS(svg.namespaceURI, "g");
    labels.setAttribute("class", compClass);
    labels.style.pointerEvents = "none";
    let ltext = "";
    const [lx, ly] = sym.label || [0, -6];
    if (!sym.hideRef) {
      const hp = HEADLINE_PARAM[inst.type];
      const cat = CAT[inst.type];
      let valTxt = "";
      if (inst.type === "prbs") {
        valTxt = `${fmtSI(globalUI())}s`;   // UI pulled from the global baud rate
      } else if (hp && cat) {
        const spec = cat.params.find((p) => p.name === hp);
        const v = (inst.settings && inst.settings[hp] !== undefined)
          ? inst.settings[hp] : spec?.default;
        if (v !== undefined) valTxt = `${fmtSI(+v)}${spec?.unit || ""}`;
      }
      const [wx, wy] = localToWorld(inst, sym, lx, ly);
      ltext += `<text class="refdes" x="${wx}" y="${wy - 11}">${id}</text>
        <text x="${wx}" y="${wy}">${valTxt}</text>`;
    }
    // pin labels for multiport symbols: offset each label away from the
    // (transform-invariant) component centre so it stays clear of the body
    if (sym.pinLabels) {
      const ccx = inst.x + sym.w / 2, ccy = inst.y + sym.h / 2;
      for (const pn of Object.keys(sym.pins)) {
        const [px, py] = localToWorld(inst, sym, ...sym.pins[pn]);
        const dx = px - ccx, dy = py - ccy;
        const len = Math.hypot(dx, dy) || 1;
        const ex = px + (dx / len) * 7, ey = py + (dy / len) * 7 + 3;
        const anchor = dx > 2 ? "start" : dx < -2 ? "end" : "middle";
        ltext += `<text x="${ex}" y="${ey}" text-anchor="${anchor}"
          style="font-size:8px">${pn}</text>`;
      }
    }
    labels.innerHTML = ltext;

    // ports (drawn inside the rotated group at local coords)
    for (const pn of Object.keys(sym.pins)) {
      const [px, py] = sym.pins[pn];
      const dom = portDomain(inst.type, pn);
      const wired = useCount[`${id},${pn}`] > 0;
      const c = document.createElementNS(svg.namespaceURI, "circle");
      c.setAttribute("cx", px); c.setAttribute("cy", py); c.setAttribute("r", 4);
      c.setAttribute("class", `port ${dom}${wired ? " wired" : ""}`);
      c.dataset.ep = `${id},${pn}`;
      c.addEventListener("mousedown", (ev) => {
        ev.stopPropagation();
        onPortMouseDown(`${id},${pn}`);
      });
      c.addEventListener("mouseup", (ev) => {
        if (mode.kind === "wire" && mode.from !== `${id},${pn}`) {
          ev.stopPropagation();
          finishWire(`${id},${pn}`);
        }
      });
      g.appendChild(c);
    }

    g.addEventListener("mousedown", (ev) => {
      if (ev.target.classList.contains("port")) return;
      ev.stopPropagation();
      if (mode.kind === "probe" || mode.kind === "wire" || mode.kind === "place") return;
      selection = { kind: "inst", id };
      const [wx, wy] = toWorld(ev);
      mode = { kind: "drag", id, dx: wx - inst.x, dy: wy - inst.y, moved: false,
               snapshot: JSON.stringify(state) };
      renderInspector(); render();
    });
    layers.comps.appendChild(g);
    layers.comps.appendChild(labels);
  }

  // --- junction dots ---------------------------------------------------------
  for (const [ep, n] of Object.entries(useCount)) {
    if (n < 2) continue;
    const [id, pin] = ep.split(",");
    const inst = state.instances[id];
    if (!inst) continue;
    const [x, y] = pinPos(inst, pin);
    const dom = portDomain(inst.type, pin);
    const c = document.createElementNS(svg.namespaceURI, "circle");
    c.setAttribute("cx", x); c.setAttribute("cy", y); c.setAttribute("r", 3.2);
    c.setAttribute("class", `junction ${dom}`);
    c.style.pointerEvents = "none";
    layers.wires.appendChild(c);
  }

  // --- probes -----------------------------------------------------------------
  state.probes.forEach((p, i) => {
    const [id, pin] = p.at.split(",");
    const inst = state.instances[id];
    if (!inst) return;
    const [x, y] = pinPos(inst, pin);
    const g = document.createElementNS(svg.namespaceURI, "g");
    g.setAttribute("class", "probe-flag" +
      (selection?.kind === "probe" && selection.id === i ? " selected" : "") +
      (p.hide ? " hidden-probe" : ""));
    const dot = p.hide
      ? `<circle cx="${x + 16}" cy="${y - 18}" r="6" fill="none"
           stroke="${p.color}" stroke-width="1.5" stroke-dasharray="2 2"/>`
      : `<circle cx="${x + 16}" cy="${y - 18}" r="6" fill="${p.color}" stroke="none"/>`;
    g.innerHTML = `
      <line x1="${x}" y1="${y}" x2="${x + 12}" y2="${y - 14}" stroke="${p.color}" stroke-width="1.5"/>
      ${dot}
      <text x="${x + 26}" y="${y - 14}" fill="${p.color}">${p.name}${p.hide ? " (hidden)" : ""}</text>`;
    g.addEventListener("mousedown", (ev) => {
      ev.stopPropagation();
      selection = { kind: "probe", id: i };
      renderInspector(); render();
    });
    layers.probes.appendChild(g);
  });

  updateSweepSelectors();
  updateRunCount();
}

// A note is a plain annotation box: {x, y, title?, text|lines, w?}. Text is
// NOT auto-wrapped — author it as `lines` (array) or `text` with "\n" breaks.
// Render-only: pointer-events are off so notes never intercept schematic edits.
function renderNote(note) {
  const body = note.lines || String(note.text || "").split("\n");
  const rows = note.title ? [note.title, ...body] : body;
  const PAD = 9, LH = 15, FS = 11, CHAR_W = 6.1;
  const longest = rows.reduce((m, s) => Math.max(m, s.length), 0);
  const w = note.w || Math.ceil(longest * CHAR_W) + PAD * 2;
  const h = PAD * 2 + rows.length * LH;
  const x = note.x || 0, y = note.y || 0;
  const tspans = rows.map((s, i) => {
    const cls = note.title && i === 0 ? "note-title" : "note-line";
    const dy = i === 0 ? FS : LH;
    return `<tspan class="${cls}" x="${x + PAD}" dy="${dy}">${escapeHtml(s)}</tspan>`;
  }).join("");
  const g = document.createElementNS(svg.namespaceURI, "g");
  g.setAttribute("class", "note");
  g.style.pointerEvents = "none";
  g.innerHTML = `
    <rect class="note-box" x="${x}" y="${y}" width="${w}" height="${h}" rx="6"/>
    <text class="note-text" y="${y + PAD}">${tspans}</text>`;
  layers.notes.appendChild(g);
}

function wirePath(x1, y1, x2, y2, dom) {
  if (dom === "optical") {
    const d = Math.max(40, Math.abs(x2 - x1) * 0.5);
    return `M ${x1} ${y1} C ${x1 + d} ${y1}, ${x2 - d} ${y2}, ${x2} ${y2}`;
  }
  const mx = snap((x1 + x2) / 2);
  return `M ${x1} ${y1} L ${mx} ${y1} L ${mx} ${y2} L ${x2} ${y2}`;
}

// ---------------------------------------------------------------------------
// interaction: wiring / placing / probing
// ---------------------------------------------------------------------------
function onPortMouseDown(ep) {
  if (mode.kind === "probe") {
    const name = freshProbeName();
    const color = PROBE_COLORS[state.probes.length % PROBE_COLORS.length];
    commit(() => state.probes.push({ name, at: ep, color }));
    setHint("Probe placed. Click another port, or Esc to exit probe mode.");
    return;
  }
  if (mode.kind === "wire") {
    if (mode.from !== ep) finishWire(ep);
    return;
  }
  mode = { kind: "wire", from: ep };
  setHint("Click a destination port to finish the wire — Esc cancels.");
}

function finishWire(ep) {
  const from = mode.from;
  const dFrom = portDomain(state.instances[from.split(",")[0]].type, from.split(",")[1]);
  const dTo = portDomain(state.instances[ep.split(",")[0]].type, ep.split(",")[1]);
  const isGnd = (e) => state.instances[e.split(",")[0]].type === "ground";
  if (dFrom !== dTo && !isGnd(from) && !isGnd(ep)) {
    setHint(`Cannot connect ${dFrom} to ${dTo} — use a photodiode or modulator to bridge domains.`, true);
    return;
  }
  const dup = state.wires.some((w) =>
    (w.from === from && w.to === ep) || (w.from === ep && w.to === from));
  mode = { kind: "idle" };
  layers.tool.innerHTML = "";
  if (!dup) commit(() => state.wires.push({ from, to: ep }));
  setHint("");
}

function armPlacement(type) {
  mode = { kind: "place", type, rot: 0, flip: false };
  document.querySelectorAll(".pal-item").forEach((el) =>
    el.classList.toggle("armed", el.dataset.type === type));
  setHint(`Placing ${CAT[type]?.label || type} — click to drop, `
    + `R rotates, F flips, Esc cancels.`);
}

function disarm() {
  mode = { kind: "idle" };
  layers.tool.innerHTML = "";
  document.querySelectorAll(".pal-item").forEach((el) => el.classList.remove("armed"));
  setHint("");
}

function setHint(msg, isErr) {
  const el = $("canvas-hint");
  el.textContent = msg;
  el.style.color = isErr ? "var(--err)" : "var(--text-dim)";
  if (isErr) setTimeout(() => { if (el.textContent === msg) el.textContent = ""; }, 4000);
}

// --- svg-level mouse handling -----------------------------------------------
svg.addEventListener("mousedown", (ev) => {
  if (mode.kind === "place") {
    const [wx, wy] = toWorld(ev);
    const type = mode.type, rot = mode.rot, flip = !!mode.flip;
    const id = newId(type);
    commit(() => {
      state.instances[id] = {
        type, x: snap(wx - S[type].w / 2), y: snap(wy - S[type].h / 2),
        rot, ...(flip ? { flip: true } : {}), settings: {},
      };
    });
    if (!ev.shiftKey) disarm();
    return;
  }
  if (mode.kind === "wire" || mode.kind === "probe") return;
  // background: pan + deselect
  selection = null;
  renderInspector();
  mode = { kind: "pan", sx: ev.clientX, sy: ev.clientY, ox: view.x, oy: view.y };
  render();
});

svg.addEventListener("mousemove", (ev) => {
  if (mode.kind === "pan") {
    view.x = mode.ox + ev.clientX - mode.sx;
    view.y = mode.oy + ev.clientY - mode.sy;
    $("viewport").setAttribute("transform",
      `translate(${view.x},${view.y}) scale(${view.k})`);
    return;
  }
  if (mode.kind === "drag") {
    const [wx, wy] = toWorld(ev);
    const inst = state.instances[mode.id];
    const nx = snap(wx - mode.dx), ny = snap(wy - mode.dy);
    if (nx !== inst.x || ny !== inst.y) {
      if (!mode.moved) { undoStack.push(mode.snapshot); redoStack = []; mode.moved = true; }
      inst.x = nx; inst.y = ny;
      render();
    }
    return;
  }
  if (mode.kind === "place") {
    const [wx, wy] = toWorld(ev);
    const sym = S[mode.type];
    const ghost = {
      x: snap(wx - sym.w / 2), y: snap(wy - sym.h / 2),
      rot: mode.rot, flip: mode.flip,
    };
    layers.tool.innerHTML =
      `<g opacity="0.5" transform="${compTransform(ghost, sym)}">${sym.draw()}</g>`;
    return;
  }
  if (mode.kind === "wire") {
    const [wx, wy] = toWorld(ev);
    const from = mode.from;
    const inst = state.instances[from.split(",")[0]];
    const [x1, y1] = pinPos(inst, from.split(",")[1]);
    const dom = portDomain(inst.type, from.split(",")[1]);
    layers.tool.innerHTML =
      `<path class="wire ghost ${dom}" d="${wirePath(x1, y1, wx, wy, dom)}"/>`;
  }
});

window.addEventListener("mouseup", () => {
  if (mode.kind === "pan" || mode.kind === "drag") {
    if (mode.kind === "drag" && mode.moved) autosave();
    mode = { kind: "idle" };
  }
});

svg.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  const r = svg.getBoundingClientRect();
  const mx = ev.clientX - r.left, my = ev.clientY - r.top;
  const factor = Math.exp(-ev.deltaY * 0.0015);
  const k2 = Math.min(4, Math.max(0.2, view.k * factor));
  view.x = mx - (mx - view.x) * (k2 / view.k);
  view.y = my - (my - view.y) * (k2 / view.k);
  view.k = k2;
  $("viewport").setAttribute("transform",
    `translate(${view.x},${view.y}) scale(${view.k})`);
}, { passive: false });

// --- keyboard -----------------------------------------------------------------
window.addEventListener("keydown", (ev) => {
  const tag = document.activeElement?.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
  const mod = ev.metaKey || ev.ctrlKey;
  if (mod && ev.key === "z" && !ev.shiftKey) { ev.preventDefault(); undo(); return; }
  if (mod && (ev.key === "y" || (ev.key === "z" && ev.shiftKey))) { ev.preventDefault(); redo(); return; }
  if (mod && ev.key === "s") { ev.preventDefault(); saveJSON(); return; }
  if (mod && ev.key === "Enter") { ev.preventDefault(); runSim(); return; }
  switch (ev.key) {
    case "Escape": if (runAbort) { stopSim(); } else { disarm(); render(); } break;
    case "Delete": case "Backspace": ev.preventDefault(); deleteSelection(); break;
    case "r": case "R":
      if (mode.kind === "place") { mode.rot = (mode.rot + 90) % 360; }
      else if (selection?.kind === "inst") rotateSelected(90);
      break;
    case "f": case "F":
      // horizontal flip; Shift+F flips vertically (= h-flip + 180 rotate)
      if (mode.kind === "place") {
        mode.flip = !mode.flip;
        if (ev.shiftKey) mode.rot = (mode.rot + 180) % 360;
      } else if (selection?.kind === "inst") {
        flipSelected(ev.shiftKey);
      }
      break;
    case "p": case "P":
      mode = { kind: "probe" };
      setHint("Probe mode — click a port to attach a probe. Esc exits.");
      break;
  }
});

function rotateSelected(deg) {
  const id = selection.id;
  commit(() => {
    const inst = state.instances[id];
    inst.rot = ((inst.rot || 0) + deg + 360) % 360;
  });
  renderInspector();
}

// Horizontal flip (mirror about the component's vertical axis); vertical flip
// is the same mirror composed with a 180-degree rotation.
function flipSelected(vertical) {
  const id = selection.id;
  commit(() => {
    const inst = state.instances[id];
    inst.flip = !inst.flip;
    if (!inst.flip) delete inst.flip;   // keep JSON clean when unset
    if (vertical) inst.rot = ((inst.rot || 0) + 180) % 360;
  });
  renderInspector();
}

// ---------------------------------------------------------------------------
// Verilog-A source viewer (read-only modal)
// ---------------------------------------------------------------------------
function showVeriloga(type, path) {
  document.getElementById("va-modal")?.remove();  // one at a time
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const back = document.createElement("div");
  back.id = "va-modal";
  back.className = "modal-backdrop";
  back.innerHTML = `
    <div class="modal" role="dialog" aria-label="Verilog-A source">
      <div class="modal-head">
        <span class="modal-title">${esc(path)}</span>
        <button class="modal-close" title="Close (Esc)">&#x2715;</button>
      </div>
      <pre class="modal-body" tabindex="0">Loading&hellip;</pre>
    </div>`;
  document.body.appendChild(back);
  const close = () => { back.remove(); document.removeEventListener("keydown", onKey); };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  document.addEventListener("keydown", onKey);
  back.addEventListener("mousedown", (e) => { if (e.target === back) close(); });
  back.querySelector(".modal-close").onclick = close;
  const pre = back.querySelector(".modal-body");
  pre.focus();
  fetch("/api/veriloga?type=" + encodeURIComponent(type))
    .then((r) => r.json())
    .then((d) => {
      pre.textContent = d.ok ? d.source
        : `Could not load Verilog-A source: ${d.error || "unknown error"}`;
    })
    .catch((e) => { pre.textContent = `Could not load Verilog-A source: ${e}`; });
}

// ---------------------------------------------------------------------------
// inspector
// ---------------------------------------------------------------------------
function renderInspector() {
  const body = $("inspector-body");
  if (!selection) {
    body.innerHTML = `<div class="insp-empty">Nothing selected.<br><br>
      Select a component to edit its parameters, a wire to inspect the net,
      or a probe to rename it.</div>`;
    return;
  }
  if (selection.kind === "inst") {
    const id = selection.id;
    const inst = state.instances[id];
    const cat = CAT[inst.type] || { params: [], doc: "", label: inst.type };
    let html = `
      <div class="insp-title"><input id="insp-rename" value="${id}"
        style="width:100%;background:var(--panel2);color:var(--text);
        border:1px solid var(--border);border-radius:4px;padding:3px 6px;font:inherit;font-weight:700"></div>
      <div class="insp-type">${cat.label}</div>
      <div class="insp-btns">
        <button id="insp-rot-ccw" title="Rotate CCW">&#x27F2;</button>
        <button id="insp-rot-cw" title="Rotate CW (R)">&#x27F3;</button>
        <button id="insp-flip-h" title="Flip horizontal (F)">&#x2194;</button>
        <button id="insp-flip-v" title="Flip vertical (Shift+F)">&#x2195;</button>
        <button id="insp-del" class="danger" title="Delete (Del)">&#x2715;</button>
      </div>`;
    const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
    for (const p of cat.params) {
      const v = inst.settings?.[p.name] !== undefined ? inst.settings[p.name] : p.default;
      if (p.kind === "enum") {
        const opts = (p.choices || []).map((c) =>
          `<option value="${c}" ${String(c) === String(v) ? "selected" : ""}>${c}</option>`).join("");
        html += `<div class="insp-row">
          <label title="${p.name}">${p.label}</label>
          <select data-eparam="${p.name}" style="flex:1;background:var(--panel2);
            color:var(--text);border:1px solid var(--border);border-radius:4px;
            padding:3px 4px;font:inherit">${opts}</select>
          <span class="unit">${p.unit || ""}</span></div>`;
      } else if (p.kind === "file") {
        const cur = String(v || "");
        html += `<div class="insp-row">
          <label title="${p.name}">${p.label}</label>
          <button data-upload="${p.name}">Upload&hellip;</button></div>
          <div class="insp-type" data-upname="${p.name}"
            style="word-break:break-all">${cur ? cur.replace(/^[0-9a-f]+_/, "") : "no file"}</div>`;
      } else if (p.kind === "text") {
        html += `<div class="insp-row"><label title="${p.name}">${p.label}</label>
          <button data-loadcsv="${p.name}" style="margin-left:auto">Load CSV&hellip;</button></div>
          <textarea data-tparam="${p.name}" rows="7" spellcheck="false"
            style="width:100%;background:var(--panel2);color:var(--text);
            border:1px solid var(--border);border-radius:4px;padding:4px 6px;
            font:11px/1.5 ui-monospace,monospace;resize:vertical">${esc(v)}</textarea>`;
      } else {
        html += `<div class="insp-row">
          <label title="${p.name}">${p.label}</label>
          <input data-param="${p.name}" value="${fmtNum(+v)}">
          <span class="unit">${p.unit || ""}</span></div>`;
      }
    }
    if (cat.veriloga) {
      const vaTitle = esc(cat.veriloga).replace(/"/g, "&quot;");
      html += `<button id="insp-va" class="insp-va-btn"
        title="View ${vaTitle}">&#x1F441; View Verilog-A source</button>`;
    }
    html += `<div class="insp-doc">${cat.doc || ""}</div>`;
    body.innerHTML = html;

    body.querySelectorAll("input[data-param]").forEach((inp) => {
      inp.addEventListener("change", () => {
        const val = parseSI(inp.value);
        const p = cat.params.find((q) => q.name === inp.dataset.param);
        if (isNaN(val)) { inp.value = fmtNum(+(inst.settings?.[p.name] ?? p.default)); return; }
        commit(() => {
          inst.settings = inst.settings || {};
          inst.settings[p.name] = val;
        });
        inp.value = fmtNum(val);
      });
      inp.addEventListener("keydown", (e) => { if (e.key === "Enter") inp.blur(); });
    });
    body.querySelectorAll("select[data-eparam]").forEach((sel) => {
      sel.addEventListener("change", () => {
        const p = cat.params.find((q) => q.name === sel.dataset.eparam);
        const numeric = typeof (p.choices || [])[0] === "number";
        commit(() => {
          inst.settings = inst.settings || {};
          inst.settings[p.name] = numeric ? parseFloat(sel.value) : sel.value;
        });
      });
    });
    body.querySelectorAll("textarea[data-tparam]").forEach((ta) => {
      ta.addEventListener("change", () => {
        commit(() => {
          inst.settings = inst.settings || {};
          inst.settings[ta.dataset.tparam] = ta.value;
        });
      });
    });
    body.querySelectorAll("button[data-upload]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const file = document.createElement("input");
        file.type = "file";
        file.accept = ".s2p,.s1p,.txt,.dat";
        file.onchange = () => {
          const f = file.files[0];
          if (!f) return;
          f.text().then(async (text) => {
            const resp = await (await fetch("/api/upload", {
              method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ name: f.name, content: text }),
            })).json();
            if (!resp.ok) { alert(`upload failed: ${resp.error}`); return; }
            commit(() => {
              inst.settings = inst.settings || {};
              inst.settings[btn.dataset.upload] = resp.id;
            });
            const lab = body.querySelector(
              `[data-upname="${btn.dataset.upload}"]`);
            if (lab) lab.textContent = resp.name;
          });
        };
        file.click();
      });
    });
    body.querySelectorAll("button[data-loadcsv]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const file = document.createElement("input");
        file.type = "file";
        file.accept = ".csv,.txt,.dat";
        file.onchange = () => {
          const f = file.files[0];
          if (!f) return;
          f.text().then((text) => {
            // keep lines whose first two comma/space fields parse as numbers
            const rows = text.split(/\r?\n/).map((line) => {
              const cols = line.trim().split(/[,;\s]+/);
              return cols.length >= 2 && isFinite(+cols[0]) && isFinite(+cols[1])
                ? `${cols[0]} ${cols[1]}` : null;
            }).filter(Boolean);
            if (!rows.length) { alert("No numeric 't v' rows found in file."); return; }
            const ta = body.querySelector(
              `textarea[data-tparam="${btn.dataset.loadcsv}"]`);
            ta.value = rows.join("\n");
            ta.dispatchEvent(new Event("change"));
          });
        };
        file.click();
      });
    });
    $("insp-rot-cw").onclick = () => rotateSelected(90);
    $("insp-rot-ccw").onclick = () => rotateSelected(-90);
    $("insp-flip-h").onclick = () => flipSelected(false);
    $("insp-flip-v").onclick = () => flipSelected(true);
    $("insp-del").onclick = deleteSelection;
    if (cat.veriloga) $("insp-va").onclick = () => showVeriloga(inst.type, cat.veriloga);
    $("insp-rename").addEventListener("change", () => renameInstance(id, $("insp-rename").value));
    return;
  }
  if (selection.kind === "wire") {
    const w = state.wires[selection.id];
    body.innerHTML = `
      <div class="insp-title">Wire</div>
      <div class="insp-type">${w.from} &#8596; ${w.to}</div>
      <div class="insp-btns"><button id="insp-del" class="danger">Delete wire</button></div>`;
    $("insp-del").onclick = deleteSelection;
    return;
  }
  if (selection.kind === "probe") {
    const p = state.probes[selection.id];
    const [pi, pp] = (p.at || "").split(",");
    const isOptical = portDomain(state.instances[pi]?.type, pp) === "optical";
    const specRow = isOptical ? `
      <div class="insp-row"><label for="insp-pspec">Optical spectrum</label>
        <input type="checkbox" id="insp-pspec" style="width:auto"
          ${p.spectrum ? "checked" : ""}></div>
      <div class="insp-doc">Adds an OSA-style plot of this node — |FFT(E)|&sup2;
        (dB) vs wavelength, centred on the laser carrier — beside the
        transient traces. One coherent carrier per node, so it shows that
        wavelength's carrier and its modulation sidebands.</div>` : "";
    body.innerHTML = `
      <div class="insp-title" style="color:${p.color}">&#9873; Probe</div>
      <div class="insp-row"><label>Name</label><input id="insp-pname" value="${p.name}"></div>
      <div class="insp-type">at ${p.at}</div>
      <div class="insp-row"><label for="insp-phide">Hide from plots</label>
        <input type="checkbox" id="insp-phide" style="width:auto"
          ${p.hide ? "checked" : ""}></div>
      <div class="insp-doc">Still measured (kept in the netlist for AC pairing
        and operating-point math) but its traces are left off the plots.</div>
      ${specRow}
      <div class="insp-btns"><button id="insp-del" class="danger">Delete probe</button></div>`;
    $("insp-pname").addEventListener("change", () => {
      const name = $("insp-pname").value.trim();
      if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) { $("insp-pname").value = p.name; return; }
      commit(() => { p.name = name; });
    });
    $("insp-phide").addEventListener("change", () => {
      commit(() => { p.hide = $("insp-phide").checked; });
      rerenderResults();  // reflect immediately without a re-run
    });
    if (isOptical) $("insp-pspec").addEventListener("change", () => {
      commit(() => {
        if ($("insp-pspec").checked) p.spectrum = true; else delete p.spectrum;
      });
    });
    $("insp-del").onclick = deleteSelection;
  }
}

function renameInstance(oldId, newId_) {
  newId_ = newId_.trim();
  if (newId_ === oldId) return;
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(newId_) || state.instances[newId_]) {
    renderInspector(); return;
  }
  commit(() => {
    state.instances[newId_] = state.instances[oldId];
    delete state.instances[oldId];
    const fix = (ep) => ep.split(",")[0] === oldId ? newId_ + "," + ep.split(",")[1] : ep;
    state.wires = state.wires.map((w) => ({ from: fix(w.from), to: fix(w.to) }));
    state.probes.forEach((p) => { p.at = fix(p.at); });
  });
  selection = { kind: "inst", id: newId_ };
  renderInspector();
}

// ---------------------------------------------------------------------------
// palette
// ---------------------------------------------------------------------------
function buildPalette() {
  const holder = $("palette-items");
  holder.innerHTML = "";
  const cats = {};
  for (const [type, entry] of Object.entries(CAT)) {
    (cats[entry.category] = cats[entry.category] || []).push([type, entry]);
  }
  const order = ["Lasers", "Modulators", "Photonic Passives",
                 "Detectors & Bridges", "Channels", "Sources", "Electrical",
                 "Amplifiers & EQ", "SKY130 FETs", "SKY130 Passives",
                 "Reference"];
  for (const cat of order.concat(Object.keys(cats).filter((c) => !order.includes(c)))) {
    if (!cats[cat]) continue;
    const h = document.createElement("div");
    h.className = "pal-cat"; h.textContent = cat;
    holder.appendChild(h);
    for (const [type, entry] of cats[cat]) {
      const sym = S[type];
      if (!sym) continue;
      const item = document.createElement("div");
      item.className = "pal-item";
      item.dataset.type = type;
      item.title = entry.doc || "";
      const scale = Math.min(28 / sym.w, 24 / sym.h, 0.6);
      item.innerHTML = `
        <svg width="34" height="26" viewBox="0 0 34 26">
          <g class="comp" transform="translate(${17 - sym.w * scale / 2},${13 - sym.h * scale / 2}) scale(${scale})">
            ${sym.draw()}</g></svg>
        <span class="pal-label">${entry.label}</span>`;
      item.addEventListener("click", () => {
        if (mode.kind === "place" && mode.type === type) disarm();
        else armPlacement(type);
      });
      holder.appendChild(item);
    }
  }
  // probe tool
  const h = document.createElement("div");
  h.className = "pal-cat"; h.textContent = "Measure";
  holder.appendChild(h);
  const item = document.createElement("div");
  item.className = "pal-item";
  item.innerHTML = `
    <svg width="34" height="26"><circle cx="14" cy="15" r="5" fill="#fbbf24"/>
      <line x1="17" y1="12" x2="26" y2="4" stroke="#fbbf24" stroke-width="1.5"/></svg>
    <span class="pal-label">Probe <b>(P)</b></span>`;
  item.addEventListener("click", () => {
    mode = { kind: "probe" };
    setHint("Probe mode — click a port to attach a probe. Esc exits.");
  });
  holder.appendChild(item);
}

// ---------------------------------------------------------------------------
// analysis toolbar
// ---------------------------------------------------------------------------
function analysisMode() { return $("sel-analysis").value; }

$("sel-analysis").addEventListener("change", () => {
  document.querySelectorAll(".an-group").forEach((g) =>
    g.hidden = g.dataset.mode !== analysisMode());
  updateSweepSelectors();
  if (typeof syncRunCfgDisabled === "function") { syncRunCfgDisabled(); updateRunCount(); }
});

// probe selects for the link-report / pulse-COM / noise controls
function updateProbeSelectors() {
  for (const [id, blank] of [["an-link-probe", true], ["an-pl-probe", false],
                             ["an-nq-probe", false]]) {
    const sel = $(id);
    const prev = sel.value;
    sel.innerHTML = blank ? '<option value="">—</option>' : "";
    for (const p of state.probes) {
      const o = document.createElement("option");
      o.value = p.name; o.textContent = p.name;
      sel.appendChild(o);
    }
    if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
  }
}

// fill an instance <select> with every placed instance that has live params
// (plus an optional "* (all)" lockstep option), preserving the current value
function fillInstOptions(selId, star) {
  const sel = $(selId);
  const prev = sel.value;
  sel.innerHTML = "";
  if (star) {
    const o = document.createElement("option");
    o.value = "*"; o.textContent = "* (all)";
    sel.appendChild(o);
  }
  for (const [id, inst] of Object.entries(state.instances)) {
    if (!(CAT[inst.type]?.params || []).length) continue;
    const o = document.createElement("option");
    o.value = id; o.textContent = id;
    sel.appendChild(o);
  }
  if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
}

// the run-configuration pane owns all sweep instance/param selectors now
function refreshRunCfgSelectors() {
  fillInstOptions("rc-inst", true); fillParamOptions("rc-inst", "rc-param");
  fillInstOptions("rc-inst2", true); fillParamOptions("rc-inst2", "rc-param2");
}

// called on analysis-mode change and after any schematic mutation
function updateSweepSelectors() {
  updateProbeSelectors();
  refreshRunCfgSelectors();
}

function fillParamOptions(instSelId, paramSelId) {
  const instSel = $(instSelId), paramSel = $(paramSelId);
  const prev = paramSel.value;
  paramSel.innerHTML = "";
  let names;
  if (instSel.value === "*") {
    // union of live params over placed instances (baked kinds are skipped
    // server-side; keep the list simple here)
    names = [...new Set(Object.values(state.instances)
      .flatMap((i) => (CAT[i.type]?.params || []).map((p) => p.name)))];
  } else {
    const inst = state.instances[instSel.value];
    if (!inst) return;
    names = (CAT[inst.type]?.params || []).map((p) => p.name);
  }
  for (const n of names) {
    const o = document.createElement("option");
    o.value = n; o.textContent = n;
    paramSel.appendChild(o);
  }
  if ([...paramSel.options].some((o) => o.value === prev)) paramSel.value = prev;
}
// probes may be added after the mode was selected — refresh on focus
$("an-link-probe").addEventListener("focus", updateProbeSelectors);
$("an-pl-probe").addEventListener("focus", updateProbeSelectors);
$("an-nq-probe").addEventListener("focus", updateProbeSelectors);

// ---------------------------------------------------------------------------
// run-configuration pane: parameter sweep + overlay (orthogonal to the
// analysis type). Enabled -> the Run button fans the analysis out over the
// swept value(s) and overlays the plots; disabled -> a single run, unchanged.
// ---------------------------------------------------------------------------
function parseSweepValues(text) {
  const t = (text || "").trim();
  if (!t) return [];
  const m = t.match(/^([^:,]+):([^:,]+):([^:,]+)$/);   // start:stop:count
  if (m) {
    const a = parseSI(m[1]), b = parseSI(m[2]);
    const n = Math.max(2, parseInt(m[3]) || 2);
    if (isNaN(a) || isNaN(b)) return [];
    return Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
  }
  return t.split(",").map((s) => parseSI(s)).filter((v) => !isNaN(v));
}

function readAxis(instId, paramId, valId) {
  const inst = $(instId).value, param = $(paramId).value;
  const values = parseSweepValues($(valId).value);
  if (!inst || !param || !values.length) return null;
  return { instance: inst, param, values };
}

function runCfgAxes() {
  const prim = readAxis("rc-inst", "rc-param", "rc-values");
  const axes = prim ? [prim] : [];
  if (prim && $("rc-2nd-on").checked) {
    const sec = readAxis("rc-inst2", "rc-param2", "rc-values2");
    if (sec) axes.push(sec);
  }
  return axes;
}

function updateRunCount() {
  const on = $("rc-enable").checked && !$("rc-enable").disabled;
  const axes = runCfgAxes();
  const n = axes.length ? axes.reduce((k, ax) => k * ax.values.length, 1) : 1;
  const el = $("rc-count");
  // DC sweeps solve vectorized (no per-run cap); everything else caps at 24
  const capped = on && analysisMode() !== "dc" && n > 24;
  el.textContent = !on ? "off"
    : capped ? `${n} runs (max 24)` : `${n} run${n === 1 ? "" : "s"}`;
  el.classList.toggle("over", capped);
}

function syncRunCfgDisabled() {
  const m = analysisMode();
  const blocked = (m === "optimize");
  $("rc-enable").disabled = blocked;
  if (blocked) $("rc-enable").checked = false;
  const on = $("rc-enable").checked && !blocked;
  $("rc-body").setAttribute("aria-disabled", String(!on));
  const sec = $("rc-2nd-on").checked;
  for (const id of ["rc-inst2", "rc-param2", "rc-values2"]) $(id).disabled = !sec;
  $("rc-hint").textContent = blocked
    ? "Optimize runs its own parameter search — sweep unavailable here."
    : m === "dc"
      ? "DC + sweep = a fast vectorized DC sweep (the former “DC sweep” mode)."
      : "runs the current analysis once per value and overlays the plots";
}

function readRunCfg() {
  return { enabled: $("rc-enable").checked, second: $("rc-2nd-on").checked,
           colorMode: $("rc-colormode").value,
           inst: $("rc-inst").value, param: $("rc-param").value,
           values: $("rc-values").value, inst2: $("rc-inst2").value,
           param2: $("rc-param2").value, values2: $("rc-values2").value };
}
function persistRunCfg() {
  try { localStorage.setItem("photonflux_runcfg",
    JSON.stringify(readRunCfg())); } catch {}
}
function restoreRunCfg() {
  let c; try { c = JSON.parse(localStorage.getItem("photonflux_runcfg")); } catch {}
  refreshRunCfgSelectors();
  if (c) {
    $("rc-enable").checked = !!c.enabled;
    $("rc-2nd-on").checked = !!c.second;
    $("rc-colormode").value = c.colorMode || "shaded";
    if (c.inst) { $("rc-inst").value = c.inst; fillParamOptions("rc-inst", "rc-param"); }
    if (c.param) $("rc-param").value = c.param;
    $("rc-values").value = c.values || "";
    if (c.inst2) { $("rc-inst2").value = c.inst2; fillParamOptions("rc-inst2", "rc-param2"); }
    if (c.param2) $("rc-param2").value = c.param2;
    $("rc-values2").value = c.values2 || "";
  }
  syncRunCfgDisabled(); updateRunCount();
}

// restore the pane from a loaded analysis object (Save/Load JSON + examples),
// mapping the legacy dcsweep / ac-sweep payloads and the new run_config alike
function restorePaneFromAnalysis(a) {
  refreshRunCfgSelectors();
  let axes = null, colorMode = "shaded";
  const rc = a.run_config;
  if (rc && rc.sweep && rc.sweep.length) {
    axes = rc.sweep.map((ax) => ({ instance: ax.instance, param: ax.param,
      values: (ax.values || []).map(fmtSI).join(", ") }));
    colorMode = (rc.overlay || {}).color_mode || "shaded";
  } else if (a.mode === "dcsweep" && a.instance) {
    axes = [{ instance: a.instance, param: a.param,
      values: a.values ? a.values.map(fmtSI).join(", ")
        : `${fmtSI(a.start)}:${fmtSI(a.stop)}:${a.points || 101}` }];
    if (a.step_instance && (a.step_values || []).length)
      axes.push({ instance: a.step_instance, param: a.step_param,
        values: (a.step_values || []).map(fmtSI).join(", ") });
    colorMode = (a.overlay || {}).color_mode || "shaded";
  } else if (a.mode === "ac" && a.sweep_instance) {
    axes = [{ instance: a.sweep_instance, param: a.sweep_param,
      values: (a.sweep_values || []).map(fmtSI).join(", ") }];
  }
  const on = !!(axes && axes.length);
  $("rc-enable").checked = on;
  $("rc-colormode").value = colorMode;
  $("rc-2nd-on").checked = on && axes.length > 1;
  if (on) {
    $("rc-inst").value = axes[0].instance; fillParamOptions("rc-inst", "rc-param");
    $("rc-param").value = axes[0].param; $("rc-values").value = axes[0].values;
    if (axes.length > 1) {
      $("rc-inst2").value = axes[1].instance; fillParamOptions("rc-inst2", "rc-param2");
      $("rc-param2").value = axes[1].param; $("rc-values2").value = axes[1].values;
    } else { $("rc-values2").value = ""; }
  }
  syncRunCfgDisabled(); updateRunCount(); persistRunCfg();
  // opening a testbench that ships a sweep -> reveal the pane so it's not hidden
  setRunCfgPanel(on);
}

// translate the pane + current analysis type into the run payload
function withRunCfg(a) {
  if (!$("rc-enable").checked || $("rc-enable").disabled) return a;
  const axes = runCfgAxes();
  if (!axes.length) return a;                 // incomplete -> single run
  const colorMode = $("rc-colormode").value;
  if (a.mode === "dc") {
    // fast vectorized DC parameter sweep (the _run_dcsweep engine)
    const b = { ...a, mode: "dcsweep", instance: axes[0].instance,
                param: axes[0].param, values: axes[0].values,
                overlay: { color_mode: colorMode } };
    if (axes[1]) {
      b.step_instance = axes[1].instance;
      b.step_param = axes[1].param;
      b.step_values = axes[1].values.slice(0, 10);
    }
    return b;
  }
  if (a.mode === "ac" && axes.length === 1) {
    // single-axis AC keeps its own recompile+prewarm engine (sweep_values)
    a.sweep_instance = axes[0].instance;
    a.sweep_param = axes[0].param;
    a.sweep_values = axes[0].values.slice(0, 8);
    return a;
  }
  // transient / noise / pulse, and 2-axis AC -> generic per-point overlay
  a.run_config = {
    sweep: axes.map((ax) => ({ instance: ax.instance, param: ax.param,
                               values: ax.values })),
    overlay: { color_mode: colorMode },
  };
  return a;
}

// show/hide the sweep-parameters pane, keeping the caret button in sync
function setRunCfgPanel(open) {
  const p = $("runcfg-panel");
  p.hidden = !open;
  $("btn-runcfg").classList.toggle("active", open);
  $("btn-runcfg").setAttribute("aria-expanded", String(open));
  if (open) { refreshRunCfgSelectors(); syncRunCfgDisabled(); updateRunCount(); }
}
$("btn-runcfg").addEventListener("click", () => {
  setRunCfgPanel($("runcfg-panel").hidden);
});
$("rc-inst").addEventListener("change", () => {
  fillParamOptions("rc-inst", "rc-param"); updateRunCount(); persistRunCfg();
});
$("rc-inst2").addEventListener("change", () => {
  fillParamOptions("rc-inst2", "rc-param2"); updateRunCount(); persistRunCfg();
});
for (const id of ["rc-enable", "rc-2nd-on", "rc-colormode", "rc-param", "rc-param2"])
  $(id).addEventListener("change", () => {
    syncRunCfgDisabled(); updateRunCount(); persistRunCfg();
  });
for (const id of ["rc-values", "rc-values2"])
  $(id).addEventListener("input", () => { updateRunCount(); persistRunCfg(); });

// derived-trace expressions (edited in the fx panel, sent with every run)
function exprText() { return $("expr-text").value; }
$("btn-exprs").addEventListener("click", () => {
  const p = $("expr-panel");
  p.hidden = !p.hidden;
  $("btn-exprs").classList.toggle("active", !p.hidden);
  if (!p.hidden) $("expr-text").focus();
});
$("expr-text").addEventListener("change", () => {
  try { localStorage.setItem("photonflux_exprs", exprText()); } catch {}
});
try { $("expr-text").value = localStorage.getItem("photonflux_exprs") || ""; } catch {}

function withExprs(a) {
  if (exprText().trim()) a.expressions = exprText();
  return a;
}

function collectAnalysis() {
  return withRunCfg(withExprs(collectAnalysisBase()));
}

function collectAnalysisBase() {
  const m = analysisMode();
  if (m === "dc") return { mode: "dc" };   // + a run-config sweep -> "dcsweep"
  if (m === "ac") {
    return {
      mode: "ac",
      f_start: parseSI($("an-ac-fstart").value),
      f_stop: parseSI($("an-ac-fstop").value),
      points: parseSI($("an-ac-points").value) || 121,
      z0: parseSI($("an-ac-z0").value) || 50,
    };
  }
  if (m === "optimize") {
    const params = $("an-op-params").value.split(",")
      .map((s) => s.trim()).filter(Boolean)
      .map((tok) => {
        const mm = tok.match(/^(\w+)\.(\w+)\s*=\s*([^:]+):(.+)$/);
        return mm && { inst: mm[1], param: mm[2],
                       min: parseSI(mm[3].trim()), max: parseSI(mm[4].trim()) };
      }).filter((p) => p && !isNaN(p.min) && !isNaN(p.max));
    const obj = $("an-op-obj").value.trim();
    return {
      mode: "optimize",
      t_stop: parseSI($("an-op-tstop").value),
      points: parseSI($("an-op-points").value) || 4000,
      optimize: {
        params, objective: obj,
        maximize: !obj.startsWith("ber"),
        iters: parseInt($("an-op-iters").value) || 30,
      },
    };
  }
  if (m === "noise") {
    return {
      mode: "noise",
      f_start: parseSI($("an-nq-fstart").value),
      f_stop: parseSI($("an-nq-fstop").value),
      points: parseSI($("an-nq-points").value) || 121,
      probe: $("an-nq-probe").value,
    };
  }
  if (m === "pulse") {
    return {
      mode: "pulse",
      t_stop: parseSI($("an-pl-tstop").value),
      points: parseSI($("an-pl-points").value) || 2000,
      probe: $("an-pl-probe").value,
      ffe_taps: parseInt($("an-pl-ffe").value) || 0,
      dfe_taps: parseInt($("an-pl-dfe").value) || 0,
    };
  }
  const a = {
    mode: "transient",
    t_stop: parseSI($("an-tstop").value),
    points: parseSI($("an-points").value) || 800,
  };
  const solver = $("an-solver").value;
  if (solver) a.solver = solver;
  const dtmax = parseSI($("an-dtmax").value);
  if (!isNaN(dtmax) && dtmax > 0) a.dtmax = dtmax;
  const lp = $("an-link-probe").value;
  if (lp) {
    a.link = { probe: lp,
               ffe_taps: parseInt($("an-link-ffe").value) || 0,
               dfe_taps: parseInt($("an-link-dfe").value) || 0 };
  }
  const nzSeeds = parseInt($("an-nz-seeds").value);
  if (nzSeeds >= 1) {
    a.noise = { seeds: nzSeeds, bw: parseSI($("an-nz-bw").value) || 50e9 };
  }
  return a;
}

function applyAnalysis(a) {
  if (!a) return;
  $("expr-text").value = a.expressions || "";
  try { localStorage.setItem("photonflux_exprs", $("expr-text").value); } catch {}
  // "dcsweep" is no longer a dropdown mode — a DC sweep is DC + a pane sweep
  $("sel-analysis").value = a.mode === "dcsweep" ? "dc" : (a.mode || "transient");
  $("sel-analysis").dispatchEvent(new Event("change"));
  if (a.mode === "transient") {
    if (a.t_stop) $("an-tstop").value = fmtSI(a.t_stop);
    if (a.points) $("an-points").value = a.points;
    $("an-solver").value = a.solver || "";
    $("an-dtmax").value = a.dtmax ? fmtSI(a.dtmax) : "";
    updateProbeSelectors();
    $("an-link-probe").value = a.link?.probe || "";
    $("an-link-ffe").value = a.link?.ffe_taps ?? 0;
    $("an-link-dfe").value = a.link?.dfe_taps ?? 0;
    $("an-nz-seeds").value = a.noise?.seeds || "";
    $("an-nz-bw").value = a.noise?.bw ? fmtSI(a.noise.bw) : "50G";
  } else if (a.mode === "optimize") {
    if (a.t_stop) $("an-op-tstop").value = fmtSI(a.t_stop);
    if (a.points) $("an-op-points").value = a.points;
    const o = a.optimize || {};
    $("an-op-obj").value = o.objective || "";
    $("an-op-iters").value = o.iters || 30;
    $("an-op-params").value = (o.params || []).map((p) =>
      `${p.inst}.${p.param}=${fmtSI(p.min)}:${fmtSI(p.max)}`).join(", ");
  } else if (a.mode === "noise") {
    if (a.f_start) $("an-nq-fstart").value = fmtSI(a.f_start);
    if (a.f_stop) $("an-nq-fstop").value = fmtSI(a.f_stop);
    if (a.points) $("an-nq-points").value = a.points;
    updateProbeSelectors();
    if (a.probe) $("an-nq-probe").value = a.probe;
  } else if (a.mode === "pulse") {
    if (a.t_stop) $("an-pl-tstop").value = fmtSI(a.t_stop);
    if (a.points) $("an-pl-points").value = a.points;
    updateProbeSelectors();
    if (a.probe) $("an-pl-probe").value = a.probe;
    $("an-pl-ffe").value = a.ffe_taps ?? 5;
    $("an-pl-dfe").value = a.dfe_taps ?? 1;
  } else if (a.mode === "ac") {
    if (a.f_start) $("an-ac-fstart").value = fmtSI(a.f_start);
    if (a.f_stop) $("an-ac-fstop").value = fmtSI(a.f_stop);
    if (a.points) $("an-ac-points").value = a.points;
    if (a.z0) $("an-ac-z0").value = a.z0;
  }
  // sweeps (legacy dcsweep / ac-sweep and the new run_config) live in the pane
  restorePaneFromAnalysis(a);
}

// ---------------------------------------------------------------------------
// run + results
// ---------------------------------------------------------------------------
// Poll GET /api/progress while a run is in flight and drive the header bar.
// The backend fraction covers the transient solve (spanning noise seeds and
// sweep points); compile time reports no fraction, so we hold the bar in an
// indeterminate sweep until the first real step arrives.
let progressGen = 0;
function startProgressPoll() {
  const bar = $("run-progress"), fill = $("run-progress-fill");
  const label = $("run-progress-label");
  if (!bar || !fill) return { stop() {} };
  // Each poll owns a generation; a stale hide-timeout from a previous run must
  // not hide the bar a newer run has just shown (rapid re-run via Cmd+Enter).
  const gen = ++progressGen;
  let stopped = false, seenFrac = false;
  bar.hidden = false;
  bar.classList.add("indeterminate");
  fill.style.width = "";
  // Text beside the bar: a short phase word while the run isn't yet stepping
  // (compile dominates the first run and reports no fraction), then a live
  // percentage once the transient solver starts advancing through sim time.
  if (label) { label.hidden = false; label.textContent = "Compiling…"; }
  const timer = setInterval(async () => {
    let p;
    try { p = await (await fetch("/api/progress")).json(); }
    catch { return; }
    if (stopped || !p || !p.active) return;
    if (p.frac > 0 || p.phase === "solving") {
      seenFrac = true;
      bar.classList.remove("indeterminate");
      const pct = Math.min(100, Math.max(0, p.frac * 100));
      fill.style.width = `${pct.toFixed(1)}%`;
      if (label) label.textContent = `${pct.toFixed(0)}%`;
    } else if (label) {
      label.textContent = "Compiling…";
    }
  }, 200);
  return {
    stop() {
      stopped = true;
      clearInterval(timer);
      // A finished solve briefly shows a full bar before it disappears.
      if (seenFrac) { fill.style.width = "100%"; if (label) label.textContent = "100%"; }
      setTimeout(() => {
        if (gen !== progressGen) return;  // a newer run now owns the bar
        bar.hidden = true;
        bar.classList.remove("indeterminate");
        fill.style.width = "0";
        if (label) { label.hidden = true; label.textContent = ""; }
      }, seenFrac ? 220 : 0);
    },
  };
}

// Holds the in-flight run's AbortController so the Stop button can abort the
// fetch. Null when no run is active.
let runAbort = null;

// Stop the current run: ask the backend to abort the solve cooperatively
// (POST /api/cancel — the transient solver checks this at its next step) and
// abort the fetch so the UI is freed immediately even if a DC/AC solve, which
// has no interruptible step loop, keeps running to completion in the background.
function stopSim() {
  if (!runAbort) return;
  const status = $("run-status");
  status.textContent = "stopping…";
  fetch("/api/cancel", { method: "POST" }).catch(() => {});
  runAbort.abort();
}

async function runSim() {
  const btn = $("btn-run"), stopBtn = $("btn-stop"), status = $("run-status");
  const payload = {
    schematic: {
      // PRBS sources pull their unit interval from the global baud rate: inject
      // UI = 1/baud here so the backend waveform builder, eye and BER post-proc
      // all see one consistent rate.
      instances: Object.fromEntries(Object.entries(state.instances).map(
        ([id, i]) => [id, { type: i.type, settings: i.type === "prbs"
          ? { ...(i.settings || {}), ui: globalUI() } : (i.settings || {}) }])),
      wires: state.wires.map((w) => [w.from, w.to]),
      probes: state.probes.map((p) => ({ name: p.name, at: p.at,
        ...(p.spectrum ? { spectrum: true } : {}) })),
    },
    analysis: collectAnalysis(),
  };
  // Swap Run for Stop while the solve is in flight so the user can abort it.
  btn.hidden = true;
  stopBtn.hidden = false;
  status.className = "";
  runAbort = new AbortController();
  const t0 = performance.now();
  // AC sweeps over a SKY130 w_um/l_um recompile per value and may extract a
  // BSIM4 card the first time (tens of seconds each) — set expectations so a
  // slow first run doesn't look hung.
  const a = payload.analysis;
  const slowSky = a.mode === "ac" && a.sweep_values &&
    /^sky130_/.test(state.instances[a.sweep_instance]?.type || "") &&
    /_um$/.test(a.sweep_param || "");
  // number of overlaid runs (generic run_config fan-out), for the ticker
  const nRuns = a.run_config?.sweep
    ? a.run_config.sweep.reduce((k, ax) => k * (ax.values?.length || 1), 1)
    : 0;
  const tick = setInterval(() => {
    const s = ((performance.now() - t0) / 1000).toFixed(0);
    status.textContent = slowSky
      ? `running… ${s}s (first run extracts SKY130 cards, ~1 min)`
      : nRuns > 1 ? `sweeping ${nRuns} runs… ${s}s`
      : `running… ${s}s`;
  }, 250);
  status.textContent = "running…";
  // Live progress bar: poll the backend, which reports where the transient
  // solver is in simulated time (see webapp/progress.py). It sits indeterminate
  // during the compile phase (no fraction yet) and fills once stepping begins.
  // Only time-domain runs report a fraction, so skip the bar for DC/AC/op —
  // their existing elapsed-time ticker is enough and avoids a bar that can
  // only ever sit indeterminate. Swept transients keep mode "transient", so
  // this covers them too, while excluding e.g. a 2-axis AC sweep.
  const timeDomain = a.mode === "transient" || a.mode === "pulse";
  const progPoll = timeDomain ? startProgressPoll() : { stop() {} };
  let res, aborted = false;
  try {
    const resp = await fetch("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload), signal: runAbort.signal,
    });
    res = await resp.json();
  } catch (e) {
    if (e && e.name === "AbortError") { aborted = true; res = { ok: false, cancelled: true }; }
    else res = { ok: false, error: `server unreachable: ${e}` };
  }
  clearInterval(tick);
  progPoll.stop();
  runAbort = null;
  btn.hidden = false;
  stopBtn.hidden = true;
  const dt = ((performance.now() - t0) / 1000).toFixed(1);
  // Run stopped by the user (fetch aborted, or the backend returned a
  // cooperatively-cancelled result): show a neutral "stopped" state, not a
  // red failure, and leave the previous results untouched.
  if (aborted || res.cancelled) {
    status.textContent = `stopped (${dt}s)`;
    status.className = "";
    return;
  }
  lastResult = res;
  showLog(res);
  if (!res.ok) {
    status.textContent = `failed (${dt}s)`;
    status.className = "err";
    setResultsTab("log");
    $("results").classList.remove("collapsed");
    return;
  }
  status.textContent = `done in ${dt}s`;
  status.className = "ok";
  $("results").classList.remove("collapsed");
  if (res.kind === "op") { renderOpTable(res); setResultsTab("op"); }
  else if (res.pulse || res.optim) {
    renderPlots(res); renderLink(); setResultsTab("link");
  }
  else {
    renderPlots(res);
    if (res.link) renderLink();
    setResultsTab("plots");
  }
}
$("btn-run").addEventListener("click", runSim);
$("btn-stop").addEventListener("click", stopSim);

function probeColor(name) {
  const p = state.probes.find((q) => q.name === name);
  return p ? p.color : "#d6d6e3";
}

/* shade a probe color across a stepped family: low step -> dim, high -> bright */
function stepShade(hex, frac) {
  const r = parseInt(hex.slice(1, 3), 16) / 255,
        g = parseInt(hex.slice(3, 5), 16) / 255,
        b = parseInt(hex.slice(5, 7), 16) / 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
  let h = 0;
  const l = (mx + mn) / 2, d = mx - mn;
  const s = d === 0 ? 0 : d / (1 - Math.abs(2 * l - 1));
  if (d !== 0) {
    if (mx === r) h = ((g - b) / d) % 6;
    else if (mx === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h *= 60; if (h < 0) h += 360;
  }
  return `hsl(${h.toFixed(0)} ${(s * 100).toFixed(0)}% ${(34 + 41 * frac).toFixed(0)}%)`;
}

function traceStroke(tr) {
  if (tr.color) return tr.color;   // server-assigned (derived traces)
  const base = probeColor(tr.probe || tr.name);
  return tr.step_frac !== undefined ? stepShade(base, tr.step_frac) : base;
}

// a probe marked "hide" keeps its trace out of the results view (still sent to
// the backend so AC pairing / op-point math is unaffected). A trace maps to
// its probe via `probe` (stepped families, AC) or `name` (plain sweep/tran).
function probeHidden(probeName) {
  const p = state.probes.find((q) => q.name === probeName);
  return !!(p && p.hide);
}
const visibleTraces = (traces) =>
  (traces || []).filter((tr) => !probeHidden(tr.probe || tr.name));

// re-render whatever the results panel currently shows, from the cached run
function rerenderResults() {
  if (!lastResult?.ok) return;
  if (lastResult.kind === "op") renderOpTable(lastResult);
  else if (lastResult.traces) renderPlots(lastResult);
  if (document.querySelector("#tab-eye.active")) renderEye();
}

// per-unit-group plot preferences (y/x scale), persisted across sessions
let plotPrefs = {};
try { plotPrefs = JSON.parse(localStorage.getItem("photonflux_plotprefs")) || {}; } catch {}

function plotPref(unit) {
  return Object.assign({ y: "linear", x: "linear" }, plotPrefs[unit] || {});
}

function setPlotPref(unit, key, val) {
  plotPrefs[unit] = Object.assign(plotPref(unit), { [key]: val });
  try { localStorage.setItem("photonflux_plotprefs", JSON.stringify(plotPrefs)); } catch {}
  if (lastResult?.ok && lastResult.traces) renderPlots(lastResult);
}

function renderPlots(res) {
  const holder = $("tab-plots");
  plots.forEach((p) => p.destroy());
  plots = [];
  holder.innerHTML = "";
  const shown = visibleTraces(res.traces);
  const groups = {};
  for (const tr of shown) (groups[tr.unit] = groups[tr.unit] || []).push(tr);
  if (!shown.length) {
    holder.innerHTML = `<div class="insp-empty" style="padding:16px">
      All probes are hidden — un-hide a probe (select it, uncheck
      &ldquo;Hide from plots&rdquo;) to see its trace.</div>`;
    return;
  }

  const isTime = res.kind === "transient";
  const xs = isTime ? res.x.map((t) => t * 1e9) : res.x;
  const xLabel = isTime ? "time [ns]" : (res.xlabel || "x");
  const xLogOk = xs.length && xs.every((v) => v > 0);

  const axisStyle = {
    stroke: "#8a8aa3",
    grid: { stroke: "#33334a", width: 1 },
    ticks: { stroke: "#33334a" },
    font: "11px -apple-system, sans-serif",
  };
  // width divisor counts every box we're about to lay out — the main-plot
  // unit groups AND the extra plots (optical spectra, spec()/psd()) — so a
  // time-domain + spectrum pair sits two-across instead of stacking full-width
  const nExtra = (res.extra_plots || [])
    .filter((ep) => (ep.traces || []).length).length;
  const nBoxes = Object.keys(groups).length + nExtra;
  const wrapW = Math.max(420,
    Math.min(760, ($("results-body").clientWidth - 40) /
      Math.min(2, nBoxes) - 20));
  const plotH = Math.max(150, $("results").clientHeight - 150);

  // SI-formatted ticks: 0.0001 -> "100u". On log axes uPlot splits every
  // mantissa 1..9 per decade; label only the 1/2/5 ones so they resolve.
  const siTicks = (u, splits) => splits.map((v) =>
    v == null ? "" : v === 0 ? "0" : fmtSI(v));
  const logTicks = (u, splits) => splits.map((v) => {
    if (v == null || v === 0) return "";
    const a = Math.abs(v);
    const m = Math.round(a / 10 ** Math.floor(Math.log10(a) + 1e-12));
    return (m === 1 || m === 2 || m === 5 || m === 10) ? fmtSI(v) : "";
  });
  const nmTicks = (u, splits) => splits.map((v) =>
    v == null ? "" : v.toFixed(3).replace(/\.?0+$/, ""));

  for (const [unit, traces] of Object.entries(groups)) {
    const pref = plotPref(unit);
    // AC sweeps default to a log frequency axis unless the user chose one
    if (res.xlog && xLogOk && !(plotPrefs[unit] && plotPrefs[unit].x)) pref.x = "log";
    if (pref.x === "log" && !xLogOk) pref.x = "linear";

    // dBm only makes sense for the optical-power (mW) group
    const yModes = unit === "mW" ? ["linear", "log", "dBm"] : ["linear", "log"];
    // noise densities span decades: default to a log y axis
    if (unit === "V/rtHz" && !(plotPrefs[unit] && plotPrefs[unit].y)) pref.y = "log";
    if (!yModes.includes(pref.y)) pref.y = "linear";
    const yLabel = pref.y === "dBm" ? "dBm" : unit;
    const yLog = pref.y === "log";
    const mapY = (v) => {
      if (v == null) return null;
      if (pref.y === "dBm") return v > 0 ? 10 * Math.log10(v) : null;
      if (yLog) return v > 0 ? v : null;  // log axis: hide non-positive points
      return v;
    };

    const box = document.createElement("div");
    box.className = "plot-box";
    const head = document.createElement("div");
    head.className = "plot-head";
    const titleTxt = unit === "mW"
      ? `Optical power [${yLabel}]` : `Signal [${yLabel}]`;
    head.innerHTML = `<span class="plot-title">${titleTxt}</span>
      <span class="spacer"></span>
      <label>y <select data-k="y">
        ${yModes.map((m) => `<option ${m === pref.y ? "selected" : ""}>${m}</option>`).join("")}
      </select></label>
      ${xLogOk ? `<label>x <select data-k="x">
        <option ${pref.x === "linear" ? "selected" : ""}>linear</option>
        <option ${pref.x === "log" ? "selected" : ""}>log</option>
      </select></label>` : ""}
      <button class="plot-reset" title="Reset zoom to full range (or double-click the plot)">&#9635; reset</button>`;
    head.querySelectorAll("select").forEach((sel) =>
      sel.addEventListener("change", () => setPlotPref(unit, sel.dataset.k, sel.value)));
    box.appendChild(head);
    holder.appendChild(box);

    const yTicks = yLog ? logTicks
      : pref.y === "dBm" ? undefined   // dB values: plain numbers read best
      : siTicks;

    const opts = {
      width: wrapW, height: plotH,
      scales: {
        x: { time: false, ...(pref.x === "log" ? { distr: 3 } : {}) },
        y: yLog ? { distr: 3 } : {},
      },
      axes: [
        { ...axisStyle, label: xLabel, labelFont: "11px sans-serif", labelSize: 18,
          ...(pref.x === "log" ? { values: logTicks } : {}) },
        { ...axisStyle, label: yLabel, labelFont: "11px sans-serif", labelSize: 18,
          size: 60, ...(yTicks ? { values: yTicks } : {}) },
      ],
      series: [
        { label: xLabel.split(" ")[0],
          value: (u, v) => v == null ? "" : v.toPrecision(5) },
        ...traces.map((tr) => ({
          label: tr.name, stroke: traceStroke(tr), width: 1.6,
          value: (u, v) => v == null ? "" : v.toPrecision(5),
        })),
      ],
      cursor: { drag: { x: true, y: false },
                sync: { key: "photonflux", setSeries: false } },
      legend: { live: true },
    };
    // AC dB plots: dashed 0 dB reference — |h21| crossing it marks f_T
    const withZeroRef = res.kind === "ac" && unit === "dB" && pref.y === "linear";
    if (withZeroRef) {
      opts.series.push({ label: "0 dB (fT crossing)", stroke: "#8a8aa3",
                         width: 1, dash: [6, 5],
                         value: () => "" });
    }
    const data = [xs, ...traces.map((t) => t.values.map(mapY))];
    if (withZeroRef) data.push(xs.map(() => 0));
    const u = new uPlot(opts, data, box);
    // drag zooms x only (drag.y is off), so a full reset just restores the
    // x scale to the data extent; y stays auto-ranged throughout.
    head.querySelector(".plot-reset").addEventListener("click", () => {
      const xd = u.data[0];
      if (xd && xd.length) {
        const a = xd[0], b = xd[xd.length - 1];
        u.setScale("x", { min: Math.min(a, b), max: Math.max(a, b) });
      }
    });
    plots.push(u);
  }

  // extra plots with their own x axis (e.g. spec()/psd() expression results)
  for (const ep of res.extra_plots || []) {
    const epTraces = ep.traces || [];
    if (!epTraces.length) continue;
    const box = document.createElement("div");
    box.className = "plot-box";
    const head = document.createElement("div");
    head.className = "plot-head";
    head.innerHTML = `<span class="plot-title">
        ${epTraces.map((t) => t.name).join(", ")} [${epTraces[0].unit || "-"}]
      </span><span class="spacer"></span>
      <button class="plot-reset" title="Reset zoom">&#9635; reset</button>`;
    box.appendChild(head);
    holder.appendChild(box);
    const xlog = !!ep.xlog && ep.x.every((v) => v > 0);
    const ydb = !!ep.ydb;                    // linear dB axis (negatives OK)
    const xvals = ep.xunit === "nm" ? nmTicks : (xlog ? logTicks : siTicks);
    const opts = {
      width: wrapW, height: plotH,
      scales: { x: { time: false, ...(xlog ? { distr: 3 } : {}) },
                y: ydb ? {} : { distr: 3 } },
      axes: [
        { ...axisStyle, label: ep.xlabel || "x", labelFont: "11px sans-serif",
          labelSize: 18, values: xvals },
        { ...axisStyle, label: ep.yunit || epTraces[0].unit || "", size: 60,
          labelFont: "11px sans-serif", labelSize: 18,
          ...(ydb ? {} : { values: logTicks }) },
      ],
      series: [
        { label: ep.xunit === "nm" ? "λ" : "f",
          value: (u2, v) => v == null ? "" : v.toPrecision(6) },
        ...epTraces.map((tr) => ({
          label: tr.name, stroke: traceStroke(tr), width: 1.4,
          value: (u2, v) => v == null ? "" : v.toPrecision(4),
        })),
      ],
      cursor: { drag: { x: true, y: false } },
      legend: { live: true },
    };
    const u = new uPlot(opts,
      [ep.x, ...epTraces.map((t) => ydb ? t.values
        : t.values.map((v) => v > 0 ? v : null))],
      box);
    head.querySelector(".plot-reset").addEventListener("click", () => {
      const xd = u.data[0];
      if (xd && xd.length) {
        u.setScale("x", { min: xd[0], max: xd[xd.length - 1] });
      }
    });
    plots.push(u);
  }
}

// --- results panel resize (drag the strip above the tab bar) ---------------
(() => {
  const rz = $("results-resizer"), panel = $("results");
  const saved = parseInt(localStorage.getItem("photonflux_results_h") || "", 10);
  if (saved >= 140) panel.style.height = saved + "px";
  let drag = null;
  rz.addEventListener("mousedown", (ev) => {
    ev.preventDefault();
    drag = { y0: ev.clientY, h0: panel.clientHeight };
    rz.classList.add("active");
  });
  window.addEventListener("mousemove", (ev) => {
    if (!drag) return;
    const h = Math.min(window.innerHeight - 160,
      Math.max(140, drag.h0 + (drag.y0 - ev.clientY)));
    panel.style.height = h + "px";
    const plotH = Math.max(150, h - 150);
    plots.forEach((p) => p.setSize({ width: p.width, height: plotH }));
  });
  window.addEventListener("mouseup", () => {
    if (!drag) return;
    drag = null;
    rz.classList.remove("active");
    try { localStorage.setItem("photonflux_results_h", String(panel.clientHeight)); } catch {}
  });
})();

window.addEventListener("resize", () => {
  if (lastResult?.ok && lastResult.traces &&
      document.querySelector("#tab-plots.active")) renderPlots(lastResult);
});

function renderOpTable(res) {
  const rows = res.rows.filter((r) => !probeHidden(r.name)).map((r) => `
    <tr><td style="color:${probeColor(r.name)}">&#9873; ${r.name}</td>
        <td>${r.domain}</td>
        <td class="num">${(+r.value).toPrecision(6)} ${r.unit}</td>
        <td>${r.extra || ""}</td></tr>`).join("");
  $("tab-op").innerHTML = `<table>
    <tr><th>Probe</th><th>Domain</th><th>Value</th><th></th></tr>${rows}</table>`;
}

function showLog(res) {
  const lines = [];
  if (res.log) lines.push(...res.log);
  if (res.error) lines.push(`<span class="err">ERROR: ${escapeHtml(res.error)}</span>`);
  if (res.traceback) lines.push(...res.traceback.map(escapeHtml));
  $("log-pre").innerHTML = lines.join("\n") || "(no log)";
}

const escapeHtml = (s) => s.replace(/[&<>]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

function setResultsTab(tab) {
  document.querySelectorAll(".rtab").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll(".rpane").forEach((p) =>
    p.classList.toggle("active", p.id === "tab-" + tab));
  if (tab === "eye") renderEye();
  if (tab === "link") renderLink();
}

// ---------------------------------------------------------------------------
// eye diagram: fold the transient record at the UI, density-render + metrics
// ---------------------------------------------------------------------------
function eyeDefaults() {
  // UI folds at the global baud rate; modulation levels come from a pattern
  // source on the canvas, if any
  let mod = 2;
  for (const inst of Object.values(state.instances)) {
    if (inst.type === "prbs") {
      mod = (inst.settings || {}).mode === "pam4" ? 4 : 2;
      break;
    }
  }
  return { ui: globalUI(), mod };
}

function eyeTraceOptions() {
  if (!(lastResult?.ok && lastResult.kind === "transient")) return [];
  // multi-seed noise families ("vout#0"..) fold together under their probe
  return [...new Set(visibleTraces(lastResult.traces)
    .map((tr) => tr.probe || tr.name))];
}

let eyeInit = false;
function renderEye() {
  const hint = $("eye-hint"), canvas = $("eye-canvas"),
        ctrls = $("eye-controls"), sel = $("eye-trace"),
        swSel = $("eye-sweep"), swWrap = $("eye-sweep-wrap");
  const names = eyeTraceOptions();
  const usable = names.length > 0;
  hint.hidden = usable;
  canvas.style.display = usable ? "" : "none";
  ctrls.style.display = usable ? "" : "none";
  if (!usable) {
    hint.textContent = "Run a transient analysis with at least one visible "
      + "probe, then fold it into an eye here.";
    return;
  }
  if (!eyeInit) {
    eyeInit = true;
    const d = eyeDefaults();
    $("eye-ui").value = fmtSI(d.ui);
    $("eye-mod").value = "auto";
    ["eye-ui", "eye-skip"].forEach((id) =>
      $(id).addEventListener("change", renderEye));
    $("eye-mod").addEventListener("change", renderEye);
    sel.addEventListener("change", renderEye);
    swSel.addEventListener("change", renderEye);
  }
  const prev = sel.value;
  sel.innerHTML = names.map((n) => `<option>${n}</option>`).join("");
  if (names.includes(prev)) sel.value = prev;

  const fam = visibleTraces(lastResult.traces)
    .filter((q) => (q.probe || q.name) === sel.value);
  if (!fam.length) return;

  // parameter sweep: overlay every swept eye, or isolate a single value.
  // Merged sweep traces are named "<probe> @ <label>" (see _merge_run); the
  // shared `probe` otherwise pools them all together, so group by that label.
  const labelOf = (q) => {
    const nm = q.name || "";
    const i = nm.indexOf(" @ ");
    return i >= 0 ? nm.slice(i + 3) : null;
  };
  const labels = [...new Set(fam.map(labelOf).filter(Boolean))];
  const isSweep = !!lastResult.sweep_overlay && labels.length > 1;
  swWrap.hidden = !isSweep;
  if (isSweep) {
    const prevSw = swSel.value;
    swSel.innerHTML = '<option value="__all__">overlay all</option>'
      + labels.map((l) => `<option value="${l}">${l}</option>`).join("");
    swSel.value = (prevSw === "__all__" || labels.includes(prevSw))
      ? prevSw : "__all__";
  }
  const shown = (isSweep && swSel.value !== "__all__")
    ? fam.filter((q) => labelOf(q) === swSel.value)
    : fam;
  if (!shown.length) return;
  const tr = shown[0];
  const t = lastResult.x;
  const ui = parseSI($("eye-ui").value);
  if (!(ui > 0) || t.length < 8) return;
  const tEnd = t[t.length - 1];
  const skipRaw = $("eye-skip").value.trim();
  const tSkip = skipRaw.endsWith("%")
    ? tEnd * parseFloat(skipRaw) / 100 : (parseSI(skipRaw) || 0);
  const nlv = $("eye-mod").value === "auto"
    ? eyeDefaults().mod : parseInt($("eye-mod").value);

  // resample each family record onto a uniform grid
  const osr = 64, dt = ui / osr;
  const n = Math.floor((tEnd - tSkip) / dt);
  if (n < 4 * osr) {
    $("eye-metrics").textContent = "record too short for this UI";
    return;
  }
  const records = shown.map((q) => {
    const v = q.values;
    const ys = new Float64Array(n);
    let j = 0;
    for (let i = 0; i < n; i++) {
      const tt = tSkip + i * dt;
      while (j < t.length - 2 && t[j + 1] < tt) j++;
      const f = (tt - t[j]) / (t[j + 1] - t[j] || 1);
      ys[i] = v[j] + (v[j + 1] - v[j]) * Math.min(Math.max(f, 0), 1);
    }
    return ys;
  });
  let lo = Infinity, hi = -Infinity;
  for (const ys of records) {
    for (const y of ys) { if (y < lo) lo = y; if (y > hi) hi = y; }
  }
  if (!(hi > lo)) { $("eye-metrics").textContent = "flat trace"; return; }
  const pad = 0.08 * (hi - lo);
  lo -= pad; hi += pad;

  // density render: overlay 2-UI segments with low alpha
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#121218";
  ctx.fillRect(0, 0, W, H);
  const span = 2 * osr;                       // 2 UI across the canvas
  const px = (k) => (k / span) * W;
  const py = (y) => H - ((y - lo) / (hi - lo)) * H;
  const alpha = Math.max(0.03, 0.10 / Math.sqrt(records.length));
  // overlaying several swept values: tint each eye with its plot colour so the
  // families stay distinguishable; otherwise use the flat per-domain hue.
  const perSweep = isSweep && swSel.value === "__all__";
  const domHue = tr.domain === "optical"
    ? "rgb(255,183,77)" : "rgb(110,203,245)";
  ctx.lineWidth = 1;
  ctx.globalAlpha = alpha;
  records.forEach((ys, ri) => {
    ctx.strokeStyle = perSweep ? traceStroke(shown[ri]) : domHue;
    for (let i = 0; i + 1 < n; i++) {
      const k = i % osr;                      // phase within one UI
      for (const rep of [0, osr]) {           // draw into both UI windows
        ctx.beginPath();
        ctx.moveTo(px(k + rep), py(ys[i]));
        ctx.lineTo(px(k + rep + 1), py(ys[i + 1]));
        ctx.stroke();
      }
    }
  });
  ctx.globalAlpha = 1;
  // UI gridlines + centre markers
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.setLineDash([4, 5]);
  for (const k of [0, osr, 2 * osr]) {
    ctx.beginPath(); ctx.moveTo(px(k), 0); ctx.lineTo(px(k), H); ctx.stroke();
  }
  ctx.setLineDash([]);

  // metrics from samples near the decision phase of each folded UI
  const centre = [];
  for (const ys of records) {
    for (let i = 0; i + 1 < n; i++) {
      const ph = i % osr;
      if (Math.abs(ph - osr / 2) <= osr * 0.06) centre.push(ys[i]);
    }
  }
  const km = kmeans1d(centre, nlv);
  const eyes = [];
  for (let e = 0; e + 1 < km.levels.length; e++) {
    const a = km.levels[e], b = km.levels[e + 1];
    // worst-case inner opening between adjacent clusters
    let top = -Infinity, botm = Infinity;
    for (let i = 0; i < centre.length; i++) {
      const y = centre[i];
      if (km.assign[i] === e && y > top) top = y;
      if (km.assign[i] === e + 1 && y < botm) botm = y;
    }
    eyes.push(Math.max(0, botm - top));
  }
  // eye width: longest run of phases where no trajectory comes near a
  // decision threshold (guard band = 5% of the smallest level separation)
  const thrs = [];
  let minSep = Infinity;
  for (let e = 0; e + 1 < km.levels.length; e++) {
    thrs.push(0.5 * (km.levels[e] + km.levels[e + 1]));
    minSep = Math.min(minSep, km.levels[e + 1] - km.levels[e]);
  }
  const guard = 0.05 * minSep;
  const phDist = new Array(osr).fill(Infinity);
  for (const ys of records) {
    for (let i = 0; i < n; i++) {
      const ph = i % osr, y = ys[i];
      for (const thr of thrs) {
        const d = Math.abs(y - thr);
        if (d < phDist[ph]) phDist[ph] = d;
      }
    }
  }
  let run = 0, best = 0;
  for (let k = 0; k < 2 * osr; k++) {          // wrap once around
    if (phDist[k % osr] > guard) { run++; if (run > best) best = run; }
    else run = 0;
  }
  const widthUI = Math.min(best, osr) / osr;

  const unit = tr.unit || "";
  // note whether these metrics pool several swept eyes or describe just one
  const swNote = isSweep
    ? (perSweep ? `${records.length} eyes overlaid  |  ` : `${swSel.value}  |  `)
    : "";
  $("eye-metrics").textContent = swNote +
    `levels: ${km.levels.map((x) => fmtSI(x)).join(" / ")} ${unit}` +
    `  |  eye height: ${eyes.map((h) => fmtSI(h)).join(" / ")} ${unit}` +
    `  |  width: ${(widthUI * 100).toFixed(0)}% UI` +
    `  |  ${centre.length} samples`;

  // draw level lines
  ctx.strokeStyle = "rgba(120,255,160,0.35)";
  ctx.setLineDash([2, 6]);
  for (const l of km.levels) {
    ctx.beginPath(); ctx.moveTo(0, py(l)); ctx.lineTo(W, py(l)); ctx.stroke();
  }
  ctx.setLineDash([]);
}

// ---------------------------------------------------------------------------
// link report tab: BER/Q metrics + bathtub (transient) or pulse/COM (pulse)
// ---------------------------------------------------------------------------
function renderLink() {
  const holder = $("tab-link");
  holder.innerHTML = "";
  const opt = lastResult?.optim;
  if (opt) {
    const rows = opt.best.map((b, i) =>
      `<tr><th>${b.inst}.${b.param}</th><td>${fmtSI(b.value)}</td>
       <td class="num">${isFinite(opt.sens[i]) ? opt.sens[i].toExponential(2) : "—"}</td></tr>`)
      .join("");
    const div = document.createElement("div");
    div.className = "link-grid";
    div.innerHTML = `
      <table id="link-table">
        <tr><th>objective</th><td colspan="2">${opt.objective}
          (${opt.maximize ? "maximized" : "minimized"})</td></tr>
        <tr><th>best value</th><td colspan="2"><b>${fmtSI(opt.best_obj)}</b>
          after ${opt.evals} runs</td></tr>
        <tr><th></th><td><b>optimum</b></td><td><b>d(obj)/d(param)</b></td></tr>
        ${rows}
        <tr><td colspan="3"><button id="btn-opt-apply">Apply to schematic</button></td></tr>
      </table>`;
    holder.appendChild(div);
    div.querySelector("#btn-opt-apply").addEventListener("click", () => {
      commit(() => {
        for (const b of opt.best) {
          const inst = state.instances[b.inst];
          if (inst) { inst.settings = inst.settings || {}; inst.settings[b.param] = b.value; }
        }
      });
      renderInspector();
      alert("optimum applied to the schematic");
    });
    // fall through: also show link/pulse cards from the final run, if any
  }
  const rep = lastResult?.link, pul = lastResult?.pulse;
  if (opt && !rep && !pul) return;
  if (!rep && !pul) {
    holder.innerHTML = `<div class="insp-empty" style="padding:16px">
      No link report. Transient: pick a received probe in the toolbar's
      &ldquo;BER vs&rdquo; select (needs a PRBS source). Or run the
      Pulse&nbsp;/&nbsp;COM analysis.</div>`;
    return;
  }
  const fmtTaps = (taps) => taps.map((v) => v.toFixed(3)).join(", ") || "—";
  if (rep) {
    const c = rep.counted, q = rep.qfit;
    const berTxt = c.bit_errors === 0
      ? `0 / ${c.bits} bits (< ${(1 / c.bits).toExponential(1)})`
      : `${c.bit_errors} / ${c.bits} bits = ${c.ber.toExponential(2)}`;
    holder.insertAdjacentHTML("beforeend", `
      <div class="link-grid">
        <table id="link-table">
          <tr><th>received probe</th><td>${rep.probe} (vs ${rep.pattern},
            ${rep.nlv === 4 ? "PAM4" : "NRZ"}, UI ${fmtSI(rep.ui)}s)</td></tr>
          <tr><th>sampling phase</th><td>${(rep.sampling_phase_ui * 100).toFixed(0)}%
            UI, lag ${rep.lag_ui} UI, corr ${rep.corr.toFixed(3)}</td></tr>
          <tr><th>counted BER</th><td>${berTxt}</td></tr>
          <tr><th>counted SER</th><td>${c.sym_errors} / ${c.symbols} symbols</td></tr>
          ${q.ok ? `
          <tr><th>Q (per eye)</th><td>${q.q.map((x) => x.toFixed(2)).join(" / ")}</td></tr>
          <tr><th>Q-fit BER</th><td>${q.ber_est.toExponential(2)}</td></tr>
          <tr><th>levels</th><td>${q.levels.map((x) => fmtSI(x)).join(" / ")}</td></tr>`
          : `<tr><th>Q fit</th><td>${q.reason || "failed"}</td></tr>`}
          <tr><th>RX FFE taps</th><td>${fmtTaps(rep.ffe_taps)}</td></tr>
          <tr><th>RX DFE taps</th><td>${fmtTaps(rep.dfe_taps)}</td></tr>
        </table>
        <div id="link-plot"></div>
      </div>`);
    const tub = rep.bathtub;
    if (tub && tub.phase_ui.length > 4) {
      const opts = {
        width: 460, height: Math.max(170, $("results").clientHeight - 170),
        scales: { x: { time: false } },
        axes: [
          { stroke: "#8a8aa3", grid: { stroke: "#33334a", width: 1 },
            ticks: { stroke: "#33334a" }, label: "sampling phase [UI]",
            labelFont: "11px sans-serif", labelSize: 18, font: "11px sans-serif" },
          { stroke: "#8a8aa3", grid: { stroke: "#33334a", width: 1 },
            ticks: { stroke: "#33334a" }, label: "log10(BER)", size: 46,
            labelFont: "11px sans-serif", labelSize: 18, font: "11px sans-serif" },
        ],
        series: [
          { label: "phase" },
          { label: "bathtub", stroke: "#f08fb0", width: 1.6,
            value: (u, v) => v == null ? "" : v.toFixed(2) },
        ],
        cursor: { drag: { x: true, y: false } },
      };
      plots.push(new uPlot(opts, [tub.phase_ui, tub.log10_ber],
                           holder.querySelector("#link-plot")));
    }
    return;
  }
  // pulse / COM report
  holder.insertAdjacentHTML("beforeend", `
    <div class="link-grid">
      <table id="link-table">
        <tr><th>probe</th><td>${pul.probe} (UI ${fmtSI(pul.ui)}s)</td></tr>
        <tr><th>raw cursor</th><td>${fmtSI(pul.h_raw_peak)} at symbol
          ${pul.cursor}</td></tr>
        <tr><th>COM-style FOM</th><td><b>${pul.fom_db.toFixed(1)} dB</b>
          (cursor / &sigma;<sub>ISI</sub>)</td></tr>
        <tr><th>&sigma;_ISI (residual)</th><td>${pul.sigma_isi.toExponential(3)}</td></tr>
        <tr><th>Wiener FFE taps</th><td>${fmtTaps(pul.ffe_taps)}</td></tr>
        <tr><th>DFE taps</th><td>${fmtTaps(pul.dfe_taps)}</td></tr>
      </table>
      <div id="link-plot"></div>
    </div>`);
  const n = pul.h.length;
  const idx = Array.from({ length: n }, (_, i) => i - pul.cursor);
  const eqAligned = new Array(n).fill(null);
  for (let i = 0; i < n; i++) {
    const j = i - pul.cursor + pul.eq_cursor;
    if (j >= 0 && j < pul.eq_pulse.length) eqAligned[i] = pul.eq_pulse[j];
  }
  const opts = {
    width: 460, height: Math.max(170, $("results").clientHeight - 170),
    scales: { x: { time: false } },
    axes: [
      { stroke: "#8a8aa3", grid: { stroke: "#33334a", width: 1 },
        ticks: { stroke: "#33334a" }, label: "symbol", labelSize: 18,
        labelFont: "11px sans-serif", font: "11px sans-serif" },
      { stroke: "#8a8aa3", grid: { stroke: "#33334a", width: 1 },
        ticks: { stroke: "#33334a" }, label: "normalised", size: 46,
        labelFont: "11px sans-serif", font: "11px sans-serif" },
    ],
    series: [
      { label: "k" },
      { label: "pulse h[k]", stroke: "#6ecbf5", width: 1.6,
        points: { show: true, size: 5 } },
      { label: "after EQ", stroke: "#8fd18f", width: 1.6, dash: [5, 4],
        points: { show: true, size: 4 } },
    ],
    cursor: { drag: { x: true, y: false } },
  };
  plots.push(new uPlot(opts, [idx, pul.h, eqAligned],
                       holder.querySelector("#link-plot")));
}

/* tiny 1-D k-means for eye levels (k = 2 or 4), quantile-initialised */
function kmeans1d(xs, k) {
  const sorted = [...xs].sort((a, b) => a - b);
  let levels = Array.from({ length: k }, (_, i) =>
    sorted[Math.floor((i + 0.5) / k * sorted.length)]);
  const assign = new Array(xs.length).fill(0);
  for (let iter = 0; iter < 24; iter++) {
    for (let i = 0; i < xs.length; i++) {
      let best = 0, bd = Infinity;
      for (let c = 0; c < k; c++) {
        const d = Math.abs(xs[i] - levels[c]);
        if (d < bd) { bd = d; best = c; }
      }
      assign[i] = best;
    }
    const sum = new Array(k).fill(0), cnt = new Array(k).fill(0);
    for (let i = 0; i < xs.length; i++) { sum[assign[i]] += xs[i]; cnt[assign[i]]++; }
    let moved = 0;
    for (let c = 0; c < k; c++) {
      if (!cnt[c]) continue;
      const nl = sum[c] / cnt[c];
      moved += Math.abs(nl - levels[c]);
      levels[c] = nl;
    }
    if (moved < 1e-12) break;
  }
  levels.sort((a, b) => a - b);
  // re-assign against sorted levels so eye pairing is ordered
  for (let i = 0; i < xs.length; i++) {
    let best = 0, bd = Infinity;
    for (let c = 0; c < k; c++) {
      const d = Math.abs(xs[i] - levels[c]);
      if (d < bd) { bd = d; best = c; }
    }
    assign[i] = best;
  }
  return { levels, assign };
}
document.querySelectorAll(".rtab").forEach((b) =>
  b.addEventListener("click", () => setResultsTab(b.dataset.tab)));
$("btn-results-toggle").addEventListener("click", () => {
  $("results").classList.toggle("collapsed");
  $("btn-results-toggle").innerHTML =
    $("results").classList.contains("collapsed") ? "&#9650;" : "&#9660;";
});

$("btn-csv").addEventListener("click", () => {
  if (!lastResult?.ok || !lastResult.traces) return;
  const res = lastResult;
  const traces = visibleTraces(res.traces);  // export matches what's plotted
  const cols = [res.xlabel || "x", ...traces.map((t) => `${t.name} [${t.unit}]`)];
  let csv = cols.join(",") + "\n";
  for (let i = 0; i < res.x.length; i++) {
    csv += [res.x[i], ...traces.map((t) => t.values[i])].join(",") + "\n";
  }
  // append each extra plot (optical spectra, spec()/psd() results) as its own
  // block: they live on a different x axis (wavelength/frequency), so they get
  // their own header + rows below a blank separator line
  for (const ep of res.extra_plots || []) {
    const eTraces = ep.traces || [];
    if (!eTraces.length) continue;
    const eCols = [ep.xlabel || "x",
      ...eTraces.map((t) => `${t.name} [${t.unit || ep.yunit || "-"}]`)];
    csv += "\n" + eCols.join(",") + "\n";
    for (let i = 0; i < ep.x.length; i++) {
      csv += [ep.x[i], ...eTraces.map((t) => t.values[i])].join(",") + "\n";
    }
  }
  download("photonflux_results.csv", csv, "text/csv");
});

// ---------------------------------------------------------------------------
// save / load / examples
// ---------------------------------------------------------------------------
function download(name, text, mime) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: mime }));
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

function saveJSON() {
  download("circuit.json", JSON.stringify(
    { title: "photonflux circuit", schematic: state, analysis: collectAnalysis() },
    null, 2), "application/json");
}
$("btn-save").addEventListener("click", saveJSON);

$("btn-load").addEventListener("click", () => $("file-load").click());
$("file-load").addEventListener("change", async () => {
  const f = $("file-load").files[0];
  if (!f) return;
  try {
    loadDocument(JSON.parse(await f.text()));
  } catch (e) { setHint(`could not load: ${e}`, true); }
  $("file-load").value = "";
});

$("btn-new").addEventListener("click", () => {
  commit(() => { state = { instances: {}, wires: [], probes: [], notes: [],
    globals: { baud: DEFAULT_BAUD } }; });
  selection = null;
  renderInspector();
});

$("btn-undo").addEventListener("click", undo);
$("btn-redo").addEventListener("click", redo);

$("glob-baud").addEventListener("change", () => {
  const v = parseSI($("glob-baud").value);
  if (!(v > 0)) { $("glob-baud").value = fmtNum(globalBaud()); return; }
  commit(() => { state.globals = state.globals || {}; state.globals.baud = v; });
  // the eye folds at the global UI — keep its control in step
  $("eye-ui").value = fmtNum(globalUI());
  if ($("tab-eye").classList.contains("active")) renderEye();
});
$("glob-baud").addEventListener("keydown",
  (e) => { if (e.key === "Enter") e.target.blur(); });

function loadDocument(doc) {
  const sch = doc.schematic || doc;
  commit(() => {
    state = {
      instances: sch.instances || {},
      wires: (sch.wires || []).map((w) =>
        Array.isArray(w) ? { from: w[0], to: w[1] } : w),
      probes: sch.probes || [],
      notes: sch.notes || [],
      globals: sch.globals || {},
    };
    adoptGlobals(state);
  });
  selection = null;
  applyAnalysis(doc.analysis);
  $("eye-ui").value = fmtNum(globalUI());   // eye folds at the loaded baud rate
  renderInspector();
  zoomToFit();
}

function zoomToFit() {
  const insts = Object.values(state.instances);
  if (!insts.length) return;
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (const i of insts) {
    const sym = S[i.type];
    x0 = Math.min(x0, i.x - 20); y0 = Math.min(y0, i.y - 30);
    x1 = Math.max(x1, i.x + sym.w + 20); y1 = Math.max(y1, i.y + sym.h + 30);
  }
  for (const note of state.notes || []) {
    const rows = (note.title ? 1 : 0) +
      (note.lines || String(note.text || "").split("\n")).length;
    const longest = (note.lines || String(note.text || "").split("\n"))
      .concat(note.title || []).reduce((m, s) => Math.max(m, s.length), 0);
    const nx = note.x || 0, ny = note.y || 0;
    x0 = Math.min(x0, nx); y0 = Math.min(y0, ny);
    x1 = Math.max(x1, nx + (note.w || longest * 6.1 + 18));
    y1 = Math.max(y1, ny + 18 + rows * 15);
  }
  const r = svg.getBoundingClientRect();
  if (r.width < 60 || r.height < 60) {  // pane hidden/not laid out yet
    requestAnimationFrame(() => setTimeout(zoomToFit, 120));
    return;
  }
  const k = Math.min(1.6, Math.max(0.25,
    Math.min(r.width / (x1 - x0), r.height / (y1 - y0)) * 0.92));
  view.k = k;
  view.x = (r.width - (x1 + x0) * k) / 2;
  view.y = (r.height - (y1 + y0) * k) / 2;
  render();
}

async function loadExamples() {
  const sel = $("sel-example");
  try {
    const list = await (await fetch("/api/examples")).json();
    // one <optgroup> per example group, in first-seen (numeric) order
    const groups = new Map();
    for (const ex of list) {
      const g = ex.group || "More";
      if (!groups.has(g)) {
        const og = document.createElement("optgroup");
        og.label = g;
        groups.set(g, og);
        sel.appendChild(og);
      }
      const o = document.createElement("option");
      o.value = ex.id; o.textContent = ex.title;
      if (ex.description) o.title = ex.description;   // hover shows the full blurb
      groups.get(g).appendChild(o);
    }
  } catch {}
  sel.addEventListener("change", async () => {
    if (!sel.value) return;
    try {
      const doc = await (await fetch("/api/examples/" + sel.value)).json();
      loadDocument(doc);
      if (doc.description) setHint(doc.description);   // info panel on load
    } catch (e) { setHint(`could not load example: ${e}`, true); }
    sel.value = "";
  });
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
// synthesize a box glyph for catalog types without a hand-drawn symbol
// (user-uploaded .va models): ports split left/right, labels on
function genericSymbol(type, entry) {
  const ports = entry.ports || [];
  const left = ports.filter((_, i) => i % 2 === 0);
  const right = ports.filter((_, i) => i % 2 === 1);
  const rows = Math.max(left.length, right.length, 1);
  const h = Math.max(40, rows * 20 + 20);
  const pins = {};
  left.forEach((p, i) => { pins[p.name] = [0, 20 + i * 20]; });
  right.forEach((p, i) => { pins[p.name] = [100, 20 + i * 20]; });
  return {
    w: 100, h, pins, label: [8, -6], pinLabels: true,
    draw: () => `
      <rect class="body body-fill" x="12" y="4" width="76" height="${h - 8}" rx="5"/>
      ${left.map((p, i) => `<line class="body" x1="0" y1="${20 + i * 20}" x2="12" y2="${20 + i * 20}"/>`).join("")}
      ${right.map((p, i) => `<line class="body" x1="88" y1="${20 + i * 20}" x2="100" y2="${20 + i * 20}"/>`).join("")}
      <text x="18" y="${h / 2 + 3}" style="font-size:8px">${type.replace(/^uva_/, "")}</text>`,
  };
}

function ensureSymbols() {
  for (const [type, entry] of Object.entries(CAT)) {
    if (!S[type]) S[type] = genericSymbol(type, entry);
  }
}

// upload a .va model: compiles server-side, appears in the palette
$("btn-upva").addEventListener("click", () => {
  const file = document.createElement("input");
  file.type = "file";
  file.accept = ".va";
  file.onchange = () => {
    const f = file.files[0];
    if (!f) return;
    f.text().then(async (text) => {
      const resp = await (await fetch("/api/upload_va", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: f.name, content: text }),
      })).json();
      if (!resp.ok) { alert(`compile failed:\n${resp.error}`); return; }
      CAT = await (await fetch("/api/components")).json();
      ensureSymbols();
      buildPalette();
      const w = (resp.warnings || []).join("\n");
      alert(`registered ${resp.entry.label}` + (w ? `\n\nnotes:\n${w}` : ""));
    });
  };
  file.click();
});

async function boot() {
  CAT = await (await fetch("/api/components")).json();
  ensureSymbols();
  buildPalette();
  await loadExamples();
  $("sel-analysis").dispatchEvent(new Event("change"));

  const saved = localStorage.getItem("photonflux_sch");
  let loaded = false;
  if (saved) {
    try {
      const sch = JSON.parse(saved);
      if (Object.keys(sch.instances || {}).length) {
        state = sch; adoptGlobals(state); loaded = true; render(); zoomToFit();
        // analysis config isn't autosaved, but the sweep pane should survive
        // a plain reload (like the fx panel) — restore it from localStorage
        restoreRunCfg();
      }
    } catch {}
  }
  if (!loaded) {
    // first visit: show the photodiode + TIA example (loadDocument ->
    // applyAnalysis -> restorePaneFromAnalysis sets the pane from the doc)
    try {
      const doc = await (await fetch("/api/examples/01_photodiode_tia")).json();
      loadDocument(doc);
      undoStack = [];
    } catch { render(); }
  }
  renderInspector();
}
boot();
