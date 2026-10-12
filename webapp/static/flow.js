// ---------------------------------------------------------------------------
// Testbench flows: a pre-configured chain of analyses (find a resonance ->
// PAM4 eye -> optimize -> evaluate) edited in the Run panel and run in one go.
//
// The flow document lives on the tab record (tabs[i].flow) and is saved with
// the testbench as doc.flow next to doc.analysis (see webapp/flow.py for the
// format and the runner). In flow mode the Run panel's Analysis / Measure /
// Vary cards edit the *selected step*; the tab's own single analysis is
// stashed in tab.flowSingle until the mode is switched back.
// ---------------------------------------------------------------------------

const FLOW_KINDS = { analysis: "Analysis", optimize: "Optimize",
                     evaluate: "Evaluate" };
// evaluate's figures of merit: key -> [label, unit, default spec]
const FLOW_FOMS = {
  rel_oma_db: ["relOMA", "dB", ""], oma_dbm: ["OMA", "dBm", ""],
  il_db: ["IL", "dB", ""], er_db: ["ER", "dB", ""],
  tdecq_db: ["TDECQ", "dB", ""], tdec_db: ["TDEC", "dB", ""],
  rlm: ["RLM", "", ""],
};
const FL = { loading: false, saveTimer: null };

const flowTab = () => tabs[activeTab];
const flowDoc = () => flowTab()?.flow || null;
function flowStep() {
  const t = flowTab();
  return t?.flow?.steps?.[t.flowSel] || null;
}
const circled = (i) => (i < 20 ? String.fromCharCode(0x2460 + i) : `(${i + 1})`);

function flowFmt(v) {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(4);
  if (a >= 100) return String(+v.toFixed(4));
  return String(+v.toPrecision(4));
}

// plain decimals in the human range (0.5, not "500m"); SI prefixes otherwise
function flowNum(v) {
  const a = Math.abs(v);
  return a === 0 || (a >= 1e-3 && a < 1e5) ? String(+v.toPrecision(9)) : fmtNum(v);
}

// --- text <-> object helpers (the editors are plain "lhs = rhs" lines) -------
function flowLines(text) {
  return String(text || "").split("\n").map((l) => l.split("#")[0].trim())
    .filter(Boolean);
}
function flowParseAssign(text, what, errs, numeric = false) {
  const out = {};
  for (const line of flowLines(text)) {
    const m = line.match(/^([A-Za-z_*][\w*]*(?:\.[A-Za-z_]\w*)?)\s*=\s*(.+)$/);
    if (!m) { errs.push(`${what}: cannot read "${line}" (want name = value)`); continue; }
    if (numeric) {
      const v = parseSI(m[2]);
      if (isNaN(v)) { errs.push(`${what}: ${m[1]} needs a number`); continue; }
      out[m[1]] = v;
    } else out[m[1]] = m[2].trim();
  }
  return out;
}
const flowFmtAssign = (obj) => Object.entries(obj || {})
  .map(([k, v]) => `${k} = ${typeof v === "number" ? flowNum(v) : v}`).join("\n");

function flowParseParams(text, errs) {
  const out = [];
  for (const line of flowLines(text)) {
    const m = line.match(/^([A-Za-z_]\w*)(?:\.([A-Za-z_]\w*))?\s*=\s*([^:]+):(.+)$/);
    const lo = m ? parseSI(m[3]) : NaN, hi = m ? parseSI(m[4]) : NaN;
    if (!m || isNaN(lo) || isNaN(hi) || !(hi > lo)) {
      errs.push(`vary: cannot read "${line}" (want name = min : max)`);
      continue;
    }
    out.push(m[2] ? { inst: m[1], param: m[2], min: lo, max: hi }
                  : { var: m[1], min: lo, max: hi });
  }
  return out;
}
const flowFmtParams = (ps) => (ps || []).map((p) =>
  `${p.var || `${p.inst}.${p.param}`} = ${flowNum(p.min)} : ${flowNum(p.max)}`)
  .join("\n");

function flowNewId(doc) {
  const used = new Set(doc.steps.map((s) => s.id));
  let n = doc.steps.length + 1;
  while (used.has(`s${n}`)) n++;
  return `s${n}`;
}

function flowNewDoc(single) {
  return { name: "", vars: {}, set: {},
           steps: [{ id: "s1", kind: "analysis", title: "Step 1",
                     analysis: single && single.mode !== "optimize"
                       ? single : { mode: "transient" },
                     set: {}, outputs: {}, checks: [] }] };
}

function flowPam4() {
  const p = Object.values(state.instances).find((i) => i.type === "prbs");
  return (p?.settings?.mode || "nrz") === "pam4";
}

// the Eye tab's stateye settings, saved into the flow so a headless run
// (notebook / CLI) scores exactly like the browser
function flowStateye() { return stateyeCfg(flowPam4() ? 4 : 2); }

function flowApplyStateye(cfg) {
  if (!cfg) return;
  const set = (id, v) => { if (v !== undefined && v !== null) $(id).value = v; };
  set("tq-rxbw", cfg.rx_bw === undefined ? undefined
    : (cfg.rx_bw === "auto" ? "auto" : String(cfg.rx_bw)));
  set("tq-rxorder", cfg.rx_order); set("tq-taps", cfg.taps);
  set("tq-pre", cfg.pre); set("tq-dfe", cfg.dfe); set("tq-method", cfg.method);
  set("tq-mu", cfg.mu); set("tq-passes", cfg.passes); set("tq-evals", cfg.max_evals);
  set("tq-manual", cfg.manual); set("tq-manual-dfe", cfg.manual_dfe);
  set("tq-snoise", cfg.s_noise); set("tq-ser", cfg.ser ?? cfg.ber);
  if (cfg.limits !== undefined) $("tq-limits").checked = !!cfg.limits;
  if (cfg.use_pattern !== undefined) $("tq-pattern").checked = !!cfg.use_pattern;
  if (cfg.nx && cfg.ny) {
    const g = `${cfg.nx}x${cfg.ny}`;
    if ([...$("tq-grid").options].some((o) => o.value === g)) $("tq-grid").value = g;
  }
  syncTdecqControls();
}

// ---------------------------------------------------------------------------
// hooks called from app.js
// ---------------------------------------------------------------------------
// the tab's single analysis while the Run panel shows a flow step
function flowSingleAnalysis(t) {
  flowSaveCards();
  return t.flowSingle || t.analysis || collectAnalysis();
}

// {analysis, flow} for Save / the notebook mirror
function flowDocFields() {
  const t = flowTab();
  if (!t) return { analysis: collectAnalysis() };
  const analysis = t.flowOn ? flowSingleAnalysis(t) : collectAnalysis();
  if (!t.flow) return { analysis };
  return { analysis, flow: { ...t.flow, stateye: flowStateye() } };
}

// a loaded document (Load / Examples): adopt its flow, if any
function flowLoadDoc(doc) {
  const t = flowTab();
  if (!t) return;
  t.flow = doc.flow ? JSON.parse(JSON.stringify(doc.flow)) : null;
  t.flowOn = !!doc.flow;
  t.flowSel = 0;
  t.flowSingle = collectAnalysis();
  t.flowResult = null; t.flowView = null;
  if (doc.flow?.stateye) flowApplyStateye(doc.flow.stateye);
  flowFillVars();
  if (t.flowOn) flowLoadStep();
  flowSyncPanel(); flowRenderSteps(); flowSyncResultsBar();
  if ($("tab-flow").classList.contains("active")) flowRenderTab();
}

// a flow pushed from a notebook (Session.push of a document with a flow)
function flowAdoptRemote(remote) {
  const t = flowTab();
  if (!t || JSON.stringify(remote) === JSON.stringify(t.flow)) return;
  if (t.flowOn) flowSaveCards();
  t.flow = JSON.parse(JSON.stringify(remote));
  t.flowSel = Math.min(t.flowSel | 0, t.flow.steps.length - 1);
  flowFillVars();
  if (t.flowOn) flowLoadStep();
  flowSyncPanel(); flowRenderSteps();
}

// tab switch: show this tab's flow (or none)
function flowCheckout(t) {
  if (t.flowOn && t.flow) {
    t.flowSingle = t.analysis;
    t.flowSel = Math.min(Math.max(0, t.flowSel | 0), t.flow.steps.length - 1);
    flowLoadStep();
  } else t.flowOn = false;
  flowFillVars();
  flowSyncPanel(); flowRenderSteps(); flowSyncResultsBar();
  if ($("tab-flow").classList.contains("active")) {
    if (t.flowResult || t.flowLive) flowRenderTab(); else setResultsTab("plots");
  }
  if (t.flowLive) t.flowLive.key = "";   // the next poll repaints, Log included
}

// ---------------------------------------------------------------------------
// panel <-> step
// ---------------------------------------------------------------------------
function flowSyncPanel() {
  const t = flowTab();
  const on = !!(t && t.flowOn && t.flow);
  document.querySelectorAll("#flow-card .flow-mode button").forEach((b) =>
    b.classList.toggle("on", (b.dataset.flow === "on") === on));
  $("flow-body").hidden = !on;
  $("flow-stephdr").hidden = !on;
  const kind = on ? (flowStep()?.kind || "analysis") : "analysis";
  for (const id of ["rp-analysis", "rp-measure", "rp-vary"])
    $(id).hidden = on && kind !== "analysis";
  $("flow-set-card").hidden = !(on && kind === "analysis");
  $("flow-measure").hidden = !(on && kind === "analysis");
  $("flow-opt-card").hidden = !(on && kind === "optimize");
  $("flow-eval-card").hidden = !(on && kind === "evaluate");
  // a flow has its own Optimize step kind; the single-run optimizer hides
  const optOpt = [...$("sel-analysis").options].find((o) => o.value === "optimize");
  if (optOpt) optOpt.hidden = on;
  $("btn-run").innerHTML = on ? "&#9654; Run flow" : "&#9654; Run";
  $("btn-run").title = on
    ? "Run the whole flow (Cmd+Enter; Shift: from the selected step; Alt: that step only)"
    : "Run simulation (Cmd+Enter)";
  $("btn-run-menu").hidden = !on;
  if (!on) $("run-menu").hidden = true;
}

function flowInnerBoxes(holder, st) {
  const doc = flowDoc();
  holder.innerHTML = "";
  const idx = doc.steps.indexOf(st);
  const cands = doc.steps.filter((s, i) => (s.kind || "analysis") === "analysis" && i < idx);
  if (!cands.length) {
    holder.innerHTML = `<span class="dim">add an analysis step before this one</span>`;
    return;
  }
  for (const s of cands) {
    const lab = document.createElement("label");
    lab.className = "flow-chk";
    const box = document.createElement("input");
    box.type = "checkbox"; box.value = s.id;
    box.checked = (st.inner || []).includes(s.id);
    lab.append(box, ` ${circled(doc.steps.indexOf(s))} ${s.title || s.id}`);
    holder.appendChild(lab);
  }
}
const flowInnerValue = (holder) =>
  [...holder.querySelectorAll("input[type=checkbox]")].filter((b) => b.checked)
    .map((b) => b.value);

function flowProbeOptions(sel, value, allowNone) {
  sel.innerHTML = "";
  if (allowNone) sel.add(new Option("—", ""));
  for (const p of state.probes) sel.add(new Option(p.name, p.name));
  sel.value = value || "";
}

function flowFomRows(st) {
  const holder = $("flow-eval-fom");
  holder.innerHTML = "";
  const pam4 = flowPam4();
  const keys = Object.keys(FLOW_FOMS).filter((k) =>
    pam4 ? k !== "tdec_db" : k !== "tdecq_db" && k !== "rlm");
  for (const k of keys) {
    const [label, unit] = FLOW_FOMS[k];
    const on = st.fom && k in st.fom;
    const row = document.createElement("div");
    row.className = "flow-fom-row";
    row.innerHTML = `<label class="flow-chk"><input type="checkbox" data-fom="${k}"
      ${on ? "checked" : ""}> ${label}${unit ? ` <span class="dim">[${unit}]</span>` : ""}</label>
      <input type="text" data-spec="${k}" placeholder="&ge; / &le; value"
        value="${on && st.fom[k] ? escapeHtml(String(st.fom[k])) : ""}">`;
    holder.appendChild(row);
  }
}

function flowStateyeNote() {
  const c = flowStateye(), pam4 = flowPam4();
  const rx = c.rx_bw === "off" ? "no ref Rx" : `BT${c.rx_order} ref Rx at ${c.rx_bw}&times;baud`;
  const allAuto = [c.taps, c.pre, c.dfe].every((v) => v === "auto");
  const eq = pam4 ? ` &middot; ${allAuto ? "lane rate's 802.3 reference EQ"
      : `FFE ${c.taps} (${c.pre} pre)${c.dfe && c.dfe !== "0" ? ` + DFE ${c.dfe}` : ""}`} ${c.method}`
    + `${c.limits ? " &middot; tap limits" : ""} &middot; SER ${c.ser}` : ` &middot; BER ${c.ber}`;
  $("flow-eval-stateye").innerHTML = `stateye: the Eye tab's settings &mdash;
    ${rx}${eq} &middot; S ${c.s_noise} mW.
    <button type="button" class="rp-link" id="flow-eval-eyelink">edit in the Eye tab</button>`;
  $("flow-eval-eyelink").addEventListener("click", () => {
    $("results").classList.remove("collapsed"); setResultsTab("eye");
  });
}

// load the selected step into the Run panel
function flowLoadStep() {
  const t = flowTab(), st = flowStep();
  if (!t || !st) return;
  FL.loading = true;
  try {
    $("flow-step-badge").textContent = `STEP ${t.flowSel + 1}`;
    $("flow-step-title").value = st.title || "";
    $("flow-step-kind").textContent = FLOW_KINDS[st.kind || "analysis"];
    for (const id of ["flow-set-err", "flow-measure-err", "flow-opt-err", "flow-eval-err"])
      $(id).textContent = "";
    if ((st.kind || "analysis") === "analysis") {
      applyAnalysis(st.analysis || { mode: "transient" });
      $("flow-step-set").value = flowFmtAssign(st.set);
      $("flow-step-outputs").value = flowFmtAssign(st.outputs);
      $("flow-step-checks").value = (st.checks || []).join("\n");
    } else if (st.kind === "optimize") {
      flowInnerBoxes($("flow-opt-inner"), st);
      $("flow-opt-params").value = flowFmtParams(st.params);
      $("flow-opt-obj").value = st.objective || "";
      $("flow-opt-max").value = st.maximize === false ? "0" : "1";
      $("flow-opt-cons").value = (st.constraints || []).join("\n");
      $("flow-opt-iters").value = st.iters || 30;
    } else {
      flowInnerBoxes($("flow-eval-inner"), st);
      $("flow-eval-record").value = st.record || "full_period";
      flowProbeOptions($("flow-eval-in"), st.probes?.in, true);
      flowProbeOptions($("flow-eval-out"), st.probes?.out, false);
      flowFomRows(st);
      flowStateyeNote();
    }
  } finally { FL.loading = false; }
  flowSyncPanel();
}

// the Run panel -> the selected step (and the vars editor -> the doc)
function flowSaveCards() {
  if (FL.loading) return;
  const t = flowTab(), doc = flowDoc(), st = flowStep();
  if (!t?.flowOn || !doc || !st) return;
  doc.name = $("flow-name").value.trim();
  st.title = $("flow-step-title").value.trim() || st.id;
  const kind = st.kind || "analysis";
  if (kind === "analysis") {
    const a = collectAnalysis();
    // keep a start:stop:count DC sweep compact instead of 1000s of values
    const m = $("rc-values").value.trim().match(/^([^:,]+):([^:,]+):([^:,]+)$/);
    if (a.mode === "dcsweep" && m && !a.step_instance) {
      delete a.values;
      Object.assign(a, { start: parseSI(m[1]), stop: parseSI(m[2]),
                         points: parseInt(m[3]) || 101 });
    }
    st.analysis = a;
    const e1 = [], e2 = [];
    st.set = flowParseAssign($("flow-step-set").value, "set", e1);
    st.outputs = flowParseAssign($("flow-step-outputs").value, "outputs", e2);
    for (const k of Object.keys(st.outputs))
      if (!/^[A-Za-z_]\w*$/.test(k)) { e2.push(`output "${k}" must be a plain name`); delete st.outputs[k]; }
    st.checks = flowLines($("flow-step-checks").value);
    $("flow-set-err").textContent = e1.join("; ");
    $("flow-measure-err").textContent = e2.join("; ");
  } else if (kind === "optimize") {
    const errs = [];
    st.inner = flowInnerValue($("flow-opt-inner"));
    st.params = flowParseParams($("flow-opt-params").value, errs);
    st.objective = $("flow-opt-obj").value.trim();
    st.maximize = $("flow-opt-max").value === "1";
    st.constraints = flowLines($("flow-opt-cons").value);
    st.iters = parseInt($("flow-opt-iters").value) || 30;
    $("flow-opt-err").textContent = errs.join("; ");
  } else {
    const errs = [];
    st.inner = flowInnerValue($("flow-eval-inner"));
    st.record = $("flow-eval-record").value;
    st.probes = { in: $("flow-eval-in").value, out: $("flow-eval-out").value };
    const fom = {};
    $("flow-eval-fom").querySelectorAll("input[data-fom]").forEach((b) => {
      if (!b.checked) return;
      const k = b.dataset.fom;
      const spec = $("flow-eval-fom").querySelector(`input[data-spec="${k}"]`).value.trim();
      if (spec && !/^(>=|<=|>|<)\s*[-+]?[\d.]+(e[-+]?\d+)?$/i.test(spec))
        errs.push(`${FLOW_FOMS[k][0]}: spec "${spec}" (want >= value or <= value)`);
      fom[k] = spec || null;
    });
    st.fom = fom;
    $("flow-eval-err").textContent = errs.join("; ");
  }
  flowRenderSteps();
}

// the vars editor shows the active flow: refill it whenever that changes
function flowFillVars() {
  const doc = flowDoc();
  $("flow-vars-text").value = doc ? flowFmtAssign(doc.vars) : "";
  $("flow-set-text").value = doc ? flowFmtAssign(doc.set) : "";
  $("flow-vars-err").textContent = "";
}

function flowVarsEdited() {
  const doc = flowDoc();
  if (!doc || !flowTab()?.flowOn) return;
  const errs = [];
  doc.vars = flowParseAssign($("flow-vars-text").value, "variables", errs, true);
  doc.set = flowParseAssign($("flow-set-text").value, "every step sets", errs);
  $("flow-vars-err").textContent = errs.join("; ");
  autosave();
}

function flowEdited() {
  if (FL.loading || !flowTab()?.flowOn) return;
  clearTimeout(FL.saveTimer);
  FL.saveTimer = setTimeout(() => { flowSaveCards(); autosave(); }, 200);
}

function flowSelect(i) {
  const t = flowTab();
  if (!t?.flow || i === t.flowSel || !t.flow.steps[i]) return;
  flowSaveCards();
  t.flowSel = i;
  flowLoadStep();
  flowRenderSteps();
  autosave();
}

function flowSetMode(on) {
  const t = flowTab();
  if (!t || !!t.flowOn === on) return;
  if (on) {
    t.flowSingle = collectAnalysis();
    if (!t.flow) t.flow = flowNewDoc(t.flowSingle);
    t.flowOn = true;
    t.flowSel = Math.min(t.flowSel | 0, t.flow.steps.length - 1);
    flowFillVars();
    flowLoadStep();
  } else {
    flowSaveCards();
    t.flowOn = false;
    applyAnalysis(t.flowSingle || t.analysis || { mode: "transient" });
  }
  flowSyncPanel(); flowRenderSteps(); autosave();
}

function flowAddStep(kind) {
  const t = flowTab(), doc = flowDoc();
  if (!doc) return;
  flowSaveCards();
  const id = flowNewId(doc);
  const before = doc.steps.slice(0, t.flowSel + 1)
    .filter((s) => (s.kind || "analysis") === "analysis")
    .map((s) => s.id);
  const probes = state.probes.map((p) => p.name);
  const st = kind === "optimize"
    ? { id, kind, title: "Optimize", inner: before, params: [], objective: "",
        maximize: true, constraints: [], iters: 30 }
    : kind === "evaluate"
      ? { id, kind, title: "Evaluate", inner: before, record: "full_period",
          probes: { in: "", out: probes[0] || "" },
          fom: Object.fromEntries(Object.keys(FLOW_FOMS)
            .filter((k) => (flowPam4() ? k !== "tdec_db" : !["tdecq_db", "rlm"].includes(k)))
            .map((k) => [k, null])) }
      : { id, kind: "analysis", title: "Analysis",
          analysis: JSON.parse(JSON.stringify(flowStep()?.analysis || { mode: "transient" })),
          set: {}, outputs: {}, checks: [] };
  doc.steps.splice(t.flowSel + 1, 0, st);
  t.flowSel += 1;
  flowLoadStep(); flowRenderSteps(); autosave();
}

function flowDuplicate() {
  const t = flowTab(), doc = flowDoc(), st = flowStep();
  if (!st) return;
  flowSaveCards();
  const copy = JSON.parse(JSON.stringify(st));
  copy.id = flowNewId(doc);
  copy.title = `${st.title || st.id} (copy)`;
  doc.steps.splice(t.flowSel + 1, 0, copy);
  t.flowSel += 1;
  flowLoadStep(); flowRenderSteps(); autosave();
}

function flowDelete() {
  const t = flowTab(), doc = flowDoc(), st = flowStep();
  if (!st || doc.steps.length < 2) { setHint("a flow keeps at least one step", true); return; }
  doc.steps.splice(t.flowSel, 1);
  for (const s of doc.steps)
    if (s.inner) s.inner = [].concat(s.inner).filter((i) => i !== st.id);
  t.flowSel = Math.min(t.flowSel, doc.steps.length - 1);
  flowLoadStep(); flowRenderSteps(); autosave();
}

function flowMove(d) {
  const t = flowTab(), doc = flowDoc();
  const j = t.flowSel + d;
  if (!doc || j < 0 || j >= doc.steps.length) return;
  flowSaveCards();
  [doc.steps[t.flowSel], doc.steps[j]] = [doc.steps[j], doc.steps[t.flowSel]];
  t.flowSel = j;
  flowLoadStep(); flowRenderSteps(); autosave();
}

// ---------------------------------------------------------------------------
// the step list
// ---------------------------------------------------------------------------
function flowStepSummary(st, doc) {
  const kind = st.kind || "analysis";
  const refs = [].concat(st.inner || []).map((id) => circled(doc.steps.findIndex((s) => s.id === id)))
    .join("");
  if (kind === "optimize") return `Optimize · re-runs ${refs || "—"}`;
  if (kind === "evaluate")
    return `Evaluate · ${st.record === "inner" ? "as set up" : "full PRBS"} · ${refs || "—"}`;
  const a = st.analysis || {};
  const names = { dc: "DC operating point", ac: "AC", noise: "Noise", pulse: "Pulse / COM" };
  if (a.mode === "dcsweep") return `DC sweep · ${a.instance}.${a.param}`;
  if (a.mode === "transient")
    return `Transient · ${fmtSI(a.t_stop || 0)}s${a.run_config?.sweep ? " · swept" : ""}`;
  return names[a.mode] || a.mode || "Analysis";
}

function flowRenderSteps() {
  const t = flowTab(), holder = $("flow-steps");
  holder.innerHTML = "";
  if (!t?.flowOn || !t.flow) return;
  if (document.activeElement !== $("flow-name")) $("flow-name").value = t.flow.name || "";
  // while the flow runs, the live record (flowLiveUpdate) replaces the last result
  const recs = Object.fromEntries(((t.flowLive ? flowLiveSteps(t) : t.flowResult?.steps)
    || []).map((s) => [s.id, s]));
  t.flow.steps.forEach((st, i) => {
    const rec = recs[st.id];
    const el = document.createElement("div");
    el.className = "flow-step" + (i === t.flowSel ? " sel" : "");
    let status = "";
    if (rec) {
      const nfail = (rec.checks || []).filter((c) => !c.ok && !c.info).length;
      status = rec.status === "running"
        ? `<span class="fl-run"><span class="fl-spin"></span>${flowLiveTag(rec)}</span>`
        : rec.status === "ok"
        ? (nfail ? `<span class="err">&#10007; ${nfail} check${nfail > 1 ? "s" : ""}</span>`
                 : `<span class="ok">&#10003; ${(rec.time_s || 0).toFixed(1)} s</span>`)
        : rec.status === "failed" ? `<span class="err">&#10007; failed</span>`
        : `<span class="dim">${rec.status}</span>`;
    }
    const outs = st.kind === "optimize" ? (st.params || []).map((p) => p.var || `${p.inst}.${p.param}`)
      : st.kind === "evaluate" ? Object.keys(st.fom || {})
      : Object.keys(st.outputs || {});
    const got = rec?.status === "ok" ? { ...(rec.outputs || {}) }
      : rec?.status === "running" ? flowLiveBest(rec).values : {};
    const chips = outs.slice(0, 6).map((n) =>
      `<span class="flow-chip">${escapeHtml(n)}${n in got
        ? ` ${flowFmt(got[n])}` : ""}</span>`).join("")
      + (outs.length > 6 ? `<span class="flow-chip">+${outs.length - 6}</span>` : "");
    el.innerHTML = `<span class="rp-num">${i + 1}</span>
      <div class="flow-step-txt"><div class="flow-step-k">${escapeHtml(flowStepSummary(st, t.flow))}</div>
        <div class="flow-step-t">${escapeHtml(st.title || st.id)}</div></div>
      <span class="flow-step-st">${status}</span>
      ${chips ? `<div class="flow-step-io">${chips}</div>` : ""}
      ${i === t.flowSel ? `<span class="flow-step-mv">
        <button type="button" data-mv="-1" title="Move up">&#9650;</button>
        <button type="button" data-mv="1" title="Move down">&#9660;</button></span>` : ""}`;
    el.addEventListener("click", (ev) => {
      const mv = ev.target.closest("button[data-mv]");
      if (mv) { flowMove(+mv.dataset.mv); return; }
      flowSelect(i);
    });
    holder.appendChild(el);
  });
}

// ---------------------------------------------------------------------------
// running
// ---------------------------------------------------------------------------
function flowRunAnalysis(kind) {
  const t = flowTab();
  flowSaveCards();
  const errs = ["flow-vars-err", "flow-set-err", "flow-measure-err", "flow-opt-err",
                "flow-eval-err"].map((id) => $(id).textContent).filter(Boolean);
  if (errs.length) { setHint(`flow: ${errs[0]}`, true); return null; }
  const a = { mode: "flow", flow: { ...JSON.parse(JSON.stringify(t.flow)),
                                    stateye: flowStateye() } };
  const st = flowStep();
  if (kind !== "all" && st && !(kind === "from" && t.flowSel === 0)) {
    if (!t.flowResult) {
      setHint("run the whole flow once first — later steps need the variables it computes", true);
      return null;
    }
    a[kind === "from" ? "start" : "only"] = st.id;
    a.prior = flowPrior(t.flowResult, t.flow.steps.slice(0, t.flowSel).map((s) => s.id));
  }
  return a;
}

// the variables/overrides earlier steps produced (knobs come from the doc, so
// edits to them count; later steps' outputs and optima must not leak back)
function flowPrior(fr, earlier) {
  const keep = (obj) => Object.fromEntries(Object.entries(obj || {})
    .filter(([k]) => earlier.includes(fr.origin?.[k])));
  return { vars: keep(fr.vars), overrides: keep(fr.overrides),
           origin: keep(fr.origin) };
}

// file a flow result on its tab (merging a partial re-run into the last one)
function flowStoreResult(tab, res, a) {
  const body = res.flow;
  body.log = res.log || [];
  if ((a.start || a.only) && tab.flowResult) {
    const prev = Object.fromEntries(tab.flowResult.steps.map((s) => [s.id, s]));
    body.steps = body.steps.map((s) => (s.status === "skipped" && prev[s.id]) ? prev[s.id] : s);
    body.failed = body.steps.find((s) => s.status === "failed")?.id || null;
    body.passed = !body.failed && body.steps.every((s) =>
      s.status === "ok" && (s.checks || []).every((c) => c.ok || c.info));
  }
  tab.flowResult = body;
  const withRes = body.steps.filter((s) => s.result);
  const keep = withRes.find((s) => s.id === tab.flowView);
  tab.flowView = keep ? keep.id : (withRes.find((s) => s.kind === "evaluate")
    || withRes[withRes.length - 1])?.id || null;
  const rec = body.steps.find((s) => s.id === tab.flowView);
  tab.lastResult = rec?.result || null;
}

function flowShowResult(tab) {
  if (tab !== flowTab()) return;
  lastResult = tab.lastResult;
  clearResults();
  if (lastResult?.ok) { rerenderResults(); renderLink(); }
  showLog({ log: tab.flowResult?.log || [] });
  flowSyncResultsBar();
  flowRenderSteps();
  flowRenderTab();
}

function flowSyncResultsBar() {
  const t = flowTab(), fr = t?.flowResult;
  $("rtab-flow").hidden = !fr && !t?.flowLive;
  $("flow-view-wrap").hidden = !fr;
  if (!fr) return;
  const sel = $("flow-view");
  sel.innerHTML = "";
  fr.steps.forEach((s, i) => {
    if (s.result) sel.add(new Option(`${circled(i)} ${s.title}`, s.id));
  });
  sel.value = t.flowView || "";
}

function flowSetView(id) {
  const t = flowTab(), fr = t?.flowResult;
  const rec = fr?.steps.find((s) => s.id === id);
  if (!rec?.result) return;
  t.flowView = id;
  t.lastResult = lastResult = rec.result;
  $("flow-view").value = id;
  clearResults();
  rerenderResults(); renderLink();
  showLog({ log: fr.log || [] });
  if (lastResult.kind === "op") renderOpTable(lastResult);
}

function flowApplyDesign() {
  const fr = flowTab()?.flowResult;
  if (!fr?.design?.length) return;
  commit(() => {
    for (const d of fr.design) {
      const inst = state.instances[d.inst];
      if (inst) (inst.settings = inst.settings || {})[d.param] = d.value;
    }
  });
  renderInspector();
  setHint(`applied ${fr.design.length} parameter values to the schematic (Cmd+Z undoes)`);
}

function flowReport() {
  const t = flowTab(), fr = t?.flowResult;
  if (!fr) return;
  const L = [`# ${t.flow?.name || tabTitle(t)}`, "",
             `Result: ${fr.passed ? "all checks pass" : "checks FAIL"}`, ""];
  for (const s of fr.steps.filter((x) => x.kind === "evaluate" && x.fom)) {
    L.push(`## ${s.title}`, "", "| figure of merit | surrogate | full record | spec | |",
           "|---|---|---|---|---|");
    for (const [k, spec] of Object.entries(s.spec || {})) {
      const chk = (s.checks || []).find((c) => c.expr.startsWith(`${k} `));
      L.push(`| ${FLOW_FOMS[k]?.[0] || k} | ${flowFmt(s.before?.[k])} | ${flowFmt(s.fom[k])} `
        + `| ${spec || ""} | ${chk ? (chk.ok ? "pass" : "FAIL") : ""} |`);
    }
    L.push("");
  }
  L.push("## Variables", "", "| name | value | from |", "|---|---|---|");
  for (const [k, v] of Object.entries(fr.vars)) L.push(`| ${k} | ${flowFmt(v)} | ${fr.origin?.[k] || ""} |`);
  for (const [k, v] of Object.entries(fr.overrides || {})) L.push(`| ${k} | ${flowFmt(v)} | ${fr.origin?.[k] || ""} |`);
  L.push("", "## Checks", "");
  fr.steps.forEach((s) => (s.checks || []).forEach((c) =>
    L.push(`- ${c.ok ? "pass" : "FAIL"} — ${s.title}: \`${c.expr}\` (${flowFmt(c.value)})`)));
  download("flow-report.md", L.join("\n") + "\n", "text/markdown");
}

// ---------------------------------------------------------------------------
// live status while a flow runs: the progress poll hands over the server's
// running record (progress.py flow_*: step states, outputs and checks, every
// Optimize evaluation, the log) a few times a second
// ---------------------------------------------------------------------------
function flowDur(sec) {
  sec = Math.max(0, Math.round(sec));
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  return m < 60 ? `${m}m${String(sec % 60).padStart(2, "0")}s`
    : `${Math.floor(m / 60)}h${String(m % 60).padStart(2, "0")}m`;
}

// "eval 12/60" (the one in flight) for a running Optimize, else "running"
const flowLiveTag = (s) => (s.iters
  ? `eval ${Math.min(s.iters, (s.eval || 0) + 1)}/${s.iters}` : "running");

// time left in a running Optimize step, from its mean evaluation time
function flowLiveEta(s) {
  const h = s.history || [], n = h.length;
  if (!s.iters || !n || n >= s.iters) return "";
  return `~${flowDur((h[n - 1].t_s / n) * (s.iters - n))} left`;
}

// the best evaluation so far: {entry, values: {param: value}}
function flowLiveBest(s) {
  let best = null;
  for (const e of s.history || []) if (e.J !== null && e.J !== undefined && (!best || e.J < best.J)) best = e;
  const values = {};
  if (best) (s.params || []).forEach((p, i) => { values[p.name] = best.x[i]; });
  return { entry: best, values };
}

// the live steps, with the last result standing in for steps a partial
// re-run (Run from / Run only) skips, as flowStoreResult merges them after
function flowLiveSteps(t) {
  const live = t.flowLive?.live;
  if (!live) return null;
  const prev = Object.fromEntries((t.flowResult?.steps || []).map((s) => [s.id, s]));
  return live.steps.map((s) => (s.status === "skipped" && prev[s.id]) || s);
}

function flowLiveStart(tab) {
  tab.flowLive = { live: null, key: "" };
  if (tab !== flowTab()) return;
  flowSyncResultsBar();
  flowRenderSteps();
  $("results").classList.remove("collapsed");
  setResultsTab("flow");
}

function flowLiveUpdate(tab, live) {
  const L = tab.flowLive;
  if (!L || !live) return;
  L.live = live;
  if (tab !== flowTab()) return;
  const key = `${live.log_n}|${live.steps.map((s) => s.status + (s.iters || "")).join()}`;
  if (key === L.key) {             // nothing new: just tick the clock
    const clk = $("fl-live-clock");
    if (clk) clk.textContent = flowDur(live.elapsed_s);
    return;
  }
  L.key = key;
  flowRenderSteps();
  if ($("tab-flow").classList.contains("active")) flowRenderTab();
  $("log-pre").textContent = live.log.join("\n");   // the Log tab streams too
}

// the run is over: a finished flow renders its result next (flowShowResult);
// a stopped or failed one falls back to the last result
function flowLiveEnd(tab, finished) {
  if (!tab?.flowLive) return;
  delete tab.flowLive;
  if (finished || tab !== flowTab()) return;
  flowSyncResultsBar();
  flowRenderSteps();
  if ($("tab-flow").classList.contains("active")) {
    if (tab.flowResult) flowRenderTab(); else setResultsTab("log");
  }
}

function flowRenderLive(t) {
  const holder = $("tab-flow"), live = t.flowLive.live, doc = t.flow;
  if (!live) { holder.innerHTML = `<div class="insp-empty">Starting the flow…</div>`; return; }
  const byId = Object.fromEntries((doc?.steps || []).map((s) => [s.id, s]));
  const steps = flowLiveSteps(t);
  const running = steps.find((s) => s.status === "running");
  let pipe = "";
  steps.forEach((s, i) => {
    const nfail = (s.checks || []).filter((c) => !c.ok && !c.info).length;
    const icon = s.status === "running" ? `<span class="fl-spin"></span>`
      : s.status === "ok" ? (nfail ? `<span class="err">&#10007;</span>` : `<span class="ok">&#10003;</span>`)
      : s.status === "failed" ? `<span class="err">&#10007;</span>` : `<span class="dim">&ndash;</span>`;
    const sub = s.status === "running" ? [flowLiveTag(s), flowLiveEta(s)].filter(Boolean).join(" · ")
      : s.time_s ? `${s.time_s.toFixed(1)} s` : s.status;
    if (i) pipe += `<div class="fl-edge">&rarr;</div>`;
    pipe += `<div class="fl-node${s.status === "running" ? " run" : ""}">
      <div class="fl-node-h"><span>${circled(i)} ${escapeHtml(s.title)}</span>${icon}</div>
      <div class="fl-node-s">${escapeHtml(flowStepSummary(byId[s.id] || s, doc))} · ${escapeHtml(sub)}</div>
      ${s.error ? `<div class="fl-node-e">${escapeHtml(s.error)}</div>` : ""}</div>`;
  });
  // Optimize steps: the convergence plot and where the best design sits
  let plots = "", best = "";
  steps.forEach((s, i) => {
    if (!s.iters) return;
    const p = flowPlotHistory(byId[s.id] || {}, { history: s.history || [], iters: s.iters,
                                                 objective: s.objective || "objective" });
    plots += `<div class="fl-plot${s.status === "running" ? " sel" : ""}">
      <div class="fl-plot-h">${circled(i)} ${escapeHtml(s.title)}</div>${p ? `${p.svg}
      <div class="fl-cap">${escapeHtml(p.cap)}</div>` : `<div class="insp-empty">waiting for the first evaluation…</div>`}</div>`;
    const b = flowLiveBest(s);
    if (!b.entry) return;
    const rows = (s.params || []).map((q) => {
      const v = b.values[q.name], u = (v - q.min) / (q.max - q.min || 1);
      return `<tr><td class="mono">${escapeHtml(q.name)}</td><td>${flowFmt(v)}</td>
        <td><div class="fl-rng" title="${flowFmt(q.min)} … ${flowFmt(q.max)}"><i style="left:${
          (100 * Math.min(1, Math.max(0, u))).toFixed(1)}%"></i></div></td></tr>`;
    }).join("");
    best += `<div class="fl-h">Best so far &mdash; ${circled(i)} eval ${s.history.indexOf(b.entry) + 1}:
      ${escapeHtml(s.objective.length > 28 ? "objective" : s.objective)} = ${flowFmt(b.entry.obj)}</div>
      <table class="fl-table"><tr><th>parameter</th><th>value</th><th>within its range</th></tr>${rows}</table>`;
  });
  const outs = steps.filter((s) => s.status === "ok" && !s.iters)
    .flatMap((s) => Object.entries(s.outputs || {}))
    .map(([k, v]) => `<tr><td class="mono">${escapeHtml(k)}</td><td>${flowFmt(v)}</td></tr>`).join("");
  const checks = steps.flatMap((s) => (s.checks || []).map((c) =>
    `<div class="fl-check">${c.ok ? `<span class="ok">&#10003;</span>`
      : c.info ? `<span class="warn">!</span>` : `<span class="err">&#10007;</span>`}
      <span><code>${escapeHtml(c.expr)}</code></span>
      <span class="dim">${c.value !== null && c.value !== undefined ? flowFmt(c.value) : ""}</span></div>`)).join("");
  const now = running ? `${escapeHtml(running.title)} · ${flowLiveTag(running)}` : "finishing";
  holder.innerHTML = `<div class="fl-res">
    <div class="fl-left"><div class="fl-pipe">${pipe}</div>
      ${plots ? `<div class="fl-plots">${plots}</div>` : ""}
      <div class="fl-h">Log</div><pre class="fl-log">${escapeHtml(live.log.slice(-14).join("\n"))}</pre></div>
    <div class="fl-right">
      <div class="fl-h">${escapeHtml(doc?.name || live.name || "Flow")} &mdash;
        <span class="fl-run">running <span id="fl-live-clock">${flowDur(live.elapsed_s)}</span></span></div>
      <div class="rp-note">Now: ${now}</div>
      ${best}
      ${outs ? `<div class="fl-h">Outputs</div><table class="fl-table">${outs}</table>` : ""}
      ${checks ? `<div class="fl-h">Checks</div>${checks}` : ""}
    </div></div>`;
  const lg = holder.querySelector(".fl-log");
  lg.scrollTop = lg.scrollHeight;     // keep the newest line in view
}

// ---------------------------------------------------------------------------
// the Flow results tab
// ---------------------------------------------------------------------------
const FSVG = { W: 260, H: 132, l: 30, r: 6, t: 6, b: 16 };
function fsvgFrame() {
  const { W, H, l, r, t, b } = FSVG;
  return {
    sx: (u) => l + u * (W - l - r), sy: (v) => t + (1 - v) * (H - t - b),
    open: `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <rect x="${l}" y="${t}" width="${W - l - r}" height="${H - t - b}" class="fl-frame"/>`,
  };
}
const fsvgText = (x, y, s, cls = "fl-ax", anchor = "start") =>
  `<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${escapeHtml(String(s))}</text>`;
const fsvgPoly = (f, pts, cls) => `<polyline class="${cls}" points="${
  pts.map(([u, v]) => `${f.sx(u).toFixed(1)},${f.sy(v).toFixed(1)}`).join(" ")}"/>`;

function fsvgRange(arrs) {
  let lo = Infinity, hi = -Infinity;
  for (const a of arrs) for (const v of a) if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (!isFinite(lo)) return [0, 1];
  if (hi - lo < 1e-30) { lo -= 0.5; hi += 0.5; }
  const pad = 0.06 * (hi - lo);
  return [lo - pad, hi + pad];
}

function flowPickTrace(res, prefer) {
  const trs = visibleTraces(res.traces || []);
  for (const n of prefer) {
    const tr = trs.find((q) => (q.probe || q.name) === n);
    if (tr) return tr;
  }
  return trs.find((q) => q.unit === "mW") || trs[0] || null;
}

// probe names an output expression mentions, in order (for the mini plot)
const flowProbesIn = (st) => Object.values(st.outputs || {}).join(" ")
  .match(/[A-Za-z_]\w*/g) || [];

function flowPlotCurve(st, rec, doc) {
  const res = rec.result;
  const tr = flowPickTrace(res, flowProbesIn(st));
  if (!tr) return "";
  const x = res.x, y = tr.values, n = x.length;
  const f = fsvgFrame();
  const [x0, x1] = [x[0], x[n - 1]], [y0, y1] = fsvgRange([y]);
  const step = Math.max(1, Math.floor(n / 400));
  const pts = [];
  for (let i = 0; i < n; i += step) pts.push([(x[i] - x0) / (x1 - x0 || 1), (y[i] - y0) / (y1 - y0)]);
  let svg = f.open + fsvgPoly(f, pts, "fl-line");
  // mark outputs that locate a point on the x axis (xat_min / xat_max / cross)
  for (const [name, expr] of Object.entries(st.outputs || {})) {
    const v = rec.outputs?.[name];
    if (!/^(xat_m(in|ax)|cross)\(/.test(String(expr).trim()) || !(v >= x0 && v <= x1)) continue;
    const u = (v - x0) / (x1 - x0);
    svg += `<line class="fl-mark" x1="${f.sx(u)}" x2="${f.sx(u)}" y1="${f.sy(0)}" y2="${f.sy(1)}"/>`
      + fsvgText(f.sx(u) + 3, f.sy(0.92), name, "fl-mark-t");
  }
  svg += fsvgText(FSVG.l, FSVG.H - 3, fmtNum(x0)) + fsvgText(FSVG.W - FSVG.r, FSVG.H - 3, fmtNum(x1), "fl-ax", "end")
    + fsvgText(2, f.sy(0.95) + 3, flowFmt(y1)) + fsvgText(2, f.sy(0.05), flowFmt(y0)) + "</svg>";
  return { svg, cap: `${tr.probe || tr.name} [${tr.unit}] vs ${res.xlabel || "x"}` };
}

function flowPlotEye(st, rec) {
  const res = rec.result;
  const prefer = st.kind === "evaluate" ? [st.probes?.out] : flowProbesIn(st);
  const tr = flowPickTrace(res, prefer);
  if (!tr) return "";
  const t = res.x, y = tr.values, ui = globalUI();
  const f = fsvgFrame();
  const [y0, y1] = fsvgRange([y]);
  const k0 = Math.ceil(t[0] / ui) + 16, k1 = Math.floor(t[t.length - 1] / ui) - 2;
  const nsym = Math.min(160, Math.max(0, k1 - k0));
  const stride = Math.max(1, Math.floor((k1 - k0) / Math.max(1, nsym)));
  let svg = f.open, j = 0;
  for (let s = 0; s < nsym; s++) {
    const k = k0 + s * stride, ta = k * ui, tb = (k + 2) * ui;
    while (j < t.length && t[j] < ta) j++;
    if (j > 0 && t[j] > ta) j--;
    const pts = [];
    const span = Math.max(1, Math.floor((tb - ta) / ((t[1] - t[0]) || ui) / 48));
    for (let i = j; i < t.length && t[i] <= tb; i += span)
      pts.push([(t[i] - ta) / (tb - ta), (y[i] - y0) / (y1 - y0)]);
    if (pts.length > 2) svg += fsvgPoly(f, pts, "fl-eye");
  }
  svg += fsvgText(2, f.sy(0.95) + 3, flowFmt(y1)) + fsvgText(2, f.sy(0.05), flowFmt(y0))
    + fsvgText(FSVG.W / 2, FSVG.H - 3, "2 UI", "fl-ax", "middle") + "</svg>";
  return { svg, cap: `${tr.probe || tr.name} [${tr.unit}] eye` };
}

function flowPlotHistory(st, rec) {
  const h = (rec.history || []).filter((e) => e.obj !== null && e.obj !== undefined);
  if (!h.length) return "";
  // live (flowRenderLive): the x axis spans the evaluation budget
  const f = fsvgFrame(), n = Math.max(rec.history.length, rec.iters || 0);
  const [y0, y1] = fsvgRange([h.map((e) => e.obj)]);
  let svg = f.open, bestJ = Infinity, best = null;
  const line = [];
  rec.history.forEach((e, i) => {
    if (e.obj === null || e.obj === undefined) return;
    const u = n > 1 ? i / (n - 1) : 0.5, v = (e.obj - y0) / (y1 - y0);
    // feasibility only means something when the step has constraints
    svg += `<circle cx="${f.sx(u).toFixed(1)}" cy="${f.sy(v).toFixed(1)}" r="2.2"
      class="${e.feasible === true ? "fl-ok" : e.feasible === false ? "fl-bad" : "fl-pt"}"/>`;
    if (e.J !== null && e.J < bestJ) { bestJ = e.J; best = e.obj; }
    if (best !== null) line.push([u, (best - y0) / (y1 - y0)]);
  });
  svg += fsvgPoly(f, line, "fl-best")
    + fsvgText(2, f.sy(0.95) + 3, flowFmt(y1)) + fsvgText(2, f.sy(0.05), flowFmt(y0))
    + fsvgText(FSVG.l, FSVG.H - 3, "eval 1") + fsvgText(FSVG.W - FSVG.r, FSVG.H - 3, String(n), "fl-ax", "end")
    + "</svg>";
  const obj = rec.objective.length > 28 ? "objective" : rec.objective;
  const legend = (rec.history || []).some((e) => e.feasible === true || e.feasible === false)
    ? " · green = meets the constraints" : "";
  return { svg, cap: `${obj} per evaluation${legend} · line = best` };
}

function flowRenderTab() {
  const holder = $("tab-flow");
  const t = flowTab(), fr = t?.flowResult, doc = t?.flow;
  if (t?.flowLive) { flowRenderLive(t); return; }
  if (!fr || !doc) { holder.innerHTML = `<div class="insp-empty">Run the flow to see its results here.</div>`; return; }
  const byId = Object.fromEntries(doc.steps.map((s) => [s.id, s]));
  const idx = Object.fromEntries(fr.steps.map((s, i) => [s.id, i]));
  // pipeline
  let pipe = "";
  fr.steps.forEach((s, i) => {
    const nfail = (s.checks || []).filter((c) => !c.ok && !c.info).length;
    const icon = s.status === "ok" ? (nfail ? `<span class="err">&#10007;</span>` : `<span class="ok">&#10003;</span>`)
      : s.status === "failed" ? `<span class="err">&#10007;</span>` : `<span class="dim">&ndash;</span>`;
    const st = byId[s.id] || s;
    if (i) pipe += `<div class="fl-edge">&rarr;</div>`;
    pipe += `<div class="fl-node${s.id === t.flowView ? " sel" : ""}" data-step="${s.id}">
      <div class="fl-node-h"><span>${circled(i)} ${escapeHtml(s.title)}</span>${icon}</div>
      <div class="fl-node-s">${escapeHtml(flowStepSummary(st, doc))}${s.time_s ? ` · ${s.time_s.toFixed(1)} s` : ""}</div>
      ${s.error ? `<div class="fl-node-e">${escapeHtml(s.error)}</div>` : ""}</div>`;
  });
  // one mini plot per step that produced something
  let plots = "";
  fr.steps.forEach((s, i) => {
    if (s.status !== "ok") return;
    const st = byId[s.id] || {};
    let p = "";
    if (s.kind === "optimize") p = flowPlotHistory(st, s);
    else if (s.result?.kind === "transient") p = flowPlotEye(st, s);
    else if (s.result?.x) p = flowPlotCurve(st, s, doc);
    if (!p) return;
    plots += `<div class="fl-plot${s.id === t.flowView ? " sel" : ""}" data-step="${s.id}">
      <div class="fl-plot-h">${circled(i)} ${escapeHtml(s.title)}</div>${p.svg}
      <div class="fl-cap">${escapeHtml(p.cap)}</div></div>`;
  });
  // scorecard(s)
  let score = "";
  for (const s of fr.steps.filter((x) => x.kind === "evaluate" && x.status === "ok")) {
    const rows = Object.entries(s.spec || {}).map(([k, spec]) => {
      const [label, unit] = FLOW_FOMS[k] || [k, ""];
      const chk = (s.checks || []).find((c) => c.expr.startsWith(`${k} `));
      let val = flowFmt(s.fom[k]);
      if (k === "rel_oma_db" && s.fom.rel_oma != null) val += ` <span class="dim">(${flowFmt(s.fom.rel_oma)})</span>`;
      return `<tr><td>${label}</td><td class="dim">${flowFmt(s.before?.[k])}</td>
        <td><b>${val}</b> <span class="dim">${unit}</span></td>
        <td class="dim">${spec ? escapeHtml(spec) : "&mdash;"}</td>
        <td>${chk ? (chk.ok ? `<span class="ok">&#10003;</span>` : `<span class="err">&#10007;</span>`) : ""}</td></tr>`;
    }).join("");
    score += `<div class="fl-h">Scorecard &mdash; ${escapeHtml(s.title)}</div>
      <table class="fl-table"><tr><th>figure of merit</th><th>before</th><th>evaluated</th><th>spec</th><th></th></tr>${rows}</table>
      <div class="rp-note">before = the value the earlier steps measured (the optimizer's short
        record); evaluated = ${s.family === "outer" ? "the full PRBS period, OMA<sub>outer</sub>"
        : "this record (no OMA<sub>outer</sub> pattern: crossing-point OMA)"}${s.p_in_mw ? `;
        P<sub>in</sub> ${flowFmt(s.p_in_mw)} mW at ${escapeHtml(byId[s.id]?.probes?.in || "")}` : ""}.
        relOMA = OMA / P<sub>in</sub>, IL = P<sub>in</sub> / top level.</div>`;
  }
  // variables
  const vrows = [...Object.entries(fr.vars), ...Object.entries(fr.overrides || {})]
    .map(([k, v]) => {
      const o = fr.origin?.[k];
      const from = o && o in idx ? circled(idx[o]) : (o === "vars" ? "var" : "");
      return `<tr><td class="mono">${escapeHtml(k)}</td><td>${flowFmt(v)}</td><td class="dim">${from}</td></tr>`;
    }).join("");
  const checks = fr.steps.flatMap((s, i) => (s.checks || []).map((c) =>
    `<div class="fl-check">${c.ok ? `<span class="ok">&#10003;</span>`
      : c.info ? `<span class="warn" title="optimizer constraint (not a spec)">!</span>`
      : `<span class="err">&#10007;</span>`}
      <span>${circled(i)} <code>${escapeHtml(c.expr)}</code>${c.info
        ? ` <span class="dim">optimizer</span>` : ""}</span>
      <span class="dim">${c.value !== null && c.value !== undefined ? flowFmt(c.value) : ""}</span></div>`)).join("");
  const verdict = fr.failed ? `<span class="err">a step failed</span>`
    : fr.passed ? `<span class="ok">all checks pass</span>` : `<span class="err">some checks fail</span>`;
  holder.innerHTML = `<div class="fl-res">
    <div class="fl-left"><div class="fl-pipe">${pipe}</div><div class="fl-plots">${plots}</div></div>
    <div class="fl-right">
      <div class="fl-h">${escapeHtml(doc.name || "Flow")} &mdash; ${verdict}</div>
      ${score}
      <div class="fl-h">Variables</div>
      <table class="fl-table"><tr><th>name</th><th>value</th><th>from</th></tr>${vrows}</table>
      ${checks ? `<div class="fl-h">Checks</div>${checks}` : ""}
      <div class="fl-apply">
        ${fr.design?.length ? `<div class="rp-note">The flow never edits the canvas. Write the final
          design (${fr.design.length} values) to the schematic?</div>` : ""}
        <div class="fl-btns">
          ${fr.design?.length ? `<button type="button" id="fl-apply">Apply design to schematic</button>` : ""}
          <button type="button" id="fl-report">Export report</button></div>
      </div>
    </div></div>`;
  holder.querySelectorAll("[data-step]").forEach((el) => el.addEventListener("click", () => {
    const id = el.dataset.step;
    const i = doc.steps.findIndex((s) => s.id === id);
    if (fr.steps.find((s) => s.id === id)?.result) flowSetView(id);
    if (i >= 0 && flowTab().flowOn) flowSelect(i);
    flowRenderTab();
  }));
  $("fl-apply")?.addEventListener("click", flowApplyDesign);
  $("fl-report")?.addEventListener("click", flowReport);
}

// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------
(function flowInit() {
  // "Set for this step" sits between the Analysis and Measure cards
  $("rp-analysis").after($("flow-set-card"));
  document.querySelectorAll("#flow-card .flow-mode button").forEach((b) =>
    b.addEventListener("click", () => flowSetMode(b.dataset.flow === "on")));
  $("flow-vars-btn").addEventListener("click", () => {
    const box = $("flow-vars");
    if (box.hidden) flowFillVars();
    box.hidden = !box.hidden;
    $("flow-vars-btn").classList.toggle("active", !box.hidden);
  });
  $("flow-add-kind").addEventListener("change", () => {
    const k = $("flow-add-kind").value;
    $("flow-add-kind").value = "";
    if (k) flowAddStep(k);
  });
  $("flow-dup").addEventListener("click", flowDuplicate);
  $("flow-del").addEventListener("click", flowDelete);
  for (const ev of ["input", "change"]) {
    $("runpanel").addEventListener(ev, (e) => {
      if (e.target.closest(".flow-mode")) return;
      if (e.target.closest("#flow-vars")) { flowVarsEdited(); return; }
      flowEdited();
    });
  }
  $("expr-text").addEventListener("input", flowEdited);
  $("flow-view").addEventListener("change", () => { flowSetView($("flow-view").value); flowRenderTab(); });
  // the Run options menu
  $("btn-run-menu").addEventListener("click", (e) => {
    e.stopPropagation();
    const menu = $("run-menu"), r = $("btn-run-menu").getBoundingClientRect();
    menu.hidden = !menu.hidden;
    if (!menu.hidden) {
      menu.style.top = `${r.bottom + 4}px`;
      menu.style.left = `${Math.max(8, r.right - menu.offsetWidth)}px`;
    }
  });
  $("run-menu").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-run]");
    if (!b) return;
    $("run-menu").hidden = true;
    runSim(b.dataset.run);
  });
  document.addEventListener("click", (e) => {
    if (!e.target.closest("#run-menu")) $("run-menu").hidden = true;
  });
})();
