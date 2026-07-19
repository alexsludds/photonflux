/* Schematic symbol library.
 *
 * Each symbol: { w, h, pins: {name: [x, y]}, draw() -> inner-SVG string,
 *               label: [x, y] (refdes anchor), pinLabels: bool }
 * Coordinates are component-local, grid pitch 20. The renderer places port
 * circles at pins (colored by domain from the backend catalog), applies
 * rotation about the bbox center, and draws refdes/value labels.
 */
"use strict";

const S = {}; // type id -> symbol

const OPT = "body-opt"; // amber stroke class
const EL = "body";      // default stroke class

// Subcircuit boundary port: a tag whose single pin `p` (on the right) wires to
// the internal net exposed as the named external port. Only used on definition
// sheets; the flattener splices it away. The renderer overlays the port name.
S.port = {
  w: 40, h: 20, pins: { p: [40, 10] }, label: [2, -4],
  draw: () => `
    <path class="${EL} body-fill" d="M2 4 h20 l8 6 l-8 6 h-20 z"/>
    <line class="${EL}" x1="30" y1="10" x2="40" y2="10"/>`,
};

S.ground = {
  w: 40, h: 30, pins: { p1: [20, 0] }, label: [24, 26], hideRef: true,
  draw: () => `
    <line class="${EL}" x1="20" y1="0" x2="20" y2="12"/>
    <line class="${EL}" x1="8"  y1="12" x2="32" y2="12"/>
    <line class="${EL}" x1="13" y1="18" x2="27" y2="18"/>
    <line class="${EL}" x1="18" y1="24" x2="22" y2="24"/>`,
};

S.cw_laser = {
  w: 80, h: 40, pins: { p1: [80, 20], p2: [40, 40] }, label: [0, -6],
  draw: () => `
    <rect class="${EL} body-fill" x="0" y="0" width="64" height="40" rx="5"/>
    <path class="${OPT}" d="M8 20 q6 -9 12 0 t12 0 t12 0"/>
    <path class="${OPT}" d="M50 20 h10 m0 0 l-5 -4 m5 4 l-5 4"/>
    <line class="${OPT}" x1="64" y1="20" x2="80" y2="20"/>
    <line class="${EL}" x1="40" y1="40" x2="40" y2="36"/>
    <text x="8" y="34">CW</text>`,
};

// DML lasers: box with laser-diode triangle, electrical drive on the left,
// optical out on the right, gnd (power-node reference) at the bottom.
function dmlGlyph(tag, extra) {
  return `
    <rect class="${EL} body-fill" x="12" y="2" width="52" height="56" rx="5"/>
    <line class="${EL}" x1="0" y1="20" x2="20" y2="20"/>
    <line class="${EL}" x1="0" y1="40" x2="20" y2="40"/>
    <path class="${EL}" d="M20 14 L 20 26 L 30 20 Z" fill="none" stroke-width="1.1"/>
    <line class="${EL}" x1="30" y1="14" x2="30" y2="26" stroke-width="1.1"/>
    <line class="${EL}" x1="20" y1="20" x2="20" y2="40" stroke-width="1.1"/>
    <path class="${OPT}" d="M36 30 h16 m0 0 l-5 -4 m5 4 l-5 4"/>
    <line class="${OPT}" x1="64" y1="30" x2="80" y2="30"/>
    <line class="${EL}" x1="40" y1="58" x2="40" y2="60"/>
    <text x="36" y="52" style="font-size:8px">${tag}</text>
    ${extra || ""}`;
}

S.laser_dml = {
  w: 80, h: 60,
  pins: { an: [0, 20], cat: [0, 40], pout: [80, 30], gnd: [40, 60] },
  label: [12, -6], pinLabels: true,
  draw: () => dmlGlyph("DML"),
};

S.laser_rate = {
  w: 80, h: 60,
  pins: { an: [0, 20], cat: [0, 40], pout: [80, 30], gnd: [40, 60] },
  label: [12, -6], pinLabels: true,
  // damped-ringing mark: relaxation oscillations are the point of this model
  draw: () => dmlGlyph("RATE",
    `<path class="${OPT}" d="M38 12 q3 -8 6 0 q2 5 4 0 q1.5 3 3 0 h6"
       fill="none" stroke-width="1.1"/>`),
};

S.mzm = {
  w: 100, h: 60, pins: { pin: [0, 30], pout: [100, 30], vp: [50, 0], vn: [50, 60] },
  label: [0, -6],
  draw: () => `
    <line class="${OPT}" x1="0" y1="30" x2="16" y2="30"/>
    <path class="${OPT}" d="M16 30 C 26 30 26 18 38 18 L 62 18 C 74 18 74 30 84 30"/>
    <path class="${OPT}" d="M16 30 C 26 30 26 42 38 42 L 62 42 C 74 42 74 30 84 30"/>
    <line class="${OPT}" x1="84" y1="30" x2="100" y2="30"/>
    <rect class="${EL}" x="40" y="10" width="20" height="5"/>
    <rect class="${EL}" x="40" y="45" width="20" height="5"/>
    <line class="${EL}" x1="50" y1="0" x2="50" y2="10"/>
    <line class="${EL}" x1="50" y1="50" x2="50" y2="60"/>`,
};

S.iq_modulator = {
  // nested MZM: an I child MZM (top) and Q child MZM (bottom) in parallel,
  // each with its own differential drive; a 90-degree combiner on the right
  w: 120, h: 80,
  pins: { pin: [0, 40], pout: [120, 40],
          vip: [40, 0], vin: [20, 0], vqp: [40, 80], vqn: [20, 80] },
  label: [0, -6], pinLabels: true,
  draw: () => `
    <line class="${OPT}" x1="0" y1="40" x2="14" y2="40"/>
    <path class="${OPT}" d="M14 40 C 22 40 22 20 34 20 L 66 20 C 78 20 78 40 86 40"/>
    <path class="${OPT}" d="M14 40 C 22 40 22 60 34 60 L 66 60 C 78 60 78 40 86 40"/>
    <path class="${OPT}" d="M34 20 C 44 20 44 12 54 12 L 60 12 C 70 12 70 20 78 20" opacity="0.85"/>
    <path class="${OPT}" d="M34 20 C 44 20 44 28 54 28 L 60 28 C 70 28 70 20 78 20" opacity="0.85"/>
    <path class="${OPT}" d="M34 60 C 44 60 44 52 54 52 L 60 52 C 70 52 70 60 78 60" opacity="0.85"/>
    <path class="${OPT}" d="M34 60 C 44 60 44 68 54 68 L 60 68 C 70 68 70 60 78 60" opacity="0.85"/>
    <line class="${OPT}" x1="86" y1="40" x2="120" y2="40"/>
    <rect class="${EL}" x="18" y="14" width="26" height="5"/>
    <line class="${EL}" x1="20" y1="0" x2="20" y2="14"/>
    <line class="${EL}" x1="40" y1="0" x2="40" y2="14"/>
    <rect class="${EL}" x="18" y="61" width="26" height="5"/>
    <line class="${EL}" x1="20" y1="66" x2="20" y2="80"/>
    <line class="${EL}" x1="40" y1="66" x2="40" y2="80"/>
    <text x="92" y="34" style="font-size:12px" class="${EL}">IQ</text>`,
};

S.mzm_tw = {
  // like the MZM but with segmented traveling-wave electrodes + gnd pin
  w: 100, h: 60,
  pins: { pin: [0, 30], pout: [100, 30], vp: [40, 0], vn: [40, 60], gnd: [80, 0] },
  label: [0, -6], pinLabels: true,
  draw: () => `
    <line class="${OPT}" x1="0" y1="30" x2="16" y2="30"/>
    <path class="${OPT}" d="M16 30 C 26 30 26 18 38 18 L 62 18 C 74 18 74 30 84 30"/>
    <path class="${OPT}" d="M16 30 C 26 30 26 42 38 42 L 62 42 C 74 42 74 30 84 30"/>
    <line class="${OPT}" x1="84" y1="30" x2="100" y2="30"/>
    <rect class="${EL}" x="34" y="10" width="8" height="5"/>
    <rect class="${EL}" x="46" y="10" width="8" height="5"/>
    <rect class="${EL}" x="58" y="10" width="8" height="5"/>
    <rect class="${EL}" x="34" y="45" width="8" height="5"/>
    <rect class="${EL}" x="46" y="45" width="8" height="5"/>
    <rect class="${EL}" x="58" y="45" width="8" height="5"/>
    <line class="${EL}" x1="40" y1="0" x2="40" y2="10"/>
    <line class="${EL}" x1="40" y1="50" x2="40" y2="60"/>
    <line class="${EL}" x1="80" y1="0" x2="80" y2="8"/>
    <line class="${EL}" x1="80" y1="8" x2="66" y2="12" stroke-width="1"/>`,
};

S.pulse_mod = {
  w: 80, h: 40, pins: { pin: [0, 20], pout: [80, 20] }, label: [0, -6],
  draw: () => `
    <line class="${OPT}" x1="0" y1="20" x2="10" y2="20"/>
    <rect class="${EL} body-fill" x="10" y="2" width="60" height="36" rx="4"/>
    <path class="${EL}" d="M20 30 h8 v-14 h14 v14 h8" stroke-width="1.3"/>
    <line class="${OPT}" x1="70" y1="20" x2="80" y2="20"/>`,
};

S.phase_shifter = {
  // straight waveguide through an electrode box: voltage rotates the field
  // phase (phi), magnitude untouched. vp electrode on top, vn on bottom,
  // gnd stub top-right.
  w: 100, h: 60,
  pins: { pin: [0, 30], pout: [100, 30], vp: [40, 0], vn: [40, 60], gnd: [70, 0] },
  label: [0, -6], pinLabels: true,
  draw: () => `
    <line class="${OPT}" x1="0" y1="30" x2="100" y2="30"/>
    <rect class="${OPT} body-fill" x="28" y="18" width="44" height="24"/>
    <text x="42" y="36" style="font-size:14px">&#966;</text>
    <rect class="${EL}" x="30" y="13" width="20" height="5"/>
    <rect class="${EL}" x="30" y="42" width="20" height="5"/>
    <line class="${EL}" x1="40" y1="0" x2="40" y2="13"/>
    <line class="${EL}" x1="40" y1="47" x2="40" y2="60"/>
    <line class="${EL}" x1="70" y1="0" x2="70" y2="18"/>`,
};

S.ring_mod = {
  w: 100, h: 80,
  pins: { pin: [0, 70], pout: [100, 70], vp: [20, 0], vn: [50, 0], gnd: [80, 0] },
  label: [0, 96], pinLabels: true,
  draw: () => `
    <line class="${OPT}" x1="0" y1="70" x2="100" y2="70"/>
    <circle class="${OPT}" cx="50" cy="44" r="20"/>
    <path class="${EL}" d="M20 0 v14 a36 36 0 0 1 60 0 v-14" fill="none" stroke-width="1.2"/>
    <line class="${EL}" x1="50" y1="0" x2="50" y2="12"/>`,
};

S.ring_mod_inj = {
  // like ring_mod but with a forward-diode mark inside the ring
  w: 100, h: 80,
  pins: { pin: [0, 70], pout: [100, 70], vp: [20, 0], vn: [50, 0], gnd: [80, 0] },
  label: [0, 96], pinLabels: true,
  draw: () => `
    <line class="${OPT}" x1="0" y1="70" x2="100" y2="70"/>
    <circle class="${OPT}" cx="50" cy="44" r="20"/>
    <path class="${EL}" d="M44 38 L 44 50 L 54 44 Z" fill="none" stroke-width="1"/>
    <line class="${EL}" x1="54" y1="38" x2="54" y2="50" stroke-width="1"/>
    <path class="${EL}" d="M20 0 v14 a36 36 0 0 1 60 0 v-14" fill="none" stroke-width="1.2"/>
    <line class="${EL}" x1="50" y1="0" x2="50" y2="12"/>`,
};

S.mzm_seg = {
  w: 120, h: 70,
  pins: { pin: [0, 35], pout: [120, 35],
          vp1: [30, 0], vn1: [30, 70], vp2: [60, 0], vn2: [60, 70],
          vp3: [85, 0], vn3: [85, 70], gnd: [105, 0] },
  label: [0, -6], pinLabels: true,
  draw: () => `
    <line class="${OPT}" x1="0" y1="35" x2="14" y2="35"/>
    <path class="${OPT}" d="M14 35 C 24 35 24 23 34 23 L 92 23 C 102 23 102 35 108 35"/>
    <path class="${OPT}" d="M14 35 C 24 35 24 47 34 47 L 92 47 C 102 47 102 35 108 35"/>
    <line class="${OPT}" x1="108" y1="35" x2="120" y2="35"/>
    <rect class="${EL}" x="20" y="15" width="20" height="4"/>
    <rect class="${EL}" x="52" y="15" width="12" height="4"/>
    <rect class="${EL}" x="80" y="15" width="8" height="4"/>
    <rect class="${EL}" x="20" y="51" width="20" height="4"/>
    <rect class="${EL}" x="52" y="51" width="12" height="4"/>
    <rect class="${EL}" x="80" y="51" width="8" height="4"/>
    <line class="${EL}" x1="30" y1="0" x2="30" y2="15"/>
    <line class="${EL}" x1="60" y1="0" x2="60" y2="15"/>
    <line class="${EL}" x1="85" y1="0" x2="85" y2="15"/>
    <line class="${EL}" x1="30" y1="55" x2="30" y2="70"/>
    <line class="${EL}" x1="60" y1="55" x2="60" y2="70"/>
    <line class="${EL}" x1="85" y1="55" x2="85" y2="70"/>
    <line class="${EL}" x1="105" y1="0" x2="105" y2="10"/>`,
};

S.waveguide = {
  // integrated (on-chip) strip waveguide: a tapered ridge with the guided
  // mode running down the core. Deliberately unlike the fibre spool
  // (fiber_cd) so the two read differently at a glance.
  w: 80, h: 20, pins: { p1: [0, 10], p2: [80, 10] }, label: [16, -8],
  draw: () => `
    <line class="${OPT}" x1="0" y1="10" x2="16" y2="10"/>
    <path class="${OPT} body-fill" d="M16 10 L24 3 L56 3 L64 10 L56 17 L24 17 Z"/>
    <line class="${OPT}" x1="21" y1="10" x2="59" y2="10" stroke-width="2.4"/>
    <line class="${OPT}" x1="64" y1="10" x2="80" y2="10"/>`,
};

S.splitter = {
  w: 60, h: 40, pins: { p1: [0, 20], p2: [60, 0], p3: [60, 40] }, label: [16, 56],
  draw: () => `
    <path class="${OPT}" d="M0 20 h14 C 34 20 34 0 54 0 L 60 0"/>
    <path class="${OPT}" d="M14 20 C 34 20 34 40 54 40 L 60 40"/>`,
};

S.dir_coupler = {
  // model semantics: inputs p1/p2 on the left, bar outputs straight across
  // (p1 -> p3, p2 -> p4), cross-coupling in the middle
  w: 80, h: 40, pins: { p1: [0, 0], p2: [0, 40], p3: [80, 0], p4: [80, 40] },
  label: [22, 56], pinLabels: true,
  draw: () => `
    <path class="${OPT}" d="M0 0 h10 C 30 0 30 14 40 14 C 50 14 50 0 70 0 L 80 0"/>
    <path class="${OPT}" d="M0 40 h10 C 30 40 30 26 40 26 C 50 26 50 40 70 40 L 80 40"/>`,
};

S.grating = {
  // grating teeth angled into a taper narrowing to the waveguide port
  w: 80, h: 40, pins: { grating: [0, 20], waveguide: [80, 20] }, label: [16, -6],
  draw: () => `
    <line class="${OPT}" x1="0" y1="20" x2="8" y2="20"/>
    <path class="${OPT}" d="M8 8 L 40 14 L 40 26 L 8 32 Z" fill="none"/>
    <line class="${OPT}" x1="14" y1="10" x2="14" y2="30" stroke-width="1.1"/>
    <line class="${OPT}" x1="20" y1="11" x2="20" y2="29" stroke-width="1.1"/>
    <line class="${OPT}" x1="26" y1="12" x2="26" y2="28" stroke-width="1.1"/>
    <line class="${OPT}" x1="32" y1="13" x2="32" y2="27" stroke-width="1.1"/>
    <line class="${OPT}" x1="40" y1="20" x2="80" y2="20"/>`,
};

S.opt_mirror = {
  // partial reflector: guide interrupted by an angled facet double-bar
  w: 60, h: 24, pins: { p1: [0, 12], p2: [60, 12] }, label: [10, -6],
  draw: () => `
    <line class="${OPT}" x1="0" y1="12" x2="25" y2="12"/>
    <line class="${OPT}" x1="35" y1="12" x2="60" y2="12"/>
    <line class="${OPT}" x1="26" y1="22" x2="34" y2="2" stroke-width="2.2"/>
    <line class="${OPT}" x1="31" y1="22" x2="39" y2="2" stroke-width="1.1"/>`,
};

S.opt_term = {
  // matched absorber: line into a filled wedge (like an RF termination)
  w: 40, h: 24, pins: { p1: [0, 12] }, label: [8, -6],
  draw: () => `
    <line class="${OPT}" x1="0" y1="12" x2="14" y2="12"/>
    <path class="${OPT}" d="M14 2 L 14 22 L 32 12 Z" fill="none"/>
    <line class="${OPT}" x1="17" y1="7" x2="17" y2="17" stroke-width="1.1"/>
    <line class="${OPT}" x1="21" y1="8.5" x2="21" y2="15.5" stroke-width="1.1"/>
    <line class="${OPT}" x1="25" y1="10" x2="25" y2="14" stroke-width="1.1"/>`,
};

S.opt_filter = {
  // tunable add-drop filter: flat-top passband glyph in a body, with a
  // tuning arrow; pin (left, in), thru (right, everything not dropped),
  // drop (bottom, the selected channel)
  w: 80, h: 60, pins: { pin: [0, 20], thru: [80, 20], drop: [40, 60] },
  label: [0, -6], pinLabels: true,
  draw: () => `
    <line class="${OPT}" x1="0" y1="20" x2="14" y2="20"/>
    <rect class="${EL} body-fill" x="14" y="4" width="52" height="32" rx="4"/>
    <path class="${OPT}" d="M20 30 L28 30 L33 12 L47 12 L52 30 L60 30"
      fill="none" stroke-width="1.4"/>
    <line class="${EL}" x1="24" y1="34" x2="56" y2="10" stroke-width="1"/>
    <path class="${EL}" d="M56 10 l-5 0 m5 0 l0 5" stroke-width="1"/>
    <line class="${OPT}" x1="66" y1="20" x2="80" y2="20"/>
    <line class="${OPT}" x1="40" y1="36" x2="40" y2="60"/>`,
};

// --- nonlinear / SOA / cavity building blocks (recent directed-wave VA
//     models: soa, ase_src, wmirror, ring_comb, ring_nl, wg_nl, ring_kerr) --

S.soa = {
  // bidirectional gain chip: tilted-facet active region (parallelogram),
  // forward rail fin->fout on top, backward rail bin->bout on the bottom,
  // gain chevrons pointing each way; electrical bias an/cat down from the top
  w: 110, h: 64,
  pins: { fin: [0, 22], bout: [0, 42], fout: [110, 22], bin: [110, 42],
          an: [42, 0], cat: [68, 0], gnd: [55, 64] },
  label: [0, -6], pinLabels: true,
  draw: () => `
    <line class="${OPT}" x1="0" y1="22" x2="27" y2="22"/>
    <line class="${OPT}" x1="0" y1="42" x2="31" y2="42"/>
    <line class="${OPT}" x1="83" y1="22" x2="110" y2="22"/>
    <line class="${OPT}" x1="79" y1="42" x2="110" y2="42"/>
    <path class="${OPT} body-fill" d="M26 12 L84 12 L92 52 L34 52 Z"/>
    <path class="${OPT}" d="M40 18 l6 4 l-6 4" fill="none" stroke-width="1.2"/>
    <path class="${OPT}" d="M50 18 l6 4 l-6 4" fill="none" stroke-width="1.2"/>
    <path class="${OPT}" d="M78 38 l-6 4 l6 4" fill="none" stroke-width="1.2"/>
    <path class="${OPT}" d="M68 38 l-6 4 l6 4" fill="none" stroke-width="1.2"/>
    <line class="${EL}" x1="42" y1="0" x2="42" y2="14"/>
    <line class="${EL}" x1="68" y1="0" x2="68" y2="14"/>
    <line class="${EL}" x1="55" y1="64" x2="55" y2="52"/>
    <text x="49" y="35" style="font-size:8px">SOA</text>`,
};

S.ase_src = {
  // in-line broadband ASE injector: the guide runs straight through and a
  // noise source stamps white field noise onto it
  w: 76, h: 44, pins: { pin: [0, 22], pout: [76, 22] }, label: [0, -6],
  draw: () => `
    <line class="${OPT}" x1="0" y1="22" x2="76" y2="22"/>
    <rect class="${EL} body-fill" x="14" y="6" width="48" height="32" rx="4"/>
    <path class="${OPT}" d="M17 22 l3 -7 l2 11 l3 -13 l2 9 l3 -11 l2 13 l3 -8 l2 5 l3 -9 l2 11 l3 -6 l2 5"
      fill="none" stroke-width="1.1"/>
    <text x="19" y="35" style="font-size:7px">ASE</text>`,
};

S.wmirror = {
  // 2x2 partial reflector on directed waves: a vertical half-mirror facet;
  // left port (li in / lo back), right port (ri in / ro through), arrows show
  // the directed waves
  w: 80, h: 60,
  pins: { li: [0, 22], lo: [0, 40], ri: [80, 22], ro: [80, 40], gnd: [40, 60] },
  label: [0, -6], pinLabels: true,
  draw: () => `
    <line class="${OPT}" x1="0" y1="22" x2="34" y2="22"/>
    <path class="${OPT}" d="M30 19 l5 3 l-5 3" fill="none" stroke-width="1.1"/>
    <line class="${OPT}" x1="0" y1="40" x2="34" y2="40"/>
    <path class="${OPT}" d="M10 37 l-5 3 l5 3" fill="none" stroke-width="1.1"/>
    <line class="${OPT}" x1="46" y1="22" x2="80" y2="22"/>
    <path class="${OPT}" d="M50 19 l-5 3 l5 3" fill="none" stroke-width="1.1"/>
    <line class="${OPT}" x1="46" y1="40" x2="80" y2="40"/>
    <path class="${OPT}" d="M70 37 l5 3 l-5 3" fill="none" stroke-width="1.1"/>
    <line class="${OPT}" x1="38" y1="12" x2="38" y2="50" stroke-width="2.2"/>
    <line class="${OPT}" x1="42" y1="12" x2="42" y2="50" stroke-width="1.1"/>
    <line class="${EL}" x1="40" y1="60" x2="40" y2="50"/>`,
};

S.circulator = {
  // non-reciprocal 3-port: circle with a clockwise circulation arrow (routes
  // 1 -> 2 -> 3 -> 1). Port 1 (TX) left, port 2 (shared line) right, port 3
  // (RX drop) bottom; each port splits into a directed input (arrow in) and
  // output (arrow out). gnd stub on top.
  w: 120, h: 120,
  pins: { p1i: [0, 40], p1o: [0, 80], p2o: [120, 40], p2i: [120, 80],
          p3o: [40, 120], p3i: [80, 120], gnd: [60, 0] },
  label: [0, -6], pinLabels: true,
  draw: () => `
    <circle class="${OPT} body-fill" cx="60" cy="60" r="36"/>
    <line class="${OPT}" x1="0" y1="40" x2="24" y2="40"/>
    <path class="${OPT}" d="M18 37 l6 3 l-6 3" fill="none" stroke-width="1.1"/>
    <line class="${OPT}" x1="0" y1="80" x2="24" y2="80"/>
    <path class="${OPT}" d="M6 77 l-6 3 l6 3" fill="none" stroke-width="1.1"/>
    <line class="${OPT}" x1="96" y1="40" x2="120" y2="40"/>
    <path class="${OPT}" d="M114 37 l6 3 l-6 3" fill="none" stroke-width="1.1"/>
    <line class="${OPT}" x1="96" y1="80" x2="120" y2="80"/>
    <path class="${OPT}" d="M102 77 l-6 3 l6 3" fill="none" stroke-width="1.1"/>
    <line class="${OPT}" x1="40" y1="96" x2="40" y2="120"/>
    <path class="${OPT}" d="M37 114 l3 6 l3 -6" fill="none" stroke-width="1.1"/>
    <line class="${OPT}" x1="80" y1="96" x2="80" y2="120"/>
    <path class="${OPT}" d="M77 102 l3 -6 l3 6" fill="none" stroke-width="1.1"/>
    <path class="${OPT}" d="M76 46 A 22 22 0 1 1 46 44" fill="none" stroke-width="1.5"/>
    <path class="${OPT}" d="M76 46 l-9 -1 l5 8 z" stroke-width="1"/>
    <line class="${EL}" x1="60" y1="0" x2="60" y2="24"/>
    <text x="11" y="34" style="font-size:8px">1</text>
    <text x="103" y="34" style="font-size:8px">2</text>
    <text x="52" y="116" style="font-size:8px">3</text>`,
};

// shared add-drop ring body: in/thru bus on top, ring, drop bus on the
// bottom whose far (add) port is dark -> a small absorber wedge
function addDropRingBase() {
  return `
    <line class="${OPT}" x1="0" y1="26" x2="104" y2="26"/>
    <circle class="${OPT}" cx="52" cy="46" r="18"/>
    <line class="${OPT}" x1="0" y1="66" x2="90" y2="66"/>
    <path class="${OPT}" d="M92 60 L92 72 L104 66 Z" fill="none"/>`;
}

S.ring_comb = {
  // add-drop ring FILTER with a resistive heater (the Vernier building
  // block): the heater resistor sits over the ring, hp/hn on top
  w: 104, h: 92,
  pins: { pin: [0, 26], thru: [104, 26], drop: [0, 66],
          hp: [38, 0], hn: [66, 0], gnd: [88, 0] },
  label: [0, 108], pinLabels: true,
  draw: () => addDropRingBase() + `
    <line class="${EL}" x1="38" y1="0" x2="38" y2="12"/>
    <line class="${EL}" x1="66" y1="0" x2="66" y2="12"/>
    <path class="${EL}" d="M38 12 h4 l3 -6 l4 12 l4 -12 l4 12 l3 -6 h6"
      fill="none" stroke-width="1.1"/>
    <line class="${EL}" x1="88" y1="0" x2="88" y2="12"/>`,
};

S.ring_kerr = {
  // add-drop Kerr FWM ring: no heater; a little frequency-comb glyph over
  // the ring marks the chi(3) comb it seeds
  w: 104, h: 92,
  pins: { pin: [0, 26], thru: [104, 26], drop: [0, 66], gnd: [88, 0] },
  label: [0, 108], pinLabels: true,
  draw: () => addDropRingBase() + `
    <line class="${OPT}" x1="36" y1="16" x2="68" y2="16" stroke-width="0.8"/>
    <line class="${OPT}" x1="40" y1="16" x2="40" y2="10"/>
    <line class="${OPT}" x1="46" y1="16" x2="46" y2="5"/>
    <line class="${OPT}" x1="52" y1="16" x2="52" y2="2"/>
    <line class="${OPT}" x1="58" y1="16" x2="58" y2="5"/>
    <line class="${OPT}" x1="64" y1="16" x2="64" y2="10"/>
    <line class="${EL}" x1="88" y1="0" x2="88" y2="12"/>
    <text x="69" y="52" style="font-size:8px">&#967;&#179;</text>`,
};

S.ring_nl = {
  // all-pass nonlinear ring (TPA/FCA-limited Q): single bus + ring, no
  // electrodes; two up-arrows (two-photon absorption) inside
  w: 84, h: 72, pins: { pin: [0, 52], pout: [84, 52], gnd: [42, 0] },
  label: [0, -6], pinLabels: true,
  draw: () => `
    <line class="${OPT}" x1="0" y1="52" x2="84" y2="52"/>
    <circle class="${OPT}" cx="42" cy="32" r="18"/>
    <line class="${EL}" x1="42" y1="0" x2="42" y2="14"/>
    <path class="${OPT}" d="M37 46 l0 -9 m-2 3 l2 -3 l2 3" fill="none" stroke-width="1"/>
    <path class="${OPT}" d="M47 46 l0 -9 m-2 3 l2 -3 l2 3" fill="none" stroke-width="1"/>
    <text x="33" y="30" style="font-size:7px">TPA</text>`,
};

S.ring_selfheat = {
  // all-pass self-heating ring (thermo-optic bistability): single bus + ring,
  // a wavelength-drive electrical pin `lam` and the thermal-reference `gnd` on
  // top, with a little heat-wave glyph marking the self-heating loop
  w: 84, h: 72, pins: { pin: [0, 52], pout: [84, 52], lam: [30, 0], gnd: [54, 0] },
  label: [0, -6], pinLabels: true,
  draw: () => `
    <line class="${OPT}" x1="0" y1="52" x2="84" y2="52"/>
    <circle class="${OPT}" cx="42" cy="32" r="18"/>
    <line class="${EL}" x1="30" y1="0" x2="30" y2="14"/>
    <line class="${EL}" x1="54" y1="0" x2="54" y2="14"/>
    <path class="${OPT}" d="M36 34 q3 -5 6 0 q3 5 6 0" fill="none" stroke-width="1"/>
    <path class="${OPT}" d="M36 40 q3 -5 6 0 q3 5 6 0" fill="none" stroke-width="1"/>
    <text x="31" y="30" style="font-size:7px">&#916;T</text>`,
};

S.wg_nl = {
  // nonlinear waveguide segment (Kerr + TPA/FCA): the strip-ridge of the
  // linear waveguide, marked chi(3), with the carrier-reference gnd stub
  w: 90, h: 44, pins: { pin: [0, 20], pout: [90, 20], gnd: [45, 44] },
  label: [0, -6], pinLabels: true,
  draw: () => `
    <line class="${OPT}" x1="0" y1="20" x2="16" y2="20"/>
    <path class="${OPT} body-fill" d="M16 20 L24 12 L66 12 L74 20 L66 28 L24 28 Z"/>
    <line class="${OPT}" x1="22" y1="20" x2="68" y2="20" stroke-width="2.4"/>
    <line class="${OPT}" x1="74" y1="20" x2="90" y2="20"/>
    <line class="${EL}" x1="45" y1="28" x2="45" y2="44"/>
    <text x="33" y="9" style="font-size:8px">&#967;&#179;</text>`,
};

S.photodiode = {
  w: 60, h: 60, pins: { po_p: [0, 20], po_n: [0, 40], cat: [60, 20], an: [60, 40] },
  label: [8, -6], pinLabels: true,
  draw: () => `
    <line class="${OPT}" x1="0" y1="20" x2="18" y2="20"/>
    <line class="${OPT}" x1="0" y1="40" x2="18" y2="40"/>
    <path class="${OPT}" d="M6 26 l8 6 m-8 -6 l3 5 m-3 -5 l5 2" stroke-width="1.1"/>
    <path class="${EL}" d="M24 16 L 24 44 L 44 30 Z" fill="none"/>
    <line class="${EL}" x1="44" y1="18" x2="44" y2="42"/>
    <line class="${EL}" x1="44" y1="20" x2="60" y2="20"/>
    <line class="${EL}" x1="44" y1="40" x2="60" y2="40"/>`,
};

S.coherent_rx = {
  // 90-degree hybrid (sig + LO) into two balanced PD pairs -> I and Q currents
  w: 80, h: 80,
  pins: { sig: [0, 24], lo: [0, 56],
          i_p: [80, 12], i_n: [80, 30], q_p: [80, 50], q_n: [80, 68] },
  label: [8, -6], pinLabels: true,
  draw: () => `
    <line class="${OPT}" x1="0" y1="24" x2="16" y2="24"/>
    <line class="${OPT}" x1="0" y1="56" x2="16" y2="56"/>
    <rect class="${OPT}" x="16" y="14" width="30" height="52" rx="3" fill="none"/>
    <text x="20" y="44" style="font-size:10px" class="${OPT}">90&#176;</text>
    <path class="${EL}" d="M50 8 L 50 24 L 62 16 Z" fill="none"/>
    <line class="${EL}" x1="62" y1="10" x2="62" y2="22"/>
    <line class="${EL}" x1="62" y1="12" x2="80" y2="12"/>
    <line class="${EL}" x1="62" y1="30" x2="80" y2="30"/>
    <line class="${EL}" x1="46" y1="16" x2="50" y2="16"/>
    <path class="${EL}" d="M50 46 L 50 62 L 62 54 Z" fill="none"/>
    <line class="${EL}" x1="62" y1="48" x2="62" y2="60"/>
    <line class="${EL}" x1="62" y1="50" x2="80" y2="50"/>
    <line class="${EL}" x1="62" y1="68" x2="80" y2="68"/>
    <line class="${EL}" x1="46" y1="54" x2="50" y2="54"/>`,
};

// --- electrical two-terminal glyphs (drawn horizontal, pins left/right) ----

S.resistor = {
  w: 60, h: 20, pins: { p1: [0, 10], p2: [60, 10] }, label: [12, -4],
  draw: () => `
    <path class="${EL}" d="M0 10 h10 l4 -7 l7 14 l7 -14 l7 14 l7 -14 l4 7 h14"/>`,
};

S.capacitor = {
  w: 60, h: 24, pins: { p1: [0, 12], p2: [60, 12] }, label: [14, -4],
  draw: () => `
    <line class="${EL}" x1="0" y1="12" x2="26" y2="12"/>
    <line class="${EL}" x1="26" y1="0" x2="26" y2="24"/>
    <line class="${EL}" x1="34" y1="0" x2="34" y2="24"/>
    <line class="${EL}" x1="34" y1="12" x2="60" y2="12"/>`,
};

S.inductor = {
  w: 60, h: 20, pins: { p1: [0, 10], p2: [60, 10] }, label: [12, -4],
  draw: () => `
    <path class="${EL}" d="M0 10 h8 a6 6 0 0 1 12 0 a6 6 0 0 1 12 0 a6 6 0 0 1 12 0 a6 6 0 0 1 12 0 h4"
      fill="none"/>`,
};

S.diode = {
  w: 60, h: 24, pins: { p1: [0, 12], p2: [60, 12] }, label: [14, -4],
  draw: () => `
    <line class="${EL}" x1="0" y1="12" x2="22" y2="12"/>
    <path class="${EL}" d="M22 2 L 22 22 L 38 12 Z" fill="none"/>
    <line class="${EL}" x1="38" y1="2" x2="38" y2="22"/>
    <line class="${EL}" x1="38" y1="12" x2="60" y2="12"/>`,
};

// --- sources (vertical, p1 top / p2 bottom) --------------------------------

function srcGlyph(inner) {
  return `
    <line class="${EL}" x1="20" y1="0" x2="20" y2="10"/>
    <circle class="${EL} body-fill" cx="20" cy="30" r="18"/>
    <line class="${EL}" x1="20" y1="50" x2="20" y2="60"/>
    ${inner}`;
}

S.vdc = {
  w: 40, h: 60, pins: { p1: [20, 0], p2: [20, 60] }, label: [44, 26],
  draw: () => srcGlyph(`
    <line class="${EL}" x1="20" y1="20" x2="20" y2="28" stroke-width="1.3"/>
    <line class="${EL}" x1="16" y1="24" x2="24" y2="24" stroke-width="1.3"/>
    <line class="${EL}" x1="16" y1="38" x2="24" y2="38" stroke-width="1.3"/>`),
};

S.vpulse = {
  w: 40, h: 60, pins: { p1: [20, 0], p2: [20, 60] }, label: [44, 26],
  draw: () => srcGlyph(`
    <path class="${EL}" d="M10 36 h5 v-12 h10 v12 h5" stroke-width="1.3"/>`),
};

S.vsin = {
  w: 40, h: 60, pins: { p1: [20, 0], p2: [20, 60] }, label: [44, 26],
  draw: () => srcGlyph(`
    <path class="${EL}" d="M10 30 q5 -12 10 0 t10 0" stroke-width="1.3"/>`),
};

S.prbs = {
  w: 40, h: 60, pins: { p1: [20, 0], p2: [20, 60] }, label: [44, 26],
  draw: () => srcGlyph(`
    <path class="${EL}" d="M9 36 h4 v-12 h4 v12 h8 v-12 h4 v12 h2"
      stroke-width="1.2"/>
    <text x="12" y="46" style="font-size:7px">PRBS</text>`),
};

S.vpwl = {
  w: 40, h: 60, pins: { p1: [20, 0], p2: [20, 60] }, label: [44, 26],
  draw: () => srcGlyph(`
    <path class="${EL}" d="M9 38 l6 -14 l5 8 l6 -6 l5 12" stroke-width="1.2"
      fill="none"/>`),
};

S.idc = {
  w: 40, h: 60, pins: { p1: [20, 0], p2: [20, 60] }, label: [44, 26],
  draw: () => srcGlyph(`
    <path class="${EL}" d="M20 40 V 20 m0 0 l-4 6 m4 -6 l4 6" stroke-width="1.3"/>`),
};

// --- transistors ------------------------------------------------------------

function fetGlyph(arrowUp, bulk) {
  // gate at (0,30), d (40,0), s (40,60), optional b (60,30)
  const arrow = arrowUp
    ? `<path class="${EL}" d="M26 44 l8 0 m-8 0 l5 -4 m-5 4 l5 4" stroke-width="1.2"/>`
    : `<path class="${EL}" d="M34 44 l-8 0 m8 0 l-5 -4 m5 4 l5 4" stroke-width="1.2"/>`;
  return `
    <line class="${EL}" x1="0" y1="30" x2="18" y2="30"/>
    <line class="${EL}" x1="18" y1="16" x2="18" y2="44"/>
    <line class="${EL}" x1="24" y1="14" x2="24" y2="46"/>
    <line class="${EL}" x1="24" y1="16" x2="40" y2="16"/>
    <line class="${EL}" x1="40" y1="0"  x2="40" y2="16"/>
    <line class="${EL}" x1="24" y1="44" x2="40" y2="44"/>
    <line class="${EL}" x1="40" y1="44" x2="40" y2="60"/>
    ${arrow}
    ${bulk ? `<line class="${EL}" x1="24" y1="30" x2="60" y2="30"/>` : ""}`;
}

S.nmos = {
  w: 60, h: 60, pins: { g: [0, 30], d: [40, 0], s: [40, 60] },
  label: [46, 30], pinLabels: true,
  draw: () => fetGlyph(true, false),
};
S.pmos = {
  w: 60, h: 60, pins: { g: [0, 30], d: [40, 0], s: [40, 60] },
  label: [46, 30], pinLabels: true,
  draw: () => fetGlyph(false, false) +
    `<circle class="${EL}" cx="14" cy="30" r="3" fill="none"/>`,
};
S.sky130_nfet = {
  w: 60, h: 60, pins: { g: [0, 30], d: [40, 0], s: [40, 60], b: [60, 30] },
  label: [8, -6], pinLabels: true,
  draw: () => fetGlyph(true, true),
};
S.sky130_pfet = {
  w: 60, h: 60, pins: { g: [0, 30], d: [40, 0], s: [40, 60], b: [60, 30] },
  label: [8, -6], pinLabels: true,
  draw: () => fetGlyph(false, true) +
    `<circle class="${EL}" cx="14" cy="30" r="3" fill="none"/>`,
};
// flavor variants share the 4-terminal glyphs
for (const t of ["sky130_nfet_lvt", "sky130_nfet_5v", "sky130_nfet_nvt"]) {
  S[t] = S.sky130_nfet;
}
for (const t of ["sky130_pfet_lvt", "sky130_pfet_hvt", "sky130_pfet_5v"]) {
  S[t] = S.sky130_pfet;
}

// PDK resistors: IEC box style (vs. the zigzag of the ideal resistor)
const pdkRes = {
  w: 60, h: 20, pins: { p1: [0, 10], p2: [60, 10] }, label: [8, -4],
  draw: () => `
    <line class="${EL}" x1="0" y1="10" x2="14" y2="10"/>
    <rect class="${EL}" x="14" y="4" width="32" height="12" fill="none"/>
    <line class="${EL}" x1="46" y1="10" x2="60" y2="10"/>`,
};
S.sky130_res_po = pdkRes;
S.sky130_res_nd = pdkRes;
S.sky130_res_high_po = pdkRes;
S.sky130_res_xhigh_po = pdkRes;

// PDK MiM caps: one straight + one bracketed plate
const pdkCap = {
  w: 60, h: 24, pins: { p1: [0, 12], p2: [60, 12] }, label: [10, -4],
  draw: () => `
    <line class="${EL}" x1="0" y1="12" x2="26" y2="12"/>
    <line class="${EL}" x1="26" y1="0" x2="26" y2="24"/>
    <path class="${EL}" d="M38 0 h-4 v24 h4" fill="none"/>
    <line class="${EL}" x1="34" y1="12" x2="60" y2="12"/>`,
};
S.sky130_cap_mim = pdkCap;
S.sky130_cap_mim2 = pdkCap;

// LTI channel blocks: box with a rolling-off response curve
function chanGlyph(tag) {
  return {
    w: 80, h: 40, pins: { inp: [0, 20], out: [80, 20] },
    label: [20, -6], pinLabels: true,
    draw: () => `
      <rect class="${EL} body-fill" x="12" y="2" width="56" height="36" rx="5"/>
      <line class="${EL}" x1="0" y1="20" x2="12" y2="20"/>
      <line class="${EL}" x1="68" y1="20" x2="80" y2="20"/>
      <path class="${EL}" d="M18 12 h14 q10 0 14 8 q4 8 14 8" fill="none"
        stroke-width="1.3"/>
      <text x="20" y="34" style="font-size:7px">${tag}</text>`,
  };
}
S.channel = chanGlyph("CHAN");
S.s2p_channel = chanGlyph("S2P");

S.fiber_cd = {
  w: 80, h: 26, pins: { p1: [0, 13], p2: [80, 13] }, label: [14, -8],
  draw: () => `
    <line class="${OPT}" x1="0" y1="13" x2="20" y2="13"/>
    <circle class="${OPT}" cx="29" cy="13" r="8" stroke-width="1.3"/>
    <circle class="${OPT}" cx="40" cy="13" r="8" stroke-width="1.3"/>
    <circle class="${OPT}" cx="51" cy="13" r="8" stroke-width="1.3"/>
    <line class="${OPT}" x1="59" y1="13" x2="80" y2="13"/>
    <path class="${OPT}" d="M26 25 q4 -3 8 0 q4 3 8 0" fill="none"
      stroke-width="1" opacity="0.8"/>`,
};

S.fiber_nl = {
  w: 80, h: 26, pins: { p1: [0, 13], p2: [80, 13] }, label: [10, -8],
  draw: () => `
    <line class="${OPT}" x1="0" y1="13" x2="20" y2="13"/>
    <circle class="${OPT}" cx="29" cy="13" r="8" stroke-width="1.3"/>
    <circle class="${OPT}" cx="40" cy="13" r="8" stroke-width="1.3"/>
    <circle class="${OPT}" cx="51" cy="13" r="8" stroke-width="1.3"/>
    <line class="${OPT}" x1="59" y1="13" x2="80" y2="13"/>
    <text x="40" y="10" text-anchor="middle" class="${OPT}"
      style="font-size:8px" stroke="none">&#967;&#8323;</text>`,
};

S.raman_amp = {
  // Raman span: signal rail through the top (sin->sout), co-pump rail below
  // (pcin->pcout) and a counter-pump rail in the middle (pctin from the right,
  // pctout out the left); fibre coil in the body, gain chevron on the signal.
  w: 120, h: 74,
  pins: { sin: [0, 18], sout: [120, 18], pctout: [0, 37], pctin: [120, 37],
          pcin: [0, 56], pcout: [120, 56], gnd: [60, 74] },
  label: [0, -6], pinLabels: true,
  draw: () => `
    <rect class="${OPT} body-fill" x="28" y="8" width="64" height="58" rx="6"/>
    <line class="${OPT}" x1="0" y1="18" x2="28" y2="18"/>
    <line class="${OPT}" x1="92" y1="18" x2="120" y2="18"/>
    <path class="${OPT}" d="M52 14 l6 4 l-6 4" fill="none" stroke-width="1.2"/>
    <path class="${OPT}" d="M62 14 l6 4 l-6 4" fill="none" stroke-width="1.2"/>
    <line class="${OPT}" x1="0" y1="37" x2="28" y2="37" opacity="0.85"/>
    <line class="${OPT}" x1="92" y1="37" x2="120" y2="37" opacity="0.85"/>
    <path class="${OPT}" d="M68 33 l-6 4 l6 4" fill="none" stroke-width="1.2"/>
    <line class="${OPT}" x1="0" y1="56" x2="28" y2="56" opacity="0.85"/>
    <line class="${OPT}" x1="92" y1="56" x2="120" y2="56" opacity="0.85"/>
    <path class="${OPT}" d="M52 52 l6 4 l-6 4" fill="none" stroke-width="1.2"/>
    <line class="${EL}" x1="60" y1="74" x2="60" y2="66"/>
    <text x="60" y="40" text-anchor="middle" style="font-size:8px">Raman</text>`,
};

S.sbs_fiber = {
  // SBS span: forward pump rail on top (fin->fout, clamped), backward Stokes
  // rail on the bottom (bout back to the source, bin the seed); coil body.
  w: 96, h: 58,
  pins: { fin: [0, 18], fout: [96, 18], bout: [0, 40], bin: [96, 40],
          gnd: [48, 58] },
  label: [0, -6], pinLabels: true,
  draw: () => `
    <rect class="${OPT} body-fill" x="24" y="8" width="48" height="42" rx="6"/>
    <line class="${OPT}" x1="0" y1="18" x2="24" y2="18"/>
    <line class="${OPT}" x1="72" y1="18" x2="96" y2="18"/>
    <path class="${OPT}" d="M40 14 l6 4 l-6 4" fill="none" stroke-width="1.2"/>
    <line class="${OPT}" x1="0" y1="40" x2="24" y2="40" opacity="0.85"/>
    <line class="${OPT}" x1="72" y1="40" x2="96" y2="40" opacity="0.85"/>
    <path class="${OPT}" d="M32 36 l-6 4 l6 4" fill="none" stroke-width="1.2"/>
    <line class="${EL}" x1="48" y1="58" x2="48" y2="50"/>
    <text x="48" y="32" text-anchor="middle" style="font-size:8px">SBS</text>`,
};

S.tia = {
  w: 80, h: 50,
  pins: { inp: [0, 25], out: [80, 25] },
  label: [26, -6], pinLabels: true,
  draw: () => `
    <path class="${EL} body-fill" d="M14 4 L 14 46 L 66 25 Z"/>
    <line class="${EL}" x1="0" y1="25" x2="14" y2="25"/>
    <line class="${EL}" x1="66" y1="25" x2="80" y2="25"/>
    <path class="${EL}" d="M14 10 h-6 v-6 h24 l3 -4 l4 8 l4 -8 l4 8 l3 -4 h14 v10"
      fill="none" stroke-width="1" opacity="0.8"/>
    <text x="22" y="29" style="font-size:9px">TIA</text>`,
};

S.ctle = {
  w: 80, h: 50,
  pins: { inp: [0, 25], out: [80, 25] },
  label: [22, -6], pinLabels: true,
  draw: () => `
    <rect class="${EL} body-fill" x="12" y="4" width="56" height="42" rx="5"/>
    <line class="${EL}" x1="0" y1="25" x2="12" y2="25"/>
    <line class="${EL}" x1="68" y1="25" x2="80" y2="25"/>
    <path class="${EL}" d="M20 34 h10 q6 0 8 -8 q2 -8 8 -8 h6 q4 0 6 6 l2 6"
      fill="none" stroke-width="1.3"/>
    <text x="22" y="16" style="font-size:8px">CTLE</text>`,
};

S.rx_ffe = {
  w: 80, h: 50,
  pins: { inp: [0, 25], out: [80, 25] },
  label: [20, -6], pinLabels: true,
  draw: () => `
    <rect class="${EL} body-fill" x="12" y="4" width="56" height="42" rx="5"/>
    <line class="${EL}" x1="0" y1="25" x2="12" y2="25"/>
    <line class="${EL}" x1="68" y1="25" x2="80" y2="25"/>
    <line class="${EL}" x1="20" y1="36" x2="20" y2="26" stroke-width="1.3"/>
    <line class="${EL}" x1="30" y1="36" x2="30" y2="22" stroke-width="1.3"/>
    <line class="${EL}" x1="40" y1="36" x2="40" y2="28" stroke-width="1.3"/>
    <line class="${EL}" x1="50" y1="36" x2="50" y2="31" stroke-width="1.3"/>
    <line class="${EL}" x1="16" y1="36" x2="54" y2="36" stroke-width="1"/>
    <text x="22" y="16" style="font-size:8px">FFE</text>`,
};

S.rx_dfe = {
  w: 80, h: 50,
  pins: { inp: [0, 25], out: [80, 25] },
  label: [20, -6], pinLabels: true,
  draw: () => `
    <rect class="${EL} body-fill" x="12" y="4" width="56" height="42" rx="5"/>
    <line class="${EL}" x1="0" y1="25" x2="12" y2="25"/>
    <line class="${EL}" x1="68" y1="25" x2="80" y2="25"/>
    <circle class="${EL}" cx="34" cy="31" r="5" fill="none" stroke-width="1.2"/>
    <path class="${EL}" d="M39 31 h12 v7 h-26 v-7" fill="none"
      stroke-width="1" opacity="0.85"/>
    <text x="22" y="16" style="font-size:8px">DFE</text>`,
};

S.opamp = {
  w: 80, h: 60,
  pins: { in_m: [0, 20], in_p: [0, 40], out_p: [80, 30], out_m: [40, 60] },
  label: [26, -6],
  draw: () => `
    <path class="${EL} body-fill" d="M12 4 L 12 56 L 68 30 Z"/>
    <line class="${EL}" x1="0" y1="20" x2="12" y2="20"/>
    <line class="${EL}" x1="0" y1="40" x2="12" y2="40"/>
    <line class="${EL}" x1="68" y1="30" x2="80" y2="30"/>
    <line class="${EL}" x1="40" y1="43" x2="40" y2="60"/>
    <text x="17" y="24" style="font-size:11px">&#8722;</text>
    <text x="17" y="45" style="font-size:11px">+</text>`,
};

// value shown under the refdes: pick the headline parameter per type
const HEADLINE_PARAM = {
  cw_laser: "power", mzm: "vpi", iq_modulator: "vpi", coherent_rx: "R",
  pulse_mod: "p_on", ring_mod: "kappa2",
  laser_dml: "Ith", laser_rate: "taup", mzm_tw: "vpi",
  ring_mod_inj: "tau_c", mzm_seg: "vpi", ring_selfheat: "rth_k_w",
  phase_shifter: "vpi",
  waveguide: "length_m", splitter: "split_ratio", dir_coupler: "coupling",
  grating: "center_wavelength_nm", opt_filter: "center_nm", opt_mirror: "R",
  circulator: "iso_db",
  photodiode: "R", vdc: "V", vpulse: "v2", vsin: "V", idc: "I",
  resistor: "R", capacitor: "C", inductor: "L", diode: "Is",
  nmos: "W", pmos: "W", opamp: "A", tia: "gain_ohm", ctle: "peaking_db",
  rx_ffe: "n_taps", rx_dfe: "n_taps",
  channel: "loss_db", s2p_channel: "z0", fiber_cd: "length_km",
  fiber_nl: "gamma_per_W_km",
  sky130_nfet: "w_um", sky130_nfet_lvt: "w_um", sky130_nfet_5v: "w_um",
  sky130_nfet_nvt: "w_um", sky130_pfet: "w_um", sky130_pfet_lvt: "w_um",
  sky130_pfet_hvt: "w_um", sky130_pfet_5v: "w_um",
  sky130_res_po: "l_um", sky130_res_nd: "l_um",
  sky130_res_high_po: "l_um", sky130_res_xhigh_po: "l_um",
  sky130_cap_mim: "w_um", sky130_cap_mim2: "w_um",
};

/* SI-prefix number formatting: 1.5e-9 -> "1.5n" */
function fmtSI(v) {
  if (v === 0) return "0";
  if (!isFinite(v)) return String(v);
  const pref = [[1e12,"T"],[1e9,"G"],[1e6,"M"],[1e3,"k"],[1,""],
                [1e-3,"m"],[1e-6,"u"],[1e-9,"n"],[1e-12,"p"],[1e-15,"f"]];
  const a = Math.abs(v);
  for (const [s, p] of pref) {
    if (a >= s * 0.9995) {
      const m = v / s;
      const str = Math.abs(m) >= 100 ? m.toFixed(0)
                : Math.abs(m) >= 10 ? m.toFixed(1) : m.toFixed(2);
      return str.replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "") + p;
    }
  }
  return v.toExponential(1);
}

/* Like fmtSI but LOSSLESS: preserves significant figures so an editable
   parameter field round-trips through parseSI. fmtSI keeps ~3 sig figs, which
   collapses e.g. a laser wavelength 1308.285 nm to "1.31k" — visually the same
   as 1310 nm, hiding the 200 GHz WDM channel spacing, and re-parsing "1.31k"
   would corrupt the value to 1310. Here values in the plain human range render
   without a prefix at full precision (1308.285 -> "1308.285"); only far-from-
   unity magnitudes take an SI prefix, still at full precision (4e-3 -> "4m",
   50e9 -> "50G"). */
function fmtNum(v) {
  if (v === 0) return "0";
  if (!isFinite(v)) return String(v);
  const a = Math.abs(v);
  const trim = (x) => {
    let s = x.toPrecision(9);
    if (s.includes("e")) s = String(x);
    return s.replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
  };
  if (a >= 1 && a < 1e4) return trim(v);            // plain, full precision
  const pref = [[1e12, "T"], [1e9, "G"], [1e6, "M"], [1e3, "k"],
                [1e-3, "m"], [1e-6, "u"], [1e-9, "n"], [1e-12, "p"],
                [1e-15, "f"]];
  for (const [s, p] of pref) if (a >= s * 0.9995) return trim(v / s) + p;
  return v.toExponential(3);
}

/* parse "4n", "1.5u", "2e-9", "3" -> number (SPICE-style suffixes) */
function parseSI(s) {
  if (typeof s === "number") return s;
  s = String(s).trim().replace(/\s+/g, "");
  const m = s.match(/^([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)(meg|[TGMkKmunpf])?$/);
  if (!m) return NaN;
  let v = parseFloat(m[1]);
  const suf = m[2];
  if (suf) {
    const map = { T: 1e12, G: 1e9, M: 1e6, meg: 1e6, k: 1e3, K: 1e3,
                  m: 1e-3, u: 1e-6, n: 1e-9, p: 1e-12, f: 1e-15 };
    v *= map[suf];
  }
  return v;
}
