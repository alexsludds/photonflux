"""Source-level Verilog-A hierarchy flattening for the VA->JAX lowering.

Verilog-A allows a module to instantiate other modules, and ``ring_mod.va``
uses that to assemble the microring from its CMT sub-components
(``directional_coupler.va`` + ``ring_phase_shifter.va`` +
``ring_waveguide.va``). openvaf, however, *parses* child instances and then
silently drops them — the lowered top module keeps its ports and internal
nodes but none of the children's physics. This module closes that gap by
flattening the hierarchy at the source level before openvaf ever sees it:
each instance's analog block is inlined into the parent with

  * child ports        -> the parent nets they are connected to,
  * child parameters   -> the parenthesised override expression (or the
                          child's own default when not overridden),
  * child internal nodes, variables and localparams -> ``<inst>_``-prefixed
    copies declared in the parent,

and the parent's ``localparam real`` derivation chain (legal Verilog-A
constant expressions, so the hierarchical source stays portable to
simulators with native hierarchy) is rewritten into ordinary analog-block
assignments — the flat idiom bosdi's default parsing handles reliably (its
localparam handling needs the literal-default repair in ``cx.py``; see
``_va_literal_defaults`` there).

Because contribution statements accumulate (``<+``), concatenating the
inlined bodies reproduces exactly the Kirchhoff sums the hierarchy means:
elements sharing a node sum their currents into it.

Flat sources are never touched: a cheap instance detector gates the strict
parser, so a model without child instances passes through to openvaf exactly
as before hierarchy support existed, whatever constructs it uses.

Supported hierarchy subset for files that DO instantiate children (kept
deliberately small and checked loudly — a construct outside it raises rather
than miscompiling):

  * one module per file; a child's module name equals its file stem (the
    library convention, see ``models/README.md``), children resolved by
    globbing the models tree;
  * named port connections and named parameter overrides only
    (``mod #(.p(expr)) inst(.port(net), ...);``);
  * every child port must be connected;
  * a child parameter left unoverridden must have a pure numeric-literal
    default (anything else would capture parent names when inlined — raise);
  * declarations limited to ``inout``/``electrical``/``parameter real``/
    ``localparam real``/``real`` and at most one ``analog begin ... end``
    block per module; no branches, no strings, no functions.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

__all__ = ["flatten_va", "flatten_file"]

_KEYWORDS = frozenset({
    "module", "endmodule", "inout", "input", "output", "electrical",
    "parameter", "localparam", "real", "integer", "string", "analog",
    "branch", "aliasparam", "genvar", "begin", "end", "if", "else", "case",
    "endcase", "for", "while", "repeat", "generate", "endgenerate",
    "function", "endfunction", "task", "endtask", "initial", "assign",
})

_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_MODULE_RE = re.compile(r"\bmodule\s+(\w+)\s*\(([^)]*)\)\s*;")
_PARAM_RE = re.compile(
    r"^parameter\s+real\s+(\w+)\s*=\s*(.*?)\s*(?:\bfrom\s*[\[(].*)?$", re.S)
_LOCALPARAM_RE = re.compile(r"^localparam\s+real\s+(\w+)\s*=\s*(.*)$", re.S)
_INSTANCE_RE = re.compile(
    r"^(\w+)\s*(?:#\s*\((.*?)\))?\s*(\w+)\s*\((.*)\)$", re.S)
_NAMED_ITEM_RE = re.compile(r"^\.(\w+)\s*\((.*)\)$", re.S)
# a numeric literal, optionally with a Verilog-A scale suffix
_LITERAL_RE = re.compile(
    r"^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?\s*([TGMKkmunpfa])?$")
_SCALE = {"T": 1e12, "G": 1e9, "M": 1e6, "K": 1e3, "k": 1e3, "m": 1e-3,
          "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15, "a": 1e-18}


def _strip_comments(text: str) -> str:
    """Remove // and /* */ comments, preserving line structure."""
    return _COMMENT_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def _has_instances(stripped: str) -> bool:
    """Cheap conservative detector: does this source contain instance-shaped
    statements (``ident [#(...)] ident (...);`` with a non-keyword head)?

    This is the gate in front of the strict subset parser: flat models —
    whatever constructs they use — must pass through to openvaf untouched,
    so the flattener's "raise loudly" policy may only apply to files that
    actually contain hierarchy. No legal flat Verilog-A statement in scope
    has this shape (contributions start with an access function call,
    assignments have no trailing port list, declarations start with a
    keyword), and a typo'd child name still counts as an instance so it
    fails loudly at resolution instead of being dropped silently by openvaf.
    """
    for stmt in stripped.split(";"):
        m = _INSTANCE_RE.match(stmt.strip())
        if m and m.group(1) not in _KEYWORDS:
            return True
    return False


def _split_top_commas(text: str) -> list[str]:
    """Split on commas at parenthesis depth 0."""
    parts, depth, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


def _subst(text: str, mapping: dict[str, str]) -> str:
    """Simultaneous whole-identifier substitution (single pass, so replacement
    text is never itself re-substituted). Backtick/`$`/dot lookbehind keeps
    macros (`M_PI), system tasks and named-connection syntax untouched."""
    if not mapping:
        return text
    pat = re.compile(
        r"(?<![\w`$.])(" + "|".join(map(re.escape, sorted(mapping))) + r")\b")
    return pat.sub(lambda m: mapping[m.group(1)], text)


class _Module:
    """One parsed (comment-stripped, already-flattened) Verilog-A module."""

    def __init__(self, path: Path, name: str, ports: list[str]):
        self.path = path
        self.name = name
        self.ports = ports
        self.header: list[str] = []           # `include / `define lines
        self.port_decls: list[str] = []       # inout/electrical stmts on ports
        self.internal_nodes: list[str] = []   # electrical, not a port
        self.params: dict[str, str] = {}      # name -> default expression
        self.param_decls: list[str] = []      # full parameter statements
        self.localparams: list[tuple[str, str]] = []  # (name, expr), in order
        self.vars: list[str] = []             # real variables
        self.analog_body: str = ""
        self.instances: list[tuple[str, str, dict, dict]] = []
        # (module_name, inst_name, overrides, connections)


def _extract_analog(body: str, path: Path) -> tuple[str, str]:
    """Split the module body into (declarations, analog block inner text)."""
    m = re.search(r"\banalog\s+begin\b", body)
    if m is None:
        if re.search(r"\banalog\b", body):
            raise NotImplementedError(
                f"{path.name}: only 'analog begin ... end' blocks are "
                "supported by the hierarchy flattener")
        return body, ""
    depth, pos = 1, m.end()
    for tok in re.finditer(r"\b(begin|end)\b", body[m.end():]):
        depth += 1 if tok.group(1) == "begin" else -1
        if depth == 0:
            pos = m.end() + tok.start()
            break
    else:
        raise ValueError(f"{path.name}: unbalanced analog begin/end")
    inner = body[m.end():pos]
    rest = body[:m.start()] + body[m.end() + len(inner) + len("end"):]
    if re.search(r"\banalog\b", rest):
        raise NotImplementedError(
            f"{path.name}: multiple analog blocks are not supported")
    return rest, inner.strip("\n")


def _resolve_child(name: str, search_dir: Path) -> Path:
    hits = sorted(p for p in search_dir.glob(f"**/{name}.va")
                  if "__jax__" not in p.parts)
    if not hits:
        raise FileNotFoundError(
            f"hierarchical instance of '{name}': no {name}.va under "
            f"{search_dir}")
    if len(hits) > 1:
        raise ValueError(f"ambiguous module '{name}': {hits}")
    return hits[0]


def _parse(path: Path, search_dir: Path, _depth: int = 0) -> _Module:
    """Parse one .va file into a _Module, recursively flattening children."""
    if _depth > 4:
        raise RecursionError(f"{path.name}: hierarchy deeper than 4 levels")
    text = _strip_comments(path.read_text())
    m = _MODULE_RE.search(text)
    if m is None:
        raise ValueError(f"{path.name}: no module header found")
    name, ports = m.group(1), [p.strip() for p in m.group(2).split(",")]
    endm = text.rfind("endmodule")
    if endm < 0:
        raise ValueError(f"{path.name}: missing endmodule")

    mod = _Module(path, name, ports)
    for line in text[:m.start()].splitlines():
        line = " ".join(line.split())
        if line.startswith("`"):
            mod.header.append(line)

    decls, mod.analog_body = _extract_analog(text[m.end():endm], path)

    for stmt in decls.split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        first = stmt.split(None, 1)[0]
        if first == "inout":
            mod.port_decls.append(" ".join(stmt.split()))
        elif first == "electrical":
            names = [n.strip() for n in stmt[len("electrical"):].split(",")]
            onports = [n for n in names if n in mod.ports]
            if onports:
                # re-emit only the port part; non-port names go to
                # internal_nodes so a mixed declaration is never duplicated
                mod.port_decls.append("electrical " + ", ".join(onports))
            mod.internal_nodes += [n for n in names if n not in mod.ports]
        elif first == "parameter":
            pm = _PARAM_RE.match(stmt)
            if pm is None:
                raise NotImplementedError(
                    f"{path.name}: unsupported parameter statement: {stmt!r}")
            mod.params[pm.group(1)] = pm.group(2)
            mod.param_decls.append(" ".join(stmt.split()))
        elif first == "localparam":
            lm = _LOCALPARAM_RE.match(stmt)
            if lm is None:
                raise NotImplementedError(
                    f"{path.name}: unsupported localparam statement: {stmt!r}")
            mod.localparams.append((lm.group(1), " ".join(lm.group(2).split())))
        elif first == "real":
            mod.vars += [v.strip() for v in stmt[len("real"):].split(",")]
        else:
            im = _INSTANCE_RE.match(stmt)
            if im is None or im.group(1) in _KEYWORDS:
                raise NotImplementedError(
                    f"{path.name}: statement not supported by the hierarchy "
                    f"flattener: {stmt!r}")
            overrides = {}
            for item in _split_top_commas(im.group(2) or ""):
                nm = _NAMED_ITEM_RE.match(item)
                if nm is None:
                    raise NotImplementedError(
                        f"{path.name}: instance '{im.group(3)}': only named "
                        f"parameter overrides are supported, got {item!r}")
                overrides[nm.group(1)] = " ".join(nm.group(2).split())
            conns = {}
            for item in _split_top_commas(im.group(4)):
                nm = _NAMED_ITEM_RE.match(item)
                if nm is None:
                    raise NotImplementedError(
                        f"{path.name}: instance '{im.group(3)}': only named "
                        f"port connections are supported, got {item!r}")
                conns[nm.group(1)] = " ".join(nm.group(2).split())
            mod.instances.append((im.group(1), im.group(3), overrides, conns))

    if mod.instances:
        _flatten_into(mod, search_dir, _depth)
    return mod


def _flatten_into(mod: _Module, search_dir: Path, depth: int) -> None:
    """Inline every child instance of ``mod`` (which is mutated in place)."""
    chunks: list[str] = []
    taken = set(mod.ports) | set(mod.params) | set(mod.vars)
    taken |= set(mod.internal_nodes) | {n for n, _ in mod.localparams}

    for child_name, inst, overrides, conns in mod.instances:
        child = _parse(_resolve_child(child_name, search_dir), search_dir,
                       depth + 1)
        if child.name != child_name:
            raise ValueError(
                f"{mod.path.name}: instance '{inst}' of '{child_name}' "
                f"resolved to {child.path.name}, but that file defines "
                f"module '{child.name}' — the module name must equal the "
                "file stem (library convention)")
        if set(conns) != set(child.ports):
            raise ValueError(
                f"{mod.path.name}: instance '{inst}' must connect exactly the "
                f"ports of {child_name} ({sorted(child.ports)}), got "
                f"{sorted(conns)}")
        unknown = set(overrides) - set(child.params)
        if unknown:
            raise ValueError(
                f"{mod.path.name}: instance '{inst}' overrides unknown "
                f"parameter(s) {sorted(unknown)} of {child_name}")

        sub = {p: conns[p] for p in child.ports}
        for pname, default in child.params.items():
            if pname in overrides:
                sub[pname] = f"({overrides[pname]})"
                continue
            # an unoverridden default is inserted verbatim, where an
            # identifier in it would capture *parent* names — only pure
            # numeric literals are safe (scale suffixes normalised, since
            # they are only legal on declarations, not in expressions)
            lit = _LITERAL_RE.match(default)
            if lit is None:
                raise NotImplementedError(
                    f"{mod.path.name}: instance '{inst}': parameter "
                    f"'{pname}' of {child_name} has non-literal default "
                    f"{default!r}; override it explicitly")
            if lit.group(1):
                sub[pname] = repr(
                    float(default[:lit.start(1)]) * _SCALE[lit.group(1)])
            else:
                sub[pname] = f"({default})"
        for v in child.vars + child.internal_nodes + \
                [n for n, _ in child.localparams]:
            sub[v] = f"{inst}_{v}"
            if sub[v] in taken:
                raise ValueError(
                    f"{mod.path.name}: name collision on '{sub[v]}' while "
                    f"inlining instance '{inst}'")
            taken.add(sub[v])

        for line in child.header:
            if line not in mod.header:
                mod.header.append(line)
        mod.internal_nodes += [f"{inst}_{n}" for n in child.internal_nodes]
        mod.vars += [f"{inst}_{v}" for v in child.vars]
        mod.vars += [f"{inst}_{n}" for n, _ in child.localparams]

        body = [f"// --- {inst} : {child_name} ({child.path.name}) ---"]
        body += [f"{inst}_{n} = {_subst(e, sub)};"
                 for n, e in child.localparams]
        body.append(_subst(child.analog_body, sub))
        chunks.append("\n".join(body))

    mod.analog_body = "\n\n".join(
        ([mod.analog_body] if mod.analog_body else []) + chunks)
    mod.instances = []


def _duplicate_defines(header: list[str]) -> None:
    seen: dict[str, str] = {}
    for line in header:
        dm = re.match(r"`define\s+(\w+)\s+(.*)$", line)
        if dm and seen.setdefault(dm.group(1), dm.group(2)) != dm.group(2):
            raise ValueError(
                f"conflicting `define {dm.group(1)} across hierarchy: "
                f"{seen[dm.group(1)]!r} vs {dm.group(2)!r}")


def _emit(mod: _Module) -> str:
    """Serialise a flattened _Module back to flat Verilog-A source."""
    _duplicate_defines(mod.header)
    lp_names = [n for n, _ in mod.localparams]
    lp_assigns = [f"{n} = {e};" for n, e in mod.localparams]
    out = [
        f"// Auto-flattened from {mod.path} by photonflux.va_hier — do not "
        "edit; regenerated on every load.",
        "",
        *mod.header,
        "",
        f"module {mod.name}({', '.join(mod.ports)});",
        *(f"    {d};" for d in mod.port_decls),
    ]
    if mod.internal_nodes:
        out.append(f"    electrical {', '.join(mod.internal_nodes)};")
    out += [f"    {d};" for d in mod.param_decls]
    if lp_names or mod.vars:
        out.append(f"    real {', '.join(lp_names + mod.vars)};")
    out.append("")
    out.append("    analog begin")
    body = "\n".join(lp_assigns + [mod.analog_body])
    out += [f"        {line}" if line.strip() else ""
            for line in body.splitlines()]
    out += ["    end", "endmodule", ""]
    return "\n".join(out)


def flatten_va(path: Path, search_dir: Path) -> str | None:
    """Flatten the hierarchy of ``path``; None if it has no child instances.

    Flat sources — whatever Verilog-A they contain — are never parsed by the
    strict subset grammar: the instance detector gates it, so the flattener
    cannot reject (or alter) a model that openvaf already handled before
    hierarchy support existed.
    """
    path = Path(path)
    if not _has_instances(_strip_comments(path.read_text())):
        return None
    mod = _parse(path, Path(search_dir))
    if not mod.analog_body:
        raise ValueError(f"{path}: module has no analog behaviour to lower")
    return _emit(mod)


def flatten_file(path: Path, search_dir: Path, out_dir: Path) -> Path | None:
    """Flatten ``path`` into ``out_dir/<srchash>/<stem>.va``; None if flat.

    The output is rewritten on every call (string work only), so it can never
    go stale relative to its constituent sources; the compile cache keys on
    the flattened text itself. The per-source-path subdirectory keeps
    same-stem models from different model trees from sharing an output file
    (the file name itself must keep the stem — the component class name is
    derived from it).
    """
    text = flatten_va(path, search_dir)
    if text is None:
        return None
    path = Path(path)
    sub = hashlib.md5(str(path.resolve()).encode()).hexdigest()[:8]
    out = out_dir / sub / path.name
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists() or out.read_text() != text:
        out.write_text(text)
    return out
