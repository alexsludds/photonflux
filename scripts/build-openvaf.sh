#!/usr/bin/env bash
# Build the cdaunt OpenVAF fork (pinned commit) and install it as
# `bin/openvaf-ir`. It is both the Verilog-A -> OSDI compiler that
# `cx.sky130_fet` (and every SKY130 FET example) needs and the `--dump-json`
# front end bosdi lowers every photonic `cx.va` model with. There is no pip
# wheel for it.
#
# Usage:   scripts/build-openvaf.sh            # build if missing
#          scripts/build-openvaf.sh --force    # rebuild even if present
#
# Prerequisites (macOS): `brew install rustup-init llvm@18` (or `rustup` +
# `llvm@18`). Linux: a Rust toolchain plus LLVM 18 dev packages. See the
# README "Rebuilding openvaf-ir" section; the container does the Linux build.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${PHOTONFLUX_OPENVAF_IR:-$REPO/bin/openvaf-ir}"
FORCE="${1:-}"
OPENVAF_REPO="https://github.com/cdaunt/OpenVAF"
OPENVAF_REV="89a1f798c0768f30743ef7e9bf6cca6eebd24307"

log() { printf '\033[1m[build-openvaf]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[build-openvaf] error:\033[0m %s\n' "$*" >&2; exit 1; }

# Emits the --dump-json MIR bosdi lowers Verilog-A from? An older
# robtaylor@vajax build runs (--version) but fails this.
dumps_json() {
  local probe
  probe="$(mktemp -d)/probe.va"
  printf '%s\n' '`include "disciplines.vams"' 'module probe(a, b);' \
    '  inout a, b;' '  electrical a, b;' '  analog I(a, b) <+ V(a, b);' \
    'endmodule' > "$probe"
  "$DEST" --dump-json "$probe" 2>/dev/null | grep -q '"schema_version": 1'
}

# Already built, runnable and the right fork? Nothing to do.
if [[ "$FORCE" != "--force" && -x "$DEST" ]]; then
  if "$DEST" --version >/dev/null 2>&1 && dumps_json; then
    log "already installed and runnable: $DEST"
    "$DEST" --version 2>&1 | sed 's/^/  /' || true
    exit 0
  fi
  log "$DEST is missing bosdi's --dump-json or does not run; rebuilding"
fi

command -v git >/dev/null   || die "git not found"
command -v cargo >/dev/null || die "cargo not found — install Rust (https://rustup.rs)"

# Locate LLVM 18. Honour an explicit LLVM_SYS_181_PREFIX; otherwise probe the
# usual Homebrew (Apple Silicon + Intel) and Linux locations.
LLVM_PREFIX="${LLVM_SYS_181_PREFIX:-}"
if [[ -z "$LLVM_PREFIX" ]]; then
  for cand in /opt/homebrew/opt/llvm@18 /usr/local/opt/llvm@18 \
              /usr/lib/llvm-18 /usr/lib/llvm18; do
    if [[ -d "$cand" ]]; then LLVM_PREFIX="$cand"; break; fi
  done
fi
[[ -n "$LLVM_PREFIX" && -d "$LLVM_PREFIX" ]] || die \
  "LLVM 18 not found. Install it (macOS: brew install llvm@18) or set LLVM_SYS_181_PREFIX."
log "using LLVM 18 at $LLVM_PREFIX"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
log "fetching $OPENVAF_REPO@$OPENVAF_REV into $WORK"
git init -q "$WORK/openvaf"
git -C "$WORK/openvaf" fetch -q --depth 1 "$OPENVAF_REPO" "$OPENVAF_REV"
git -C "$WORK/openvaf" checkout -q FETCH_HEAD

# openvaf/target/build.rs builds Windows UCRT import libs on every host with
# unversioned `clang`/`llvm-lib`, which Homebrew's keg-only llvm@18 does not put
# on PATH. Take its RUST_CHECK skip path unconditionally (unused off Windows).
BUILD_RS="$WORK/openvaf/openvaf/target/build.rs"
if [[ -f "$BUILD_RS" ]] && grep -q 'let check = ' "$BUILD_RS"; then
  sed -i.bak 's/let check = .*;/let check = true;/' "$BUILD_RS" && rm -f "$BUILD_RS.bak"
fi

log "building openvaf-r (release) — this takes a few minutes"
(
  cd "$WORK/openvaf"
  LLVM_SYS_181_PREFIX="$LLVM_PREFIX" LLVM_SYS_181_PREFER_DYNAMIC=1 \
    cargo build --release -p openvaf-driver --bin openvaf-r --features llvm18
)

mkdir -p "$(dirname "$DEST")"
cp "$WORK/openvaf/target/release/openvaf-r" "$DEST"
chmod +x "$DEST"
log "installed: $DEST"
"$DEST" --version 2>&1 | sed 's/^/  /' || true
log "done — verify with: python -m photonflux doctor"
