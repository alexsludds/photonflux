#!/usr/bin/env bash
# Build the ChipFlow OpenVAF fork (branch `vajax`) and install it as
# `bin/openvaf-ir` — the Verilog-A -> OSDI compiler that `cx.sky130_fet`
# (and every SKY130 FET example) needs. Stock OpenVAF miscompiles BSIM4, so
# the fork is required; there is no pip wheel for it.
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

log() { printf '\033[1m[build-openvaf]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[build-openvaf] error:\033[0m %s\n' "$*" >&2; exit 1; }

# Already built and runnable? Nothing to do.
if [[ "$FORCE" != "--force" && -x "$DEST" ]]; then
  if "$DEST" --version >/dev/null 2>&1; then
    log "already installed and runnable: $DEST"
    "$DEST" --version 2>&1 | sed 's/^/  /' || true
    exit 0
  fi
  log "$DEST exists but does not run; rebuilding"
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
log "cloning robtaylor/OpenVAF@vajax into $WORK"
git clone --depth 1 -b vajax https://github.com/robtaylor/OpenVAF "$WORK/openvaf"

# macOS-only guard: the Windows UCRT import-lib check in build.rs is cfg(windows)
# and a no-op elsewhere, but force it green in case a stale checkout trips it.
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
