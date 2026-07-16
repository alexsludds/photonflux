# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# photonflux public web demo — single image that serves the static editor AND
# runs the JAX/circulax solver (webapp/server.py). Target: a Linux x86_64 host
# that can run a container (Hugging Face Spaces "Docker" SDK, Google Cloud Run,
# a VPS, …). The whole app is one origin, so there is no CORS/frontend change.
#
# Two things force a from-source native build rather than a plain `pip install`:
#   * the repo vendors Mac-arm64 `bin/openvaf-*` and `models/*.osdi`; neither
#     runs on Linux, so the SKY130 FET path needs a Linux `openvaf-ir` and
#     freshly-compiled Linux `.osdi`.
#   * `openvaf-ir` is the ChipFlow OpenVAF fork (branch `vajax`) — it has NO
#     released binary, so stage `ovbuild` compiles it (Rust + LLVM 18), exactly
#     per README "Rebuilding the openvaf binaries".
# The photonic models need no native toolchain at runtime: `bosdi`/`openvaf-py`
# lower them to pure-Python JAX (cached under models/__jax__/), and the warmup
# step pre-populates that cache.
#
# NOTE: this image has not been built on this (arm64, no-daemon) machine. The
# authoritative build runs on the Space/Cloud builder. The `ovbuild` stage is
# the one to watch on the first build — see docs/README-DEPLOY.md.
# ---------------------------------------------------------------------------

ARG PY_VER=3.12
ARG LLVM_VER=18
# SKY130 PDK pin — matches the local dev box (open_pdks commit).
ARG SKY130_PDK_COMMIT=c6d73a35f524070e85faff4a6a9eef49553ebc2b

# ===========================================================================
# Stage 1 — build the ChipFlow OpenVAF fork -> Linux `openvaf-ir` (BSIM4->OSDI)
#   Stock OpenVAF miscompiles BSIM4 to OSDI; the `vajax` fork is the fix, so it
#   must be built from source (Rust + LLVM 18), then static-linked so the
#   runtime image needs no LLVM.
# ===========================================================================
FROM rust:1-bookworm AS ovbuild
ARG LLVM_VER
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates lsb-release wget gnupg software-properties-common \
        cmake zlib1g-dev libffi-dev libncurses-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*
# LLVM 18 from the official apt.llvm.org repo (bookworm ships 16 by default).
RUN curl -fsSL https://apt.llvm.org/llvm.sh -o /tmp/llvm.sh \
    && chmod +x /tmp/llvm.sh && /tmp/llvm.sh ${LLVM_VER} \
    && apt-get update && apt-get install -y --no-install-recommends \
        llvm-${LLVM_VER}-dev libclang-${LLVM_VER}-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 -b vajax https://github.com/robtaylor/OpenVAF /src/openvaf
WORKDIR /src/openvaf
# Static LLVM link (no PREFER_DYNAMIC) so the produced openvaf-ir is
# self-contained and the runtime image needs no libLLVM. If the static link
# fails on the first build, fall back to the README recipe by adding
# `LLVM_SYS_181_PREFER_DYNAMIC=1` here and copying libLLVM into the runtime.
# (If it stops on the Windows-only UCRT import-lib step, patch
# openvaf/target/build.rs to `let check = true;` — it is cfg(windows), usually
# a no-op on Linux.)
RUN LLVM_SYS_181_PREFIX=/usr/lib/llvm-${LLVM_VER} \
    cargo build --release -p openvaf-driver --bin openvaf-r --features llvm${LLVM_VER} \
    && cp target/release/openvaf-r /openvaf-ir \
    && /openvaf-ir --version || true

# ===========================================================================
# Stage 2 — runtime: python + circulax/JAX + libngspice + PDK + warmed caches
# ===========================================================================
FROM python:${PY_VER}-slim AS runtime
ARG SKY130_PDK_COMMIT

# Native runtime libs: libngspice0 (SKY130 model-card extraction via ctypes);
# libstdc++/zlib/tinfo/xml2 are the shared libs a statically-LLVM-linked
# openvaf-ir still resolves at runtime. git is kept for volare's PDK fetch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libngspice0 git ca-certificates libstdc++6 zlib1g libtinfo6 libxml2 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces (and good hygiene) run the container as a non-root user, uid 1000.
RUN useradd -m -u 1000 user
ENV HOME=/home/user
WORKDIR /app

# --- Python deps -----------------------------------------------------------
# bosdi is NOT on PyPI; install its Linux wheel from the GitHub release first so
# circulax's `bosdi>=0.1.3` requirement is already satisfied. Then circulax
# (pure-python) pulls jax/jaxlib/diffrax/klujax (all manylinux) from PyPI.
RUN pip install --no-cache-dir \
      "https://github.com/gdsfactory/bosdi/releases/download/v0.1.5/bosdi-0.1.5-cp312-cp312-manylinux_2_28_x86_64.whl" \
    && pip install --no-cache-dir \
      "circulax==0.2.1" "openvaf-py==0.1.5" volare numpy matplotlib

# --- app source + Linux openvaf-ir (overwrites the vendored Mac binary) -----
COPY . /app
COPY --from=ovbuild /openvaf-ir /app/bin/openvaf-ir
RUN chmod +x /app/bin/openvaf-ir \
    && pip install --no-cache-dir -e /app \
    # the vendored Mac-arm64 .osdi cannot load on Linux — force a clean recompile
    && rm -f /app/models/*.osdi /app/models/__jax__/*.osdi \
    && chown -R user:user /app

USER user

# --- SKY130 PDK (pinned) + warm every model cache --------------------------
# volare drops the PDK under $HOME/.volare/sky130A (toolchain.py's default).
RUN python -m volare enable --pdk sky130 ${SKY130_PDK_COMMIT}
# build_models() lowers all photonic .va -> models/__jax__/*.py and compiles the
# SKY130 FET flavors -> Linux .osdi; then a few representative examples JIT-warm
# the solver and prove the toolchain end to end. Failures are logged, not fatal.
RUN python /app/webapp/warmup.py

# --- server config ---------------------------------------------------------
ENV HOST=0.0.0.0 \
    PORT=7860 \
    PHOTONFLUX_OPENVAF_IR=/app/bin/openvaf-ir \
    PHOTONFLUX_ALLOW_VA_UPLOAD=0 \
    PHOTONFLUX_RUN_TIMEOUT_S=600 \
    JAX_ENABLE_X64=1
EXPOSE 7860
CMD ["python", "/app/webapp/server.py"]
