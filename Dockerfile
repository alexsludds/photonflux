# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# photonflux public web demo — single image that serves the static editor AND
# runs the JAX/circulax solver (webapp/server.py). Target: a Linux x86_64 host
# that can run a container (GitHub Codespaces via ghcr.io, Google Cloud Run,
# a VPS, …). The whole app is one origin, so there is no CORS/frontend change.
#
# The SKY130 FET path needs a Linux `openvaf-ir` (the ChipFlow OpenVAF fork,
# branch `vajax`, which has no upstream release) and freshly-compiled Linux
# `.osdi`. Compiling openvaf-ir takes Rust + LLVM 18, so CI builds it once
# (docker/openvaf-ir.Dockerfile, .github/workflows/openvaf-ir.yml) and this
# image downloads that release asset (OPENVAF_IR_URL).
# The photonic models need no native toolchain at runtime: `bosdi`/`openvaf-py`
# lower them to pure-Python JAX (cached under models/__jax__/), and the warmup
# step pre-populates that cache.
#
# NOTE: this image has not been built on this (arm64, no-daemon) machine. The
# authoritative build is .github/workflows/image.yml — see docs/README-DEPLOY.md.
# ---------------------------------------------------------------------------

ARG PY_VER=3.12
ARG OPENVAF_IR_URL=https://github.com/alexsludds/photonflux/releases/download/openvaf-ir/openvaf-ir-linux-x86_64
# SKY130 PDK pin — matches the local dev box (open_pdks commit).
ARG SKY130_PDK_COMMIT=c6d73a35f524070e85faff4a6a9eef49553ebc2b

# ===========================================================================
# Runtime: python + circulax/JAX + libngspice + PDK + warmed caches
# ===========================================================================
FROM python:${PY_VER}-slim AS runtime
ARG SKY130_PDK_COMMIT
ARG OPENVAF_IR_URL

# Native runtime libs: libngspice0 (SKY130 model-card extraction via ctypes);
# libstdc++/zlib/tinfo/xml2 are the shared libs a statically-LLVM-linked
# openvaf-ir still resolves at runtime. git is kept for volare's PDK fetch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libngspice0 git ca-certificates libstdc++6 zlib1g libtinfo6 libxml2 \
    && rm -rf /var/lib/apt/lists/*

# Non-root uid 1000 — the Codespaces devcontainer's remoteUser.
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
RUN mkdir -p /app/bin && python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], '/app/bin/openvaf-ir')" "${OPENVAF_IR_URL}" \
    && chmod +x /app/bin/openvaf-ir \
    && pip install --no-cache-dir -e /app \
    # the vendored Mac-arm64 .osdi cannot load on Linux — force a clean recompile
    && rm -f /app/models/*.osdi /app/models/__jax__/*.osdi \
    && chown -R user:user /app

USER user

# --- SKY130 PDK (pinned) + warm every model cache --------------------------
# volare drops the PDK under $HOME/.volare/sky130A (toolchain.py's default).
# Only the ngspice library and the sky130_fd_pr device models it includes are
# used; drop the ~1 GB of standard cells, IO, SRAM and sky130B in the same layer
# so the published image (and every Codespace pull) stays small.
RUN python -m volare enable --pdk sky130 ${SKY130_PDK_COMMIT} \
    && pdk="$(readlink -f "$HOME/.volare/sky130A")" \
    && find "$pdk" -mindepth 1 -maxdepth 1 ! -name libs.tech ! -name libs.ref -exec rm -rf {} + \
    && find "$pdk/libs.tech" -mindepth 1 -maxdepth 1 ! -name ngspice -exec rm -rf {} + \
    && find "$pdk/libs.ref" -mindepth 1 -maxdepth 1 ! -name sky130_fd_pr -exec rm -rf {} + \
    && rm -rf "$(dirname "$pdk")/sky130B" "$HOME/.volare/sky130B"
# build_models() lowers all photonic .va -> models/__jax__/*.py and compiles the
# SKY130 FET flavors -> Linux .osdi; then a few representative examples JIT-warm
# the solver and prove the toolchain end to end. Failures are logged, not fatal.
RUN python /app/webapp/warmup.py

# --- server config ---------------------------------------------------------
# PHOTONFLUX_RELOAD=0 keeps this a single plain process: the dev auto-reloader
# (a supervisor that re-spawns the server on .py edits) is ON by default for
# local `server.py`, but a container image ships fixed source and must not run
# the extra supervisor/watcher.
ENV HOST=0.0.0.0 \
    PORT=7860 \
    PHOTONFLUX_OPENVAF_IR=/app/bin/openvaf-ir \
    PHOTONFLUX_ALLOW_VA_UPLOAD=0 \
    PHOTONFLUX_ENABLE_BRIDGE=0 \
    PHOTONFLUX_RUN_TIMEOUT_S=600 \
    PHOTONFLUX_RELOAD=0 \
    JAX_ENABLE_X64=1
EXPOSE 7860
CMD ["python", "/app/webapp/server.py"]
