# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Linux x86_64 build of `openvaf-ir` — the ChipFlow OpenVAF fork (branch
# `vajax`) that compiles BSIM4 to OSDI correctly. It has no released binary and
# needs Rust + LLVM 18, so it is built once in CI
# (.github/workflows/openvaf-ir.yml) and published as a GitHub release asset
# that the main Dockerfile downloads. Build locally with:
#
#   docker build -f docker/openvaf-ir.Dockerfile --target out --output dist .
# ---------------------------------------------------------------------------

ARG LLVM_VER=18
ARG OPENVAF_REF=vajax

FROM rust:1-bookworm AS ovbuild
ARG LLVM_VER
ARG OPENVAF_REF
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates lsb-release wget gnupg software-properties-common \
        cmake zlib1g-dev libzstd-dev libffi-dev libncurses-dev libxml2-dev \
    && rm -rf /var/lib/apt/lists/*
# LLVM 18 from the official apt.llvm.org repo (bookworm ships 16 by default).
RUN curl -fsSL https://apt.llvm.org/llvm.sh -o /tmp/llvm.sh \
    && chmod +x /tmp/llvm.sh && /tmp/llvm.sh ${LLVM_VER} \
    && apt-get update && apt-get install -y --no-install-recommends \
        llvm-${LLVM_VER}-dev libclang-${LLVM_VER}-dev clang-${LLVM_VER} libpolly-${LLVM_VER}-dev \
    && rm -rf /var/lib/apt/lists/*
# openvaf/target/build.rs shells out to unversioned `clang` and `llvm-lib`
# (it builds the Windows UCRT import libs on every host).
ENV PATH=/usr/lib/llvm-${LLVM_VER}/bin:$PATH
RUN git clone --depth 1 -b ${OPENVAF_REF} https://github.com/robtaylor/OpenVAF /src/openvaf
WORKDIR /src/openvaf
# Static LLVM link so openvaf-ir is self-contained and the runtime image needs
# no libLLVM: the fork pins llvm-sys to `prefer-dynamic`, so flip it to
# `force-static` (the smoke stage fails if libLLVM is still a dependency).
RUN grep -rl --include=Cargo.toml '"prefer-dynamic"' . | xargs sed -i 's/"prefer-dynamic"/"force-static"/' \
    && LLVM_SYS_181_PREFIX=/usr/lib/llvm-${LLVM_VER} \
    cargo build --release -p openvaf-driver --bin openvaf-r --features llvm${LLVM_VER} \
    && cp target/release/openvaf-r /openvaf-ir \
    && git rev-parse HEAD > /openvaf-ir.commit

# Smoke test: the binary must resolve every shared lib inside the same slim
# base + apt libs the runtime image (../Dockerfile) installs.
FROM python:3.12-slim AS smoke
RUN apt-get update && apt-get install -y --no-install-recommends \
        libstdc++6 zlib1g libzstd1 libtinfo6 libxml2 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ovbuild /openvaf-ir /usr/local/bin/openvaf-ir
RUN ldd /usr/local/bin/openvaf-ir && ! ldd /usr/local/bin/openvaf-ir | grep -q 'not found'

FROM scratch AS out
COPY --from=smoke /usr/local/bin/openvaf-ir /openvaf-ir-linux-x86_64
COPY --from=ovbuild /openvaf-ir.commit /openvaf-ir.commit
