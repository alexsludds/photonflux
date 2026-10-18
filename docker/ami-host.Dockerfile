# ami_host for one Linux platform: serves an IBIS-AMI shared library's
# AMI_Init / AMI_GetWave / AMI_Close over stdin/stdout so photonflux can run
# Linux vendor models (32-bit i386 or x86-64) from any host. Built on demand by
# photonflux.ami_build.ensure_host_image:
#
#   docker build --platform linux/386 -t photonflux-ami-host:x86 \
#       -f docker/ami-host.Dockerfile models/ibis_ami/host
#
# libstdc++ stays in the image: C++ models (most vendor kits) link it.
FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY ami_host.c /src/ami_host.c
RUN gcc -std=c99 -D_POSIX_C_SOURCE=200809L -O2 -o /usr/local/bin/ami_host \
        /src/ami_host.c -ldl
USER nobody
