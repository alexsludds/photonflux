# Deploying lightspice as a public web demo

The lightspice web editor runs **real** JAX/circulax circuit solves, so it
cannot be a pure static / WebAssembly site: `jaxlib` (XLA) and `klujax` have no
WebAssembly build, and adding a model shells out to native OpenVAF. Instead we
ship **one container that serves both** the static editor and the `/api/run`
solver (single origin — no CORS, no frontend change). The stdlib server in
[`webapp/server.py`](webapp/server.py) already does both.

Everything is in the repo-root [`Dockerfile`](Dockerfile) + build-time
[`webapp/warmup.py`](webapp/warmup.py). This doc is how to build and publish it.

## What the image contains

| Piece | Source | Notes |
|---|---|---|
| jaxlib, klujax, diffrax, sax | PyPI (manylinux) | pulled by `circulax` |
| circulax 0.2.1 | PyPI (pure-python) | the solver |
| openvaf-py 0.1.5 | PyPI (manylinux) | Verilog-A front-end binding |
| **bosdi 0.1.5** | **GitHub release wheel** | *not on PyPI* — installed from `gdsfactory/bosdi` release `v0.1.5` (manylinux_2_28_x86_64) |
| libngspice0 | apt | SKY130 model-card extraction |
| **openvaf-ir** | **built from source** | ChipFlow fork `robtaylor/OpenVAF@vajax`, LLVM 18 — no released binary exists |
| SKY130 PDK | volare | pinned `open_pdks c6d73a35…` |
| photonic `models/__jax__/*.py` + Linux `*.osdi` | warmup | pre-compiled into the image |

Runtime env baked in: `HOST=0.0.0.0`, `PORT=7860`,
`LIGHTSPICE_ALLOW_VA_UPLOAD=0` (public safety — untrusted VA upload compiles
native code, so it is **off**), `LIGHTSPICE_RUN_TIMEOUT_S=600` (10-min ceiling —
generous enough for the multi-minute Vernier example, bounds runaways),
`LIGHTSPICE_OPENVAF_IR=/app/bin/openvaf-ir`.

## ⚠️ The one step to watch on the first build

`openvaf-ir` (the BSIM4-correct OpenVAF fork) has **no prebuilt binary**, so the
`ovbuild` stage compiles it with Rust + LLVM 18 (per the README "Rebuilding the
openvaf binaries" recipe). This is the only fragile stage. After the build,
**read the warmup summary in the build log** — it prints `PASS/FAIL` per model:

```
[warmup] summary: 5/5 ok
  ✓ 04_sky130_nfet_output_curves: 12 traces, 0.8s   <- FET/OSDI path works
```

If the FET lines FAIL but the photonic ones pass, the image still ships and all
photonics works; only SKY130 examples are affected. To ship photonics-only
deliberately (skip the whole OpenVAF build), comment out the `ovbuild` stage and
its `COPY --from=ovbuild …` line — the photonic path needs no native toolchain at
runtime. (`webapp/warmup.py` already tolerates FET failures; set `WARMUP_STRICT=1`
to make any failure fail the build instead.)

## Option A — Google Cloud Run (recommended free path)

**Free for a low-traffic demo** via Google's Always Free tier (2M requests +
180k vCPU-sec + 360k GB-sec/month), scale-to-zero, and it **builds the image in
the cloud** — no local Docker needed. Requires a GCP project with **billing
enabled** (a card on file; you pay ~$0 at demo traffic). The heavy OpenVAF build
needs a **two-step** flow — the naive `gcloud run deploy --source .` hits Cloud
Build's 10-min default timeout.

```bash
# 0) one-time: install gcloud, log in, pick a project + region
gcloud init
PROJECT=$(gcloud config get-value project); REGION=us-central1
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
    artifactregistry.googleapis.com
gcloud artifacts repositories create lightspice \
    --repository-format=docker --location=$REGION

IMG=$REGION-docker.pkg.dev/$PROJECT/lightspice/app:latest

# 1) build in the cloud — raise the timeout (Rust/LLVM + PDK is slow) and use a
#    bigger builder so it finishes; watch the log for the warmup PASS/FAIL summary
gcloud builds submit --tag "$IMG" --timeout=3600s --machine-type=e2-highcpu-8 .

# 2) deploy the built image
gcloud run deploy lightspice --image "$IMG" --region $REGION \
    --port 7860 --memory 4Gi --cpu 2 --allow-unauthenticated \
    --timeout 900 --min-instances 0 --max-instances 3
```

`--min-instances 0` → $0 when idle (cold start on the next hit); `--max-instances`
caps runaway cost; `--memory 4Gi` gives JAX headroom (try 2Gi to trim cost).
`e2-highcpu-8` is a small paid build cost but keeps the LLVM compile under the
timeout; drop it to use free build minutes if you don't mind a slower build.

## Option B — Hugging Face Spaces (Docker SDK — now paid)

As of early 2026 HF moved the Docker (and Gradio) SDKs behind a paid plan for
free accounts, so a free Docker Space is no longer available. If you have HF PRO
(or the policy reverts): free CPU is 2 vCPU / 16 GB, Docker SDK, a persistent
public URL that sleeps after ~48 h idle.

1. Create a **Docker** Space at https://huggingface.co/new-space (SDK = Docker,
   hardware = CPU basic).
2. Put this YAML **frontmatter at the very top of the Space repo's `README.md`**
   (HF reads it to configure the Space):

   ```yaml
   ---
   title: lightspice
   emoji: 🔦
   colorFrom: indigo
   colorTo: purple
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```

3. Push this repo (with the `Dockerfile`) to the Space's git remote:

   ```bash
   git remote add space https://huggingface.co/spaces/<user>/lightspice
   git push space main
   ```

   The Space builds the image and serves it at
   `https://<user>-lightspice.hf.space`. First build is long (Rust/LLVM +
   PDK download); watch the build log for the warmup summary.

## Build and run locally (optional pre-flight)

Requires Docker running on an x86_64 builder (or `buildx` emulation on Apple
Silicon — slow):

```bash
docker build -t lightspice .
docker run --rm -p 7860:7860 lightspice
# open http://localhost:7860 ; load example 39 (photonics) and 04 (SKY130 FET)
docker logs <id>        # confirm /api/run 200s, no missing-binary/PDK errors
```

## Operational notes

- **Single-flight compute.** `webapp/server.py` serializes runs behind one lock,
  so concurrent visitors queue — fine for a demo, not a load-bearing service.
- **VA upload is disabled** on the public image. To allow it on a trusted deploy,
  set `LIGHTSPICE_ALLOW_VA_UPLOAD=1` (it compiles untrusted Verilog-A natively —
  only do this behind auth).
- **Local dev is unchanged.** Running `.venv-circulax/bin/python webapp/server.py`
  with no env vars keeps 127.0.0.1:8642, VA upload on, and no run timeout — every
  new behavior is opt-in via env.
- **Pinned versions:** circulax 0.2.1 · bosdi 0.1.5 · openvaf-py 0.1.5 ·
  SKY130 `open_pdks c6d73a35f524070e85faff4a6a9eef49553ebc2b` · LLVM 18 ·
  ngspice/libngspice (Debian). Bump the `ARG`s in the Dockerfile to move them.
