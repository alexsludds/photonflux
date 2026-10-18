"""Build IBIS-AMI executables for this host, and the hosts that run others.

``python -m photonflux ami build`` calls :func:`build_all`, which compiles
into ``models/ibis_ami/build`` (gitignored):

  ``<os-arch>/``          the photonflux models (pf_tx, pf_rx) + ``ami_host``
  ``public/ibisami-<os-arch>/``
                          capn-freako/ibisami's example Tx/Rx (BSD-3), from
                          the sources vendored in examples/ibis_ami/public;
                          needs a C++ compiler and Boost headers (Spirit)
  ``public/sisoft-<os-arch>/``
                          SiSoft's 2007 IBIS-AMI Eval Toolkit Tx, built from
                          the kit's own C source (header fix: see below)
  ``public/sisoft-original/``
                          the kit's original Linux i386 ``libIBIS_AMI_Tx.so``,
                          which runs in a ``linux/386`` container

Each step is skipped when its inputs are unchanged (a content hash is kept
next to the outputs) and reports, rather than raises, a missing toolchain.

Out-of-process hosting: :func:`ami_host` compiles ``ami_host`` for this
machine; :func:`container_argv` returns the ``docker run`` command that
serves a Linux library from a ``photonflux-ami-host:<arch>`` image (built on
first use from ``docker/ami-host.Dockerfile``), with networking disabled and
the model directory mounted read-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .ibisami import BUILD_ROOT, PF_MODELS, PUBLIC, REPO, Platform, host_platform

EXPORTS = ("AMI_Init", "AMI_GetWave", "AMI_Close")
SISOFT_ZIP = PUBLIC / "sisoft_eval_v2.21" / "SiSoft_IBIS-AMI_Eval_Toolkit_v2.21.zip"
IBISAMI_SRC = PUBLIC / "ibisami"
HOST_SRC = PF_MODELS / "host" / "ami_host.c"
HOST_DOCKERFILE = REPO / "docker" / "ami-host.Dockerfile"


@dataclass
class Step:
    name: str
    ok: bool
    detail: str
    outputs: tuple[Path, ...] = ()

    def __str__(self) -> str:
        return f"{'ok  ' if self.ok else 'skip'}  {self.name:<22} {self.detail}"


# ---------------------------------------------------------------------------
# toolchain
# ---------------------------------------------------------------------------

def _which(*names: str) -> str | None:
    for n in names:
        if n and shutil.which(n):
            return shutil.which(n)
    return None


def c_compiler() -> str | None:
    return _which(os.environ.get("CC", ""), "cc", "gcc", "clang")


def cxx_compiler() -> str | None:
    return _which(os.environ.get("CXX", ""), "c++", "g++", "clang++")


def boost_include() -> Path | None:
    """A directory holding boost/spirit/include/qi.hpp (ibisami's parser)."""
    cands = [os.environ.get("BOOST_ROOT", ""), os.environ.get("BOOST_INCLUDEDIR", ""),
             "/opt/homebrew/include", "/usr/local/include", "/usr/include",
             "/opt/local/include"]
    for c in cands:
        if c and (Path(c) / "boost" / "spirit" / "include" / "qi.hpp").exists():
            return Path(c)
    return None


def lib_suffix(plat: Platform | None = None) -> str:
    return ".dll" if (plat or host_platform()).os == "windows" else ".so"


def _link_flags(workdir: Path) -> list[str]:
    """Shared-library flags exporting only the AMI entry points."""
    plat = host_platform()
    if plat.os == "darwin":
        f = workdir / "exports.txt"
        f.write_text("".join(f"_{s}\n" for s in EXPORTS))
        return ["-dynamiclib", f"-Wl,-exported_symbols_list,{f}"]
    if plat.os == "linux":
        f = workdir / "exports.map"
        f.write_text("{ global: " + " ".join(f"{s};" for s in EXPORTS)
                     + " local: *; };\n")
        return ["-shared", f"-Wl,--version-script={f}"]
    return ["-shared"]


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])} ... failed:\n{r.stderr[-4000:]}")


def _digest(paths: list[Path], extra: str = "") -> str:
    h = hashlib.sha256(extra.encode())
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _fresh(out: Path, digest: str) -> bool:
    stamp = out / ".build.json"
    try:
        return json.loads(stamp.read_text()).get("digest") == digest
    except (OSError, ValueError):
        return False


def _stamp(out: Path, digest: str) -> None:
    (out / ".build.json").write_text(json.dumps({"digest": digest}))


# ---------------------------------------------------------------------------
# build steps
# ---------------------------------------------------------------------------

def build_photonflux(force: bool = False) -> Step:
    """pf_tx + pf_rx (C99) and their .ami/.ibs, for this host."""
    plat = host_platform()
    out = BUILD_ROOT / plat.tag
    cc = c_compiler()
    if cc is None:
        return Step("photonflux models", False, "no C compiler (set CC)")
    src = PF_MODELS / "src"
    inputs = sorted(src.glob("*.[ch]")) + sorted(PF_MODELS.glob("*.ami")) + \
        [PF_MODELS / "pf_serdes.ibs"]
    digest = _digest(inputs, cc)
    sfx = lib_suffix(plat)
    libs = tuple(out / f"{m}{sfx}" for m in ("pf_tx", "pf_rx"))
    if not force and _fresh(out, digest) and all(p.exists() for p in libs):
        return Step("photonflux models", True, f"up to date ({out})", libs)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        flags = _link_flags(Path(td))
        for m, lib in zip(("pf_tx", "pf_rx"), libs):
            _run([cc, "-std=c99", "-O2", "-fPIC", "-fvisibility=hidden",
                  "-Wall", *flags, "-o", str(lib), str(src / f"{m}.c"),
                  str(src / "pf_ami.c"), "-lm"])
    for f in list(PF_MODELS.glob("*.ami")) + [PF_MODELS / "pf_serdes.ibs"]:
        shutil.copy2(f, out / f.name)
    _stamp(out, digest)
    return Step("photonflux models", True, f"built pf_tx, pf_rx -> {out}", libs)


def ami_host(plat: Platform | None = None, force: bool = False) -> Path:
    """Compile (once) and return this machine's ``ami_host``."""
    here = host_platform()
    if plat is not None and plat != here:
        raise ValueError(f"cannot build ami_host for {plat} on {here}; "
                         "use a container")
    out = BUILD_ROOT / here.tag
    exe = out / ("ami_host.exe" if here.os == "windows" else "ami_host")
    digest = _digest([HOST_SRC])
    stamp = out / ".ami_host.json"
    try:
        fresh = json.loads(stamp.read_text()).get("digest") == digest
    except (OSError, ValueError):
        fresh = False
    if exe.exists() and fresh and not force:
        return exe
    cc = c_compiler()
    if cc is None:
        raise RuntimeError("no C compiler to build ami_host (set CC)")
    out.mkdir(parents=True, exist_ok=True)
    libs = ["-ldl"] if here.os == "linux" else []
    _run([cc, "-std=c99", "-D_POSIX_C_SOURCE=200809L", "-O2", "-o", str(exe),
          str(HOST_SRC), *libs])
    stamp.write_text(json.dumps({"digest": digest}))
    return exe


def build_ibisami(force: bool = False) -> Step:
    """capn-freako/ibisami example_tx / example_rx from the vendored source."""
    plat = host_platform()
    out = BUILD_ROOT / "public" / f"ibisami-{plat.tag}"
    cxx = cxx_compiler()
    boost = boost_include()
    if cxx is None:
        return Step("ibisami examples", False, "no C++ compiler (set CXX)")
    if boost is None:
        return Step("ibisami examples", False,
                    "Boost headers not found (brew install boost / apt install "
                    "libboost-dev, or set BOOST_ROOT)")
    lib_srcs = sorted((IBISAMI_SRC / "src").glob("*.cpp"))
    ex = IBISAMI_SRC / "example"
    inputs = lib_srcs + sorted((IBISAMI_SRC / "include").glob("*.h")) + \
        sorted(ex.glob("example_*"))
    digest = _digest(inputs, cxx + str(boost))
    # file names the kit's own .ibs gives its 64-bit Linux executables
    libs = tuple(out / f"{m}_x86_amd64{lib_suffix(plat)}" for m in
                 ("example_tx", "example_rx"))
    if not force and _fresh(out, digest) and all(p.exists() for p in libs):
        return Step("ibisami examples", True, f"up to date ({out})", libs)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        flags = _link_flags(Path(td))
        objs = []
        for s in lib_srcs:
            o = Path(td) / (s.stem + ".o")
            _run([cxx, "-std=gnu++11", "-O2", "-fPIC", "-w", "-c",
                  f"-I{IBISAMI_SRC}", f"-I{boost}", "-o", str(o), str(s)])
            objs.append(str(o))
        for m, lib in zip(("example_tx", "example_rx"), libs):
            _run([cxx, "-std=gnu++11", "-O2", "-fPIC", "-w", *flags,
                  f"-I{IBISAMI_SRC}", f"-I{boost}", "-o", str(lib),
                  str(ex / f"{m}.cpp"), *objs])
    for m in ("example_tx", "example_rx"):
        for ext in (".ami", ".ibs"):
            shutil.copy2(ex / f"{m}{ext}", out / f"{m}{ext}")
    shutil.copy2(IBISAMI_SRC / "LICENSE", out / "LICENSE")
    _stamp(out, digest)
    return Step("ibisami examples", True, f"built example_tx, example_rx -> {out}",
                libs)


# SiSoft's toolkit ships an IBIS_AMI_Tx.h older than IBIS_AMI_Tx.c: the
# struct lacks the write pointer the 2008 GetWave uses. The original binary
# has it; this restores the field (the kit's license permits modification).
_SISOFT_HDR_FIX = ("   int samples;           //Number of samples per symbol\n",
                   "   int samples;           //Number of samples per symbol\n"
                   "   int wr_ptr;            //photonflux: field the .c uses\n")


def _sisoft_members(zf: zipfile.ZipFile, sub: str) -> list[str]:
    return [n for n in zf.namelist() if f"/{sub}/" in n and not n.endswith("/")]


def build_sisoft(force: bool = False) -> Step:
    """SiSoft IBIS_AMI_Tx, compiled natively from the kit's C source."""
    plat = host_platform()
    out = BUILD_ROOT / "public" / f"sisoft-{plat.tag}"
    cc = c_compiler()
    if cc is None:
        return Step("SiSoft Tx (native)", False, "no C compiler (set CC)")
    if not SISOFT_ZIP.exists():
        return Step("SiSoft Tx (native)", False, f"{SISOFT_ZIP} missing")
    digest = _digest([SISOFT_ZIP], cc)
    lib = out / f"libIBIS_AMI_Tx{lib_suffix(plat)}"
    if not force and _fresh(out, digest) and lib.exists():
        return Step("SiSoft Tx (native)", True, f"up to date ({out})", (lib,))
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(SISOFT_ZIP) as zf:
        src = Path(td)
        for n in _sisoft_members(zf, "src"):
            base = Path(n).name
            if base.endswith((".c", ".h", ".ami", ".ibs")):
                (src / base).write_bytes(zf.read(n))
        hdr = src / "IBIS_AMI_Tx.h"
        text = hdr.read_text(errors="replace").replace("\r\n", "\n")
        if "wr_ptr" not in text:
            text = text.replace(*_SISOFT_HDR_FIX)
        hdr.write_text(text)
        _run([cc, "-std=gnu99", "-O2", "-fPIC", "-w", *_link_flags(src),
              f"-I{src}", "-o", str(lib), str(src / "IBIS_AMI_Tx.c"),
              str(src / "IBIS_AMI_tree.c"), "-lm"])
        for f in ("IBIS_AMI_Tx.ami", "sisoft_tx.ibs"):
            shutil.copy2(src / f, out / f)
        (out / "license.txt").write_bytes(zf.read(next(
            n for n in zf.namelist() if n.endswith("license.txt"))))
    _stamp(out, digest)
    return Step("SiSoft Tx (native)", True, f"built from the kit's source -> {out}",
                (lib,))


def extract_sisoft_original(force: bool = False) -> Step:
    """The kit's original Linux i386 libIBIS_AMI_Tx.so (container-hosted)."""
    out = BUILD_ROOT / "public" / "sisoft-original"
    if not SISOFT_ZIP.exists():
        return Step("SiSoft Tx (original)", False, f"{SISOFT_ZIP} missing")
    digest = _digest([SISOFT_ZIP])
    lib = out / "libIBIS_AMI_Tx.so"
    if not force and _fresh(out, digest) and lib.exists():
        return Step("SiSoft Tx (original)", True, f"up to date ({out})", (lib,))
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SISOFT_ZIP) as zf, tempfile.TemporaryDirectory() as td:
        tar_name = next(n for n in zf.namelist() if n.endswith("linux.tar"))
        tpath = Path(td) / "linux.tar"
        tpath.write_bytes(zf.read(tar_name))
        with tarfile.open(tpath) as tf:
            member = next(m for m in tf.getmembers()
                          if m.name.endswith("libIBIS_AMI_Tx.so") and m.isfile())
            lib.write_bytes(tf.extractfile(member).read())
        for n in _sisoft_members(zf, "src"):
            if n.endswith(("IBIS_AMI_Tx.ami", "sisoft_tx.ibs")):
                (out / Path(n).name).write_bytes(zf.read(n))
        (out / "license.txt").write_bytes(zf.read(next(
            n for n in zf.namelist() if n.endswith("license.txt"))))
    _stamp(out, digest)
    return Step("SiSoft Tx (original)", True,
                f"Linux i386 binary -> {out} (runs in a linux/386 container)",
                (lib,))


def build_all(force: bool = False, quiet: bool = False) -> list[Step]:
    steps = [build_photonflux(force)]
    try:
        ami_host(force=force)
        steps.append(Step("ami_host", True, "local out-of-process host"))
    except (RuntimeError, ValueError) as exc:
        steps.append(Step("ami_host", False, str(exc).splitlines()[0]))
    for fn in (build_ibisami, build_sisoft, extract_sisoft_original):
        try:
            steps.append(fn(force))
        except RuntimeError as exc:   # a public kit must not block the rest
            steps.append(Step(fn.__name__, False, str(exc).splitlines()[0]))
    if not quiet:
        for s in steps:
            print(s)
    return steps


# ---------------------------------------------------------------------------
# container hosting
# ---------------------------------------------------------------------------

def host_image(plat: Platform) -> str:
    return f"photonflux-ami-host:{plat.arch}"


def ensure_host_image(plat: Platform, *, quiet: bool = True) -> str:
    """Build the ``ami_host`` image for a Linux platform if it is missing."""
    if plat.container is None:
        raise ValueError(f"no container platform for {plat}")
    tag = host_image(plat)
    have = subprocess.run(["docker", "image", "inspect", tag],
                          capture_output=True)
    if have.returncode == 0:
        return tag
    cmd = ["docker", "build", "--platform", plat.container, "-t", tag,
           "-f", str(HOST_DOCKERFILE), str(HOST_SRC.parent)]
    r = subprocess.run(cmd, capture_output=quiet, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"docker build for {plat.container} failed:\n"
                           + (r.stderr or "")[-3000:])
    return tag


def container_argv(plat: Platform, dll: Path) -> list[str]:
    """``docker run`` argv serving ``dll`` through ami_host (stdin/stdout)."""
    tag = ensure_host_image(plat)
    return ["docker", "run", "--rm", "-i", "--platform", plat.container,
            "--network", "none", "--read-only",
            "-v", f"{dll.parent}:/model:ro", tag,
            "/usr/local/bin/ami_host", f"/model/{dll.name}"]
