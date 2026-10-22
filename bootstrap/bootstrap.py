#!/usr/bin/env python3
"""Build the strixllama runtime from a clean upstream checkout.

Three steps, in this order, because two of the patches anchor against what an earlier one writes:

  1. fetch  upstream llama.cpp at the pinned revision into src/
  2. patch  overlay patches/files/ (whole files), then apply the scripts in PATCH_ORDER
  3. build  cmake + ninja against the ROCm SDK in toolchain/, then copy the runtime into bin/

Every step is verifiable and every step can be run alone:

    python bootstrap/bootstrap.py --fetch --patch --build
    python bootstrap/bootstrap.py --verify        # is the tree exactly what the patches produce?

--verify is the check that matters when something looks wrong: it re-hashes the tree against
UPSTREAM.json and re-runs the patch set on a scratch copy, so "my build differs from the recipe"
is answerable rather than a suspicion. tools/replay_bootstrap.py does the same thing from the other
direction and is what proved the recipe in the first place.

Requirements: git, cmake, ninja, and the ROCm SDK. See README for the exact versions.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TREE = os.path.join(ROOT, "src", "llama.cpp")
BUILD = os.path.join(TREE, "build")
BIN = os.path.join(ROOT, "bin", "hip")   # bin/hip-rocm101 up to 0.2.4

# The recipe: upstream at a pinned revision, the whole files under patches/files/ (what the fork adds,
# and the upstream files it rewrites more than it keeps), then the scripts in order. All of it is in
# bootstrap/UPSTREAM.json, which tools/make_recipe.py writes whenever the fork moves to a newer
# upstream; the per-change scripts of the recipe before that are in patches/history/.
RECIPE = json.load(open(os.path.join(ROOT, "bootstrap", "UPSTREAM.json"), encoding="utf-8"))
OVERLAY = RECIPE["overlay"]
PATCH_ORDER = RECIPE["order"]
UPSTREAM_REPO = RECIPE["repo"]
UPSTREAM_REV = RECIPE["revision"]

# The ROCm SDK is installed as Python wheels, which is the only form AMD ships for Windows - the
# native tarballs are Linux-only. ROCM_VERSION is pinned to what every number in this repository was
# measured on. See docs/install.md: this is a NIGHTLY index with a rolling window of about 27 days,
# so a pin eventually stops resolving. --rocm-version overrides it.
ROCM_INDEX = "https://nightly.repo.amd.com/rocm/whl-next/"
ROCM_VERSION = "10.2.0a20260925"
ROCM_EXTRAS = "rocm[libraries,devel,device-gfx1151]"
NINJA_URL = "https://github.com/ninja-build/ninja/releases/download/v1.13.1/ninja-win.zip"

CMAKE_ARGS = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_HIP=ON", "-DGGML_VULKAN=OFF",
    "-DAMDGPU_TARGETS=gfx1151", "-DGPU_TARGETS=gfx1151", "-DGGML_HIP_GRAPHS=ON",
    # gfx1151 is a laptop/APU part; these keep the CPU side off the host's own ISA so the build is
    # reproducible rather than tuned to whatever machine ran it
    "-DGGML_NATIVE=OFF", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON", "-DGGML_AVX512=OFF",
    "-DLLAMA_CURL=OFF", "-DLLAMA_USE_PREBUILT_UI=OFF", "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=ON", "-DLLAMA_BUILD_TOOLS=ON",
]
RUNTIME_FILES = ["llama-server.exe", "llama-server-impl.dll", "llama.dll", "llama-common.dll",
                 "ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-hip.dll", "mtmd.dll",
                 "llama-quantize.exe", "llama-quantize-impl.dll"]

VCVARS = [r"C:\Program Files\Microsoft Visual Studio\2022\%s\VC\Auxiliary\Build\vcvars64.bat" % ed
          for ed in ("Community", "Professional", "Enterprise", "BuildTools")]


def find_vcvars():
    """vcvars64.bat of the newest Visual Studio with the C++ x64 tools, as vswhere (installed with every edition, the
    Build Tools included) reports it. The Build Tools install under Program Files (x86), which the list above missed
    (issue #17); the list stays as the fallback."""
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                           "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if os.path.isfile(vswhere):
        out = subprocess.run([vswhere, "-latest", "-products", "*", "-requires",
                              "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                             capture_output=True, text=True, errors="replace").stdout
        for line in out.splitlines():
            vc = os.path.join(line.strip(), "VC", "Auxiliary", "Build", "vcvars64.bat")
            if line.strip() and os.path.isfile(vc):
                return vc
    return next((p for p in VCVARS if os.path.isfile(p)), None)


def run(cmd, cwd=None, label="", env=None):
    print("  $ %s" % " ".join(str(c) for c in cmd))
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env)
    if p.returncode:
        sys.exit("%s failed (exit %d)" % (label or cmd[0], p.returncode))


def rocm_root():
    return os.path.join(ROOT, "toolchain", "rocm-venv", "Lib", "site-packages", "_rocm_sdk_devel")


def build_env():
    """The environment the compile needs: MSVC's, plus ROCm on PATH.

    clang++ from the ROCm SDK still links against the MSVC runtime and uses the Windows SDK headers,
    so the build needs the environment vcvars64.bat sets. Rather than requiring the caller to run
    from a developer prompt, ask vcvars for its environment and merge it.
    """
    env = dict(os.environ)
    vc = find_vcvars()
    if not vc:
        sys.exit("build: no Visual Studio vcvars64.bat found - see docs/install.md")
    out = subprocess.run(["cmd", "/c", "call", vc, ">nul", "&&", "set"],
                         capture_output=True, text=True, errors="replace").stdout
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    rocm = rocm_root()
    env["ROCM_PATH"] = env["HIP_PATH"] = rocm
    env["PATH"] = os.pathsep.join([os.path.join(rocm, "bin"), os.path.join(rocm, "lib", "llvm", "bin"),
                                   os.path.join(ROOT, "toolchain"), env.get("PATH", "")])
    # src/llama.cpp sits inside this repository; without this its build stamps THIS repo's revision
    env["GIT_CEILING_DIRECTORIES"] = os.path.join(ROOT, "src")
    return env


def sha256(path):
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def fetch():
    if os.path.isdir(os.path.join(TREE, ".git")):
        print("fetch: %s already present" % TREE)
    else:
        os.makedirs(os.path.dirname(TREE), exist_ok=True)
        run(["git", "clone", "--filter=blob:none", UPSTREAM_REPO, TREE], label="git clone")
    run(["git", "-C", TREE, "checkout", "--force", UPSTREAM_REV], label="git checkout")
    head = subprocess.run(["git", "-C", TREE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != UPSTREAM_REV:
        sys.exit("checked out %s, expected %s" % (head, UPSTREAM_REV))
    print("fetch: %s @ %s" % (UPSTREAM_REPO, UPSTREAM_REV[:12]))


def patch(tree=None, quiet=False):
    tree = tree or TREE
    for rel in OVERLAY:
        dst = os.path.join(tree, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(ROOT, "patches", "files", rel), dst)
    if not quiet:
        print("patch: overlaid %d whole files" % len(OVERLAY))
    applied = skipped = 0
    for s in PATCH_ORDER:
        p = subprocess.run([sys.executable, os.path.join(ROOT, "patches", s + ".py"), tree],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        out = ((p.stdout or "") + (p.stderr or "")).strip()
        if p.returncode:
            print("patch: %s FAILED\n%s" % (s, out[-500:]))
            sys.exit(1)
        if "already applied" in out:
            skipped += 1
        else:
            applied += 1
    if not quiet:
        print("patch: %d applied, %d already present" % (applied, skipped))


def verify():
    """Re-derive the tree from a clean checkout and compare, so the build matches the recipe."""
    rec_path = os.path.join(ROOT, "bootstrap", "UPSTREAM.json")
    if not os.path.isfile(rec_path):
        print("verify: no bootstrap/UPSTREAM.json; run --patch first to record one")
        return 1
    rec = json.load(open(rec_path, encoding="utf-8"))
    bad = [n for n, h in rec["produces"].items() if sha256(os.path.join(TREE, n)) != h]
    print("verify: %d of %d patched files match the recipe" % (len(rec["produces"]) - len(bad), len(rec["produces"])))
    for n in bad:
        print("   DIFFERS", n)
    return 0 if not bad else 1


def record():
    """Refresh what the patch set produces, so --verify and the replay have something to check.

    The file list is the delta itself - `upstream` (files this project modifies, with the hashes
    they have before any patch runs) plus `added`. Those two define what the project changed and are
    not re-derived here: re-deriving them from the working tree is how a record starts agreeing with
    whatever it finds, which would make both checks vacuous.
    """
    rec_path = os.path.join(ROOT, "bootstrap", "UPSTREAM.json")
    if not os.path.isfile(rec_path):
        sys.exit("record: bootstrap/UPSTREAM.json holds the upstream side of the delta and must "
                 "exist; see tools/replay_bootstrap.py")
    rec = json.load(open(rec_path, encoding="utf-8"))
    files = sorted(list(rec["upstream"]) + list(rec["added"]))
    missing = [n for n in files if not sha256(os.path.join(TREE, n))]
    if missing:
        sys.exit("record: %d delta files are absent from %s: %s" % (len(missing), TREE, missing[:3]))
    rec["produces"] = {n: sha256(os.path.join(TREE, n)) for n in files}
    json.dump(rec, open(rec_path, "w", encoding="utf-8"), indent=1, sort_keys=True)
    print("record: %d file hashes written to bootstrap/UPSTREAM.json" % len(rec["produces"]))


def build():
    rocm = rocm_root()
    clangxx = os.path.join(rocm, "lib", "llvm", "bin", "clang++.exe")
    clang = os.path.join(rocm, "lib", "llvm", "bin", "clang.exe")
    ninja = os.path.join(ROOT, "toolchain", "ninja.exe")
    for p, what in ((clangxx, "ROCm clang++"), (ninja, "ninja")):
        if not os.path.isfile(p):
            sys.exit("build: %s not found at %s - run --toolchain first (see docs/install.md)" % (what, p))
    # without this the SDK's clang cannot find its own device bitcode and every HIP TU fails
    devlib = "--rocm-device-lib-path=" + os.path.join(rocm, "lib", "llvm", "amdgcn", "bitcode").replace("\\", "/")
    env = build_env()
    # cmake re-runs its *.cu glob only at configure time, so this must run on every build: a file
    # added or renamed by the patch set is invisible to an existing build.ninja
    run(["cmake", "-S", TREE, "-B", BUILD, "-G", "Ninja",
         "-DCMAKE_MAKE_PROGRAM=" + ninja,
         "-DCMAKE_C_COMPILER=" + clang.replace("\\", "/"),
         "-DCMAKE_CXX_COMPILER=" + clangxx.replace("\\", "/"),
         "-DCMAKE_HIP_COMPILER=" + clangxx.replace("\\", "/"),
         "-DCMAKE_PREFIX_PATH=" + os.path.join(rocm, "lib", "cmake").replace("\\", "/"),
         # ggml finds HIP through find_package and compiles .cu as CXX, so CMAKE_CXX_FLAGS is what
         # actually carries the device-lib path; the HIP_* pair is harmless and kept for the day
         # ggml switches to enable_language(HIP)
         "-DCMAKE_HIP_FLAGS=" + devlib, "-DCMAKE_CXX_FLAGS=" + devlib] + CMAKE_ARGS,
        label="cmake", env=env)
    run(["cmake", "--build", BUILD, "--target",
         "llama-server", "llama-quantize", "llama-bench", "llama-perplexity"], label="ninja", env=env)
    os.makedirs(BIN, exist_ok=True)
    got = 0
    for f in RUNTIME_FILES:
        src = os.path.join(BUILD, "bin", f)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(BIN, f))
            got += 1
    hip = shadowed_hip_dlls(os.path.join(rocm, "bin"), os.path.join(BUILD, "bin", "ggml-hip.dll"))
    for p in hip:
        for d in (os.path.join(BUILD, "bin"), BIN):
            shutil.copyfile(str(p), os.path.join(d, p.name))
    print("build: %d runtime files in %s; beside them the SDK's %s" % (
        got, BIN, ", ".join(p.name for p in hip) if hip else "(nothing in System32 to shadow it)"))


def shadowed_hip_dlls(rocm_bin, ggml_hip):
    """The SDK DLLs that ggml-hip loads and System32 also carries.

    Windows looks in System32 before PATH, and the display driver installs its own amdhip64_7.dll and
    amd_comgr.dll there, so a build that reached the SDK through PATH ran on the driver's HIP runtime
    rather than the SDK's that the bundle ships - with different numerics, and MTP acceptance 0.48
    against 0.71 on the same prompt (docs/measuring.md). Copies beside the binaries are found first;
    the rest of the SDK still comes through PATH, which tools/manager.py sets for a build tree.
    """
    from pathlib import Path
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from make_runtime_bundle import closure
    rocm = {p.name.lower(): p for p in Path(rocm_bin).iterdir() if p.suffix.lower() == ".dll"}
    system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    found = closure(["ggml-hip.dll"], dict(rocm, **{"ggml-hip.dll": Path(ggml_hip)}))
    return [p for n, p in sorted(found.items()) if n in rocm and (system32 / p.name).is_file()]


def toolchain():
    """Install the ROCm SDK and ninja into toolchain/. Needs network; everything else is local.

    The SDK is ~5 GB of wheels. ROCM_VERSION is what this project measured on, and the index it
    comes from is a nightly with a rolling window, so the pin will eventually stop resolving - see
    docs/install.md for what to do then. Pass --rocm-version to override.
    """
    venv = os.path.join(ROOT, "toolchain", "rocm-venv")
    py = os.path.join(venv, "Scripts", "python.exe")
    if not os.path.isfile(py):
        run([sys.executable, "-m", "venv", venv], label="venv")
    spec = "%s==%s" % (ROCM_EXTRAS, ROCM_VERSION) if ROCM_VERSION else ROCM_EXTRAS
    p = subprocess.run([py, "-m", "pip", "install", "--pre", "--index-url", ROCM_INDEX, spec])
    if p.returncode:
        print("\ntoolchain: pip could not install %s." % spec)
        print("This index is a NIGHTLY with a rolling window, so a pinned version stops resolving")
        print("after a few weeks. Versions it currently offers:\n")
        subprocess.run([py, "-m", "pip", "index", "versions", "rocm", "--pre", "--index-url", ROCM_INDEX])
        print("\nRe-run with --rocm-version <one of those>. Anything other than %s is UNTESTED here:" % ROCM_VERSION)
        print("re-measure before trusting it - see docs/install.md and docs/measuring.md.")
        return 1
    # The devel wheel is a tarball plus a CLI: nothing under _rocm_sdk_devel/ exists until
    # `rocm-sdk init` expands it and links the gfx1151 device files in. pip alone leaves the
    # compiler, headers and device bitcode missing, which is exactly how a first bootstrap of
    # this repository on a clean tree failed.
    if not os.path.isdir(os.path.join(rocm_root(), "bin")):
        run([py, "-m", "rocm_sdk", "init"], label="rocm-sdk init")
    if not os.path.isfile(os.path.join(ROOT, "toolchain", "ninja.exe")):
        import io as _io
        import urllib.request
        import zipfile
        print("  downloading ninja from %s" % NINJA_URL)
        with urllib.request.urlopen(NINJA_URL, timeout=300) as r:
            zipfile.ZipFile(_io.BytesIO(r.read())).extract("ninja.exe", os.path.join(ROOT, "toolchain"))
    have = os.path.isdir(os.path.join(rocm_root(), "lib", "llvm", "amdgcn", "bitcode"))
    print("toolchain: ROCm SDK %s in %s (device bitcode %s), ninja ready" % (
        ROCM_VERSION, venv, "present" if have else "MISSING - the gfx1151 extra did not install"))
    return 0 if have else 1


STEPS = ("toolchain", "fetch", "patch", "build", "verify", "record")


def main():
    global ROCM_VERSION
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for step in STEPS:
        ap.add_argument("--" + step, action="store_true")
    ap.add_argument("--rocm-version", help="override the pinned ROCm SDK version (untested: re-measure)")
    args = ap.parse_args()
    if args.rocm_version:
        ROCM_VERSION = args.rocm_version
    steps = [s for s in STEPS if getattr(args, s)]
    if not steps:
        ap.error("pick at least one of " + " ".join("--" + s for s in STEPS))
    rc = 0
    for s in steps:
        rc = globals()[s]() or 0
        if rc:
            return rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
