#!/usr/bin/env python3
"""Build the strixllama runtime from a clean upstream checkout.

Three steps, in this order, because two of the patches anchor against what an earlier one writes:

  1. fetch  pwilkin/llama.cpp at the pinned revision into src/
  2. patch  overlay patches/iq3s-kernel/ (whole files), then apply the scripts in PATCH_ORDER
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
UPSTREAM_REPO = "https://github.com/pwilkin/llama.cpp"
UPSTREAM_REV = "f5daaa3cfa6358e5dd398911ec741813745a5440"
TREE = os.path.join(ROOT, "src", "llama.cpp")
BUILD = os.path.join(TREE, "build")
BIN = os.path.join(ROOT, "bin", "hip")   # bin/hip-rocm101 up to 0.2.4

# Whole files, laid down before the scripts run. These are the IQ3_S MMB kernel and the sigmoid
# fusion: too large to express as anchored edits, and the scripts that follow edit ggml-cuda.cu on
# top of this version of it. llama-lazy-reader.h is deliberately NOT here - apply_win_lazy_reader
# writes it, and handing a patch its own finished output makes it skip the rest of its work.
SNAPSHOT = {"ggml-cuda.cu": "ggml/src/ggml-cuda/ggml-cuda.cu",
            "mmb.cu":       "ggml/src/ggml-cuda/mmb.cu",
            "mmb.cuh":      "ggml/src/ggml-cuda/mmb.cuh",
            "softcap.cu":   "ggml/src/ggml-cuda/softcap.cu",
            "softcap.cuh":  "ggml/src/ggml-cuda/softcap.cuh"}

# Order matters where several scripts write the same file; see patches/MANIFEST.md.
PATCH_ORDER = [
    "apply_win_lazy_reader",        # first: later patches anchor against the reader it installs
    "apply_mmvq_rdna35_rows", "apply_chain_fusion", "apply_getrows_cast_fusion",
    "apply_graph_key_shape", "apply_skip_ops_ablation", "apply_ple_overlapped_reads",
    "apply_qsa_block_key_cache", "apply_qsa_decode_gather", "apply_qsa_small_batch_mask",
    "apply_qsa_kb_image_guard", "apply_rs_pos_warn_stateless", "apply_spec_draft_ubatch",
    "apply_spec_timing", "apply_decode_timing", "apply_iq3s_vecdot", "apply_node_timing",
    "apply_mmid_sort_note", "apply_ple_prefetch_launch", "apply_trunk_twins",
    "apply_iq3s_mmq_hip",           # after apply_iq3s_vecdot: it rewrites the sign table that one added
    "apply_prompt_cache_disk", "apply_multi_stream_qsa", "apply_small_m_mmvf",      # after apply_spec_timing and friends: anchors in the patched server loop
    "apply_cache_ram_and_mtp",      # after apply_prompt_cache_disk: it rewrites the tier that one adds
    "apply_spc_direct_io",          # after it: the reader it replaces is the one that patch leaves
    "apply_disk_tier_v2",           # after both: it rewrites the disk tier they built
    "apply_hc_q8_fusions", "apply_qsa3_bitonic",
    "apply_disk_restore_lazy",      # after apply_disk_tier_v2: the restore path it changes is that one's
    "apply_ple_pregather",          # after apply_disk_restore_lazy: both edit the server loop
    "apply_state_copy_trim",        # after apply_ple_pregather: both edit qwen4exp.cpp
    "apply_moe_glu3",
    "apply_qsa_active_blocks", "apply_state_read_coalesce", "apply_kv_regions", "apply_compute_buffer_headroom", "apply_alloc_failure_report", "apply_empty_slot_cache",
    "apply_qsa_kb_rebuild_f16",
    "apply_disk_tier_v3",           # after apply_empty_slot_cache and the v2 tier: it adds to the server loop they patch
    "apply_slot_state_guard", "apply_disk_ckpt_step", "apply_disk_v3_vision",
    "apply_kv_q8_0", "apply_state_hash_debug",
    "apply_spec_draft_by_slots", "apply_kv_zero_freed", "apply_mtp_carry_state",
    "apply_gdn_direct_rows", "apply_ple_unbuffered_cache", "apply_prefill_kernels_020",
    "apply_spec_even_drafts", "apply_small_k_membership",
    "apply_bf16_mid_batches", "apply_q6k_mmq_rdna35",
    "apply_small_batch_decode",
    "apply_verify_small_products",
    "apply_image_dense_bound",
    "apply_prefix_cache_026", "apply_gdn_deferred_rollback", "apply_image_rank_per_seq",
    "apply_agent_slots_027", "apply_ragged_ubatch_027",
    "apply_qsa3_wide_window_028", "apply_prompt_alone_028", "apply_disk_store_v4_028",
]

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
    vc = next((p for p in VCVARS if os.path.isfile(p)), None)
    if not vc:
        sys.exit("build: no Visual Studio 2022 vcvars64.bat found - see docs/install.md")
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
    for name, rel in SNAPSHOT.items():
        shutil.copyfile(os.path.join(ROOT, "patches", "iq3s-kernel", name), os.path.join(tree, rel))
    if not quiet:
        print("patch: overlaid %d snapshot files" % len(SNAPSHOT))
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
