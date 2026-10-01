#!/usr/bin/env python3
"""Monte Carlo check of speculative sampling (patches/apply_spec_sampling_042.py).

Slices the functions under test out of the built tree's common/sampling.cpp (spec_draw, spec_q_of,
common_sampler_spec_draw_q, spec_select), compiles them alone with tools/spec_sampling_mc.cpp using the SDK's
clang++ (headers only, no llama libraries) and runs it: a token drawn from q and verified against p must come out
distributed as p (35 parameter sets), and streams of drafts cut by a confidence rule must give p's joint law. A last
stream that drops drafts shorter than two is expected to come out biased - why the drafter keeps them.

Usage: python tools/spec_sampling_mc.py   (about two minutes; exit code 0 when every check passes)
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TREE = os.path.join(ROOT, "src", "llama.cpp")
OUT = os.path.join(ROOT, "tmp", "spec_sampling_mc")
sys.path.insert(0, os.path.join(ROOT, "bootstrap"))
import bootstrap as b  # noqa: E402

src = open(os.path.join(TREE, "common", "sampling.cpp"), encoding="utf-8").read()
i0 = src.index("// an index of cur_p drawn with probabilities proportional to w")
i1 = src.index("// STRIX_SPEC_SAMPLING_STATS=1 (measurement only)")
os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "spec_fns.inc"), "w", encoding="utf-8").write(src[i0:i1])

clangxx = os.path.join(b.rocm_root(), "lib", "llvm", "bin", "clang++.exe")
exe = os.path.join(OUT, "spec_sampling_mc.exe")
subprocess.run([clangxx, "-O2", "-std=c++17", "-I", os.path.join(TREE, "include"), "-I", os.path.join(TREE, "ggml", "include"),
                "-I", OUT, os.path.join(ROOT, "tools", "spec_sampling_mc.cpp"), "-o", exe], check=True)
sys.exit(subprocess.run([exe]).returncode)
