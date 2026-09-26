#!/usr/bin/env python3
"""Give the MTP draft its own, smaller copy of the LM head.

Unsloth's shared-* draft files borrow output.weight from the target (nextn_shared_target_tensors), so
every draft step streams the target's Q6_K head: 248320 x 2560 = 521 MB, 2.4 ms of the 4.3 ms a draft
step costs on this machine. The draft only needs the argmax and its probability, and every token it
proposes is verified by the target, so a coarser head costs acceptance at most, never correctness.

Two stages, so the base draft tensors stay byte-identical:
  1. write a minimal GGUF (the base file's metadata + output.weight dequantized to F32) and let
     llama-quantize turn that one tensor into --type;
  2. copy the base file and append the quantized output.weight as a raw tensor.
The loader creates model.output from the draft file when it is present (TENSOR_NOT_REQUIRED) and only
falls back to the target's head when it is not, so nothing else changes.

    python tools/make_draft_head.py --base <model dir>/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
    python tools/make_draft_head.py --base ... --type q4_0 --no-quantize-tool   # numpy Q4_0, no llama-quantize

The result lands beside the base, i.e. inside a registered model directory, which is the only place
the manager lists drafts from; press Rescan afterwards and profile() prefers a *-head-* file.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "llama.cpp", "gguf-py"))   # bootstrap --fetch puts it there
sys.path.insert(0, os.path.join(ROOT, "tools"))

import numpy as np  # noqa: E402
import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType  # noqa: E402

QUANTIZE = next((p for p in (os.path.join(ROOT, "bin", d, "llama-quantize.exe") for d in ("hip", "hip-rocm101"))
                 if os.path.isfile(p)), os.path.join(ROOT, "bin", "hip", "llama-quantize.exe"))   # what bootstrap --build installs


def has_output_weight(path):
    try:
        return any(t.name == "output.weight" for t in GGUFReader(path).tensors)
    except Exception:
        return False


def find_target(base):
    """The target shard that holds output.weight: a non-draft, non-projector GGUF in the base's
    directory. The head is written next to the base for the same reason the base is searched here -
    the manager only lists files under a registered model directory, and Unsloth's mtp-*.gguf sits
    beside the model it drafts for."""
    folder = os.path.dirname(os.path.abspath(base))
    for name in sorted(os.listdir(folder)):
        low = name.lower()
        if not low.endswith(".gguf") or low.startswith(("mtp-", "mmproj")) or "-head-" in low:
            continue
        if has_output_weight(os.path.join(folder, name)):
            return os.path.join(folder, name)
    sys.exit("no GGUF with output.weight next to %s; pass --target" % base)


def copy_metadata(reader, writer):
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)


def write_gguf(path, arch, reader_for_metadata, tensors):
    """tensors: list of (name, ndarray, raw_dtype or None). Raw tensors are written as-is."""
    writer = GGUFWriter(path, arch)
    copy_metadata(reader_for_metadata, writer)
    for name, data, raw in tensors:
        if raw is None:
            writer.add_tensor(name, data)
        else:
            writer.add_tensor(name, data, raw_shape=data.shape, raw_dtype=raw)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Unsloth's mtp-*-shared-*.gguf, kept next to the model")
    ap.add_argument("--target", help="target shard that holds output.weight (default: found beside --base)")
    ap.add_argument("--type", default="iq4_xs", help="ggml type for the draft's head")
    ap.add_argument("--out", help="default: <base>-head-<type>.gguf, beside the base so the manager lists it")
    ap.add_argument("--no-quantize-tool", action="store_true", help="quantize in numpy (q4_0/q8_0 only)")
    ap.add_argument("--head-only", action="store_true",
                    help="write just the quantised head (mtp-<family>-head-<type>.gguf, ~340 MB): the release asset "
                         "the manager merges with a user's shared draft at rescan (manager.merge_draft_head)")
    args = ap.parse_args()
    if args.head_only:
        low = os.path.basename(args.base).lower()
        family = os.path.basename(args.base)[4:low.index("-shared-")]
        out = args.out or os.path.join(os.path.dirname(args.base), "mtp-%s-head-%s.gguf" % (family, args.type))
    else:
        out = args.out or args.base.replace(".gguf", "-head-%s.gguf" % args.type)
    if os.path.exists(out):
        sys.exit("%s already exists" % out)
    args.target = args.target or find_target(args.base)
    print("target shard:", args.target)

    base = GGUFReader(args.base)
    arch = base.fields[gguf.Keys.General.ARCHITECTURE].contents()
    if any(t.name == "output.weight" for t in base.tensors):
        sys.exit("base already has output.weight")

    t0 = time.time()
    tgt = GGUFReader(args.target)
    src = next(t for t in tgt.tensors if t.name == "output.weight")
    print("target output.weight: %s %s %.0f MB" % (src.tensor_type.name, list(int(x) for x in src.shape), src.n_bytes / 1e6))
    f32 = gguf.quants.dequantize(src.data, src.tensor_type)
    print("dequantized to %s in %.0fs" % (f32.shape, time.time() - t0))

    if args.no_quantize_tool:
        qtype = GGMLQuantizationType[args.type.upper()]
        q = gguf.quants.quantize(f32, qtype)
        head = (q, qtype)
        print("numpy-quantized to %s: %.0f MB" % (qtype.name, q.nbytes / 1e6))
    else:
        tmp_f32 = os.path.join(ROOT, "tmp", "draft-head-f32.gguf")
        tmp_q = os.path.join(ROOT, "tmp", "draft-head-%s.gguf" % args.type)
        os.makedirs(os.path.dirname(tmp_f32), exist_ok=True)
        write_gguf(tmp_f32, arch, base, [("output.weight", f32, None)])
        del f32
        print("wrote %s, quantizing with llama-quantize ..." % tmp_f32)
        import manager
        env = manager.runtime_environment({})   # the bundled ROCm on PATH, nothing model-specific
        p = subprocess.run([QUANTIZE, "--output-tensor-type", args.type, tmp_f32, tmp_q, "q8_0"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        if p.returncode != 0:
            print(p.stdout[-2000:]); print(p.stderr[-3000:])
            sys.exit("llama-quantize failed rc=%d" % p.returncode)
        qr = GGUFReader(tmp_q)
        qt = next(t for t in qr.tensors if t.name == "output.weight")
        head = (qt.data, qt.tensor_type)
        print("llama-quantize: output.weight -> %s %.0f MB" % (qt.tensor_type.name, qt.n_bytes / 1e6))

    tensors = [] if args.head_only else [(t.name, t.data, t.tensor_type) for t in base.tensors]
    tensors.append(("output.weight", head[0], head[1]))
    write_gguf(out, arch, base, tensors)
    print("wrote %s (%.0f MB) in %.0fs" % (out, os.path.getsize(out) / 1e6, time.time() - t0))


if __name__ == "__main__":
    main()
