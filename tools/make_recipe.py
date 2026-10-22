#!/usr/bin/env python3
"""Write the whole recipe again when the fork moves to a newer upstream.

The scripts in patches/ edit upstream at one revision. Once the fork has merged a newer upstream they
no longer replay - their anchors are upstream's old code - so the recipe is regenerated from the two
commits themselves:

  - every file the fork adds, and every upstream file it rewrites more than it keeps, goes whole
    under patches/files/ (the overlay)
  - every other modified file becomes hunks in one generated script per area, patches/apply_<area>.py
  - bootstrap/UPSTREAM.json records the base (repo + revision), each modified file's upstream git blob
    and SHA-256, the overlay, the script order and the SHA-256 of every file the recipe produces

Contents come from git objects (`git cat-file`), never from a working tree, so line endings and local
edits cannot leak in. Before anything is written the recipe is replayed onto upstream in a scratch
directory and must give the fork's files exactly; tools/replay_bootstrap.py checks the same later.

    python tools/make_recipe.py --clone src/llama.cpp --base <upstream commit> --rev rulith \
        --repo https://github.com/ggml-org/llama.cpp
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from make_patch_script import pairs_for, render  # noqa: E402

# a file whose hunks would carry more than this share of its new text is kept whole: such a script is
# no easier to read than the file, and much slower to anchor
WHOLE_SHARE = 0.5

# (script area, test on the path); the first that matches wins
AREAS = [
    ("ggml_cuda", lambda p: p.startswith("ggml/src/ggml-cuda/")),
    ("ggml", lambda p: p.startswith("ggml/")),
    ("models", lambda p: p.startswith("src/models/")),
    ("llama", lambda p: p.startswith("src/") or p.startswith("include/")),
    ("common", lambda p: p.startswith("common/")),
    ("server", lambda p: p.startswith("tools/server/")),
    ("tools", lambda p: p.startswith("tools/")),
    ("tests", lambda p: p.startswith("tests/")),
    ("python", lambda p: p.endswith(".py") or p.startswith("gguf-py/") or p.startswith("conversion/")),
    ("docs", lambda p: True),
]


def git(clone, *args):
    p = subprocess.run(["git", "-C", clone] + list(args), capture_output=True)
    if p.returncode:
        sys.exit("git %s: %s" % (" ".join(args), p.stderr.decode(errors="replace").strip()))
    return p.stdout


def area_of(rel):
    return next(name for name, test in AREAS if test(rel))


def marker_for(blocks, befores):
    """a new line of the script that no file it edits has before it runs: one of the project's own
    comments if there is one, short enough to read, else the longest"""
    best, best_key = "", None
    for rel, pairs in blocks:
        for old, new in pairs:
            for line in new.splitlines():
                s = line.strip()
                if len(s) < 12 or s in old or any(s in b for b in befores.values()):
                    continue
                key = ("strixllama" in s or "STRIX_" in s, len(s) <= 120, len(s))
                if best_key is None or key > best_key:
                    best, best_key = s, key
    if not best:
        sys.exit("no marker line for the script of %s" % ", ".join(rel for rel, _ in blocks))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clone", required=True, help="a clone holding both commits")
    ap.add_argument("--base", required=True, help="the upstream commit the recipe starts from")
    ap.add_argument("--rev", required=True, help="the fork commit the recipe must give")
    ap.add_argument("--repo", required=True, help="upstream's URL, recorded for bootstrap --fetch")
    ap.add_argument("--out", default=os.path.join(ROOT, "patches"))
    ap.add_argument("--record", default=os.path.join(ROOT, "bootstrap", "UPSTREAM.json"))
    args = ap.parse_args()

    base = git(args.clone, "rev-parse", args.base).decode().strip()
    rev = git(args.clone, "rev-parse", args.rev).decode().strip()
    status = git(args.clone, "diff", "--name-status", "--no-renames", base, rev).decode().splitlines()
    changed, added = [], []
    for line in status:
        kind, rel = line.split("\t", 1)
        if kind == "M":
            changed.append(rel)
        elif kind == "A":
            added.append(rel)
        else:
            sys.exit("%s %s: the recipe has no way to delete or retype an upstream file" % (kind, rel))
    print("base %s, fork %s: %d modified, %d added" % (base[:12], rev[:12], len(changed), len(added)))

    def blob(commit, rel):
        return git(args.clone, "cat-file", "blob", "%s:%s" % (commit, rel))

    upstream, overlay, scripts = {}, {}, {}
    produces = {}
    for rel in changed:
        b, a = blob(base, rel), blob(rev, rel)
        upstream[rel] = {"git_blob": git(args.clone, "rev-parse", "%s:%s" % (base, rel)).decode().strip(),
                         "sha256": hashlib.sha256(b).hexdigest()}
        produces[rel] = hashlib.sha256(a).hexdigest()
        try:
            before, after = b.decode("utf-8"), a.decode("utf-8")
        except UnicodeDecodeError:
            overlay[rel] = a
            continue
        if "\r" in before or "\r" in after:      # the scripts read with universal newlines
            overlay[rel] = a
            continue
        pairs = pairs_for(before, after, rel)
        if sum(len(new) for _, new in pairs) > WHOLE_SHARE * len(after):
            overlay[rel] = a
            continue
        scripts.setdefault(area_of(rel), []).append((rel, pairs, before))
    for rel in added:
        a = blob(rev, rel)
        overlay[rel] = a
        produces[rel] = hashlib.sha256(a).hexdigest()

    texts = {}
    for area, entries in scripts.items():
        blocks = [(rel, pairs) for rel, pairs, _ in entries]
        befores = {rel: before for rel, _, before in entries}
        name = "apply_%s" % area
        title = "%s: the fork's changes to %d upstream file(s) of this area." % (name, len(blocks))
        why = ("Upstream %s @ %s -> fork %s. Regenerated whole by tools/make_recipe.py when the fork moved\n"
               "to this upstream; the scripts it replaced, one per change with its reasons, are in\n"
               "patches/history/ (see patches/MANIFEST.md)." % (args.repo, base[:12], rev[:12]))
        texts[name] = render(blocks, marker_for(blocks, befores), title, why, "patches/%s.py" % name)
        print("  %-16s %3d files, %4d hunks" % (name, len(blocks), sum(len(p) for _, p in blocks)))
    print("  overlay          %3d files (%d added, %d kept whole)" % (len(overlay), len(added), len(overlay) - len(added)))
    order = sorted(texts)

    # replay before writing: upstream blobs + overlay + scripts must give the fork's files exactly
    with tempfile.TemporaryDirectory() as work:
        sdir = os.path.join(work, "patches")
        os.makedirs(sdir)
        for name in order:
            open(os.path.join(sdir, name + ".py"), "w", encoding="utf-8", newline="\n").write(texts[name])
        tree = os.path.join(work, "tree")
        for rel in changed:
            dst = os.path.join(tree, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, "wb").write(blob(base, rel))
        for rel, data in overlay.items():
            dst = os.path.join(tree, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, "wb").write(data)
        for name in order:
            p = subprocess.run([sys.executable, os.path.join(sdir, name + ".py"), tree], capture_output=True, text=True)
            if p.returncode:
                sys.exit("replay: %s failed: %s" % (name, (p.stdout + p.stderr)[-400:]))
        bad = [rel for rel, h in produces.items()
               if hashlib.sha256(open(os.path.join(tree, rel), "rb").read()).hexdigest() != h]
        if bad:
            sys.exit("replay: %d files differ from the fork, e.g. %s" % (len(bad), bad[:3]))
    print("replay: %d / %d files reproduced" % (len(produces), len(produces)))

    # write
    files = os.path.join(args.out, "files")
    if os.path.isdir(files):
        shutil.rmtree(files)
    for rel, data in overlay.items():
        dst = os.path.join(files, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "wb").write(data)
    for name in order:
        open(os.path.join(args.out, name + ".py"), "w", encoding="utf-8", newline="\n").write(texts[name])
    rec = {"repo": args.repo, "revision": base, "fork_revision": rev,
           "upstream": upstream, "added": sorted(added), "overlay": sorted(overlay),
           "order": order, "produces": produces}
    json.dump(rec, open(args.record, "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    print("wrote %d scripts, %d overlay files, %s" % (len(order), len(overlay), args.record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
