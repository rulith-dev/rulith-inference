#!/usr/bin/env python3
"""Prove (or disprove) that the patch set rebuilds the tree it claims to.

bootstrap.py --verify answers "is my tree what the recipe produced?" by comparing hashes. That is
the weaker question: it passes just as happily if the tree was edited by hand and the hashes were
re-recorded afterwards. This answers the stronger one - replay the whole recipe from clean upstream
and see whether the result is byte-identical:

    clean upstream at the pinned revision
      + patches/iq3s-kernel/       (the whole-file snapshot)
      + PATCH_ORDER                (every script, in order)
      = the 29 files of the delta, exactly

    python tools/replay_bootstrap.py            # offline, using the clone's own git objects
    python tools/replay_bootstrap.py --fetch    # no clone objects: download the 25, hash-checked

Clean upstream is reconstructed rather than re-cloned. Only 25 files differ from upstream, so only
those need restoring, and bootstrap/UPSTREAM.json records each one's git blob hash and SHA-256.
`git cat-file blob <hash>` returns content addressed by that hash, so a file restored this way is
upstream's by construction - no network, and nothing to trust. --fetch is the fallback when the
clone has no objects (a tarball, or a blobless clone that has since been pruned): it downloads
through a mirror and checks each file against the recorded SHA-256 before using it.

Exit code is 0 only when all 29 files match. Anything else means the recipe no longer describes the
tree, which is the failure mode this project cares most about - a patch that silently stops applying
leaves a build nobody can reproduce.
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
sys.path.insert(0, os.path.join(ROOT, "bootstrap"))
# one source of truth for what to apply and in what order; a second copy here is exactly the kind of
# drift this tool exists to catch
from bootstrap import SNAPSHOT, PATCH_ORDER, TREE  # noqa: E402

# the patches only touch these; copying the rest of upstream would be minutes of I/O for nothing
SUBS = ["ggml/src/ggml-cuda", "ggml/include", "src", "common", "tools/server", "tools/mtmd"]


def sha256(path):
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def restore_from_git(tree, rel, blob, dst):
    """Ask the clone for the upstream blob by hash. Authenticated by git's content addressing."""
    p = subprocess.run(["git", "-C", tree, "cat-file", "blob", blob], capture_output=True)
    if p.returncode or not p.stdout:
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as f:
        f.write(p.stdout)
    return True


def restore_from_mirror(rel, want, dst, repo, rev, mirror):
    """Download one file and refuse it unless it hashes to what the record says."""
    if sha256(dst) == want:
        return True
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = "%s/https://raw.githubusercontent.com/%s/%s/%s" % (
        mirror.rstrip("/"), repo.split("github.com/", 1)[-1].rstrip("/"), rev, rel)
    subprocess.run(["curl", "-s", "--noproxy", "*", "-m", "120", "-o", dst, url], capture_output=True)
    return sha256(dst) == want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=TREE, help="the checkout to replay against (default src/llama.cpp)")
    ap.add_argument("--work", default=os.path.join(ROOT, "tmp", "replay"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "tmp", "upstream"), help="downloaded upstream files")
    ap.add_argument("--mirror", default="https://gh-proxy.com", help="prefix for raw.githubusercontent.com")
    ap.add_argument("--fetch", action="store_true", help="download upstream files instead of reading git objects")
    args = ap.parse_args()

    rec = json.load(open(os.path.join(ROOT, "bootstrap", "UPSTREAM.json"), encoding="utf-8"))
    changed, added = sorted(rec["upstream"]), rec["added"]
    if not os.path.isdir(args.tree):
        sys.exit("no checkout at %s - run: python bootstrap/bootstrap.py --fetch" % args.tree)
    print("upstream : %s @ %s" % (rec["repo"], rec["revision"][:12]))
    print("delta    : %d modified + %d added" % (len(changed), len(added)))

    # 1. a scratch copy of the tree, then the delta files rolled back to upstream
    shutil.rmtree(args.work, ignore_errors=True)
    for s in SUBS:
        shutil.copytree(os.path.join(args.tree, s), os.path.join(args.work, s))
    how = {"git": 0, "mirror": 0}
    for n in changed:
        dst = os.path.join(args.work, n)
        if not args.fetch and restore_from_git(args.tree, n, rec["upstream"][n]["git_blob"], dst):
            how["git"] += 1
            continue
        cached = os.path.join(args.cache, n)
        if not restore_from_mirror(n, rec["upstream"][n]["sha256"], cached,
                                   rec["repo"], rec["revision"], args.mirror):
            sys.exit("could not obtain an authentic upstream copy of %s" % n)
        shutil.copyfile(cached, dst)
        how["mirror"] += 1
    for a in added:                      # added files do not exist upstream
        p = os.path.join(args.work, a)
        if os.path.isfile(p):
            os.remove(p)

    dirty = [n for n in changed if sha256(os.path.join(args.work, n)) != rec["upstream"][n]["sha256"]]
    print("restored : %d from git objects, %d from %s" % (how["git"], how["mirror"], args.mirror))
    print("clean    : %s" % ("upstream, verified" if not dirty else "FAILED %s" % dirty[:3]))
    if dirty:
        return 2

    # 2. replay the recipe
    for name, rel in SNAPSHOT.items():
        shutil.copyfile(os.path.join(ROOT, "patches", "iq3s-kernel", name), os.path.join(args.work, rel))
    failed = []
    for s in PATCH_ORDER:
        p = subprocess.run([sys.executable, os.path.join(ROOT, "patches", s + ".py"), args.work],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
        if p.returncode:
            failed.append(s)
    print("patches  : %d applied, %d failed %s\n" % (len(PATCH_ORDER) - len(failed), len(failed), failed or ""))

    # 3. compare against what the record says the recipe produces
    same, diff = [], []
    for n in changed + added:
        (same if sha256(os.path.join(args.work, n)) == rec["produces"][n] else diff).append(n)
    print("REPRODUCED %d / %d" % (len(same), len(same) + len(diff)))
    for n in diff:
        print("   DIFFERS", n)
    return 0 if not diff else 1


if __name__ == "__main__":
    sys.exit(main())
