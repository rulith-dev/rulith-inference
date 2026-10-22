#!/usr/bin/env python3
"""Generate a patch script from two versions of a file, with anchors that are unique by construction.

The hand-written scripts in patches/ drifted: three of them now fail against clean upstream because
their anchors were copied from an already-modified tree. This closes that loop mechanically. Give it
the "before" tree (what the earlier patches produce) and the "after" tree (what must come out), and
it emits a script in the same shape as the rest of patches/: a list of (old, new) pairs, each checked
to occur exactly once before it is replaced.

Each hunk's anchor starts as the lines that actually change and grows by one context line at a time
until it is unique in the before-file, so a generated anchor cannot be ambiguous.

    python tools/make_patch_script.py --before tmp/replay --after src/llama.cpp \
        --file src/llama-model.cpp --out patches/apply_x.py --marker LLAMA_TRUNK_DECODE_Q6K \
        --title "..." --why "..."
"""
import argparse
import difflib
import io
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def hunks(before, after):
    """(old, new) line-block pairs, in order, from a line diff."""
    a, b = before.splitlines(keepends=True), after.splitlines(keepends=True)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag != "equal":
            out.append((i1, i2, j1, j2))
    return a, b, out


def anchored(a, b, i1, i2, j1, j2, text):
    """Grow context around a hunk until the old block occurs exactly once in the whole file."""
    for pad in range(0, 400):
        lo, hi = max(0, i1 - pad), min(len(a), i2 + pad)
        old = "".join(a[lo:hi])
        if old and text.count(old) == 1:
            new = "".join(a[lo:i1]) + "".join(b[j1:j2]) + "".join(a[i2:hi])
            return old, new, lo, hi
    return None, None, None, None


def merge_close(hs, gap):
    """hunks whose old ranges sit within `gap` lines of each other, as one"""
    out = []
    for i1, i2, j1, j2 in hs:
        if out and i1 - out[-1][1] <= gap:
            out[-1] = (out[-1][0], i2, out[-1][2], j2)
        else:
            out.append((i1, i2, j1, j2))
    return out


def anchored_pairs(a, b, hs, before, after):
    """The finest split that replays, else nearby hunks merged with a doubling gap, else one hunk."""
    pairs = anchored_pairs_once(a, b, hs, before, after)
    if pairs is not None:
        return pairs
    # a rewrite that keeps common lines diffs into many small hunks in one region, and applying them in
    # order can plant a later anchor inside an earlier hunk's new text; the fix is coarser hunks, not one
    # hunk from the first change to the last (which takes in every unchanged line between)
    gap = 8
    while gap <= max(len(a), 1):
        merged = merge_close(hs, gap)
        pairs = anchored_pairs_once(a, b, merged, before, after)
        if pairs is not None:
            return pairs
        if len(merged) == 1:
            break
        gap *= 2
    return anchored_pairs_single(a, b, hs, before, after)


def anchored_pairs_once(a, b, hs, before, after):
    """Anchor every hunk, merging neighbours whose context would overlap: the script applies hunks in
    order, so a later anchor that reaches into an earlier hunk's lines no longer matches once that
    hunk has been applied (found out the hard way on four hunks of one loop body). The result is
    replayed against `before` and must give `after` exactly, or None."""
    pairs, k = [], 0
    while k < len(hs):
        i1, i2, j1, j2 = hs[k]
        while True:
            old, new, lo, hi = anchored(a, b, i1, i2, j1, j2, before)
            if old is None:
                return None
            if k + 1 < len(hs) and hs[k + 1][0] <= hi:      # next hunk starts inside this anchor
                k += 1
                i2, j2 = hs[k][1], hs[k][3]
                continue
            break
        pairs.append((old, new))
        k += 1
    text = before
    for old, new in pairs:
        if text.count(old) != 1:
            pairs = None
            break
        text = text.replace(old, new, 1)
    if pairs is not None and text == after:
        return pairs
    return None


def anchored_pairs_single(a, b, hs, before, after):
    """one hunk spanning every change"""
    i1, i2, j1, j2 = hs[0][0], hs[-1][1], hs[0][2], hs[-1][3]
    old, new, lo, hi = anchored(a, b, i1, i2, j1, j2, before)
    if old is None or before.replace(old, new, 1) != after:
        return None
    return [(old, new)]


def pairs_for(before, after, rel):
    """the (old, new) pairs that turn `before` into `after`; exits when no anchoring replays"""
    a, b, hs = hunks(before, after)
    pairs = anchored_pairs(a, b, hs, before, after)
    if pairs is None:
        sys.exit("could not build unique anchors that replay to the target in %s" % rel)
    return pairs


def render(blocks, marker, title, why, out):
    """the patch script for `blocks` = [(relative path, pairs)], as text"""
    total = sum(len(pairs) for _, pairs in blocks)

    def lit(s):
        return "'''" + s.replace("\\", "\\\\").replace("'''", "\\'\\'\\'") + "'''"

    body = ['#!/usr/bin/env python3', '"""%s' % title, '']
    if why:
        body += [why, '']
    body += ['Generated by tools/make_patch_script.py from the tree it must reproduce, so every anchor is',
             'unique by construction. %d hunks across %d file(s).' % (total, len(blocks)), '',
             'Usage: python %s [tree]' % out.replace("\\", "/"), '"""',
             'import io', 'import os', 'import sys', '',
             'ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))',
             'tree = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "src", "llama.cpp")',
             'MARKER = %r' % marker, '', 'EDITS = {']
    for rel, pairs in blocks:
        parts = rel.split("/")
        body.append('    os.path.join(tree, %s): [' % ", ".join('"%s"' % p for p in parts))
        for old, new in pairs:
            body.append('        (%s,' % lit(old))
            body.append('         %s),' % lit(new))
        body.append('    ],')
    body += ['}', '', '',
             'def main():',
             '    texts = {p: io.open(p, encoding="utf-8").read() for p in EDITS}',
             '    if any(MARKER in t for t in texts.values()):',
             '        print("already applied")',
             '        return',
             '    for p, pairs in EDITS.items():',
             '        for old, new in pairs:',
             '            if texts[p].count(old) != 1:',
             '                sys.exit("anchor found %d times, expected once, in %s" % (texts[p].count(old), p))',
             '            texts[p] = texts[p].replace(old, new, 1)',
             '    for p, t in texts.items():',
             '        io.open(p, "w", encoding="utf-8", newline="").write(t)',
             '    print("patched", ", ".join(os.path.basename(p) for p in EDITS))',
             '', '',
             'if __name__ == "__main__":',
             '    main()', '']
    return "\n".join(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--file", required=True, action="append", dest="files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--marker", required=True, help="string that proves the patch is already applied")
    ap.add_argument("--title", default="Generated patch.")
    ap.add_argument("--why", default="")
    args = ap.parse_args()

    blocks = []
    for rel in args.files:
        before = io.open(os.path.join(args.before, rel), encoding="utf-8", errors="replace").read()
        after = io.open(os.path.join(args.after, rel), encoding="utf-8", errors="replace").read()
        if before == after:
            print("  %s: already identical, skipped" % rel)
            continue
        pairs = pairs_for(before, after, rel)
        blocks.append((rel, pairs))
        print("  %s: %d hunks" % (rel, len(pairs)))

    if not blocks:
        print("nothing to generate")
        return 1

    io.open(args.out, "w", encoding="utf-8", newline="\n").write(
        render(blocks, args.marker, args.title, args.why, args.out))
    print("wrote %s (%d hunks)" % (args.out, sum(len(p) for _, p in blocks)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
