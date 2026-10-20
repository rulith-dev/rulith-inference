#!/usr/bin/env python3
"""Multi-turn image conversation with a divergent branch - the shape of three server aborts.

Enabling image input produced three separate aborts in the sparse-attention block machinery, and
**none of them reproduced on a single-turn test at short context**. Two needed the third or fourth
turn of a growing image conversation, and one needed a divergent branch, which makes the server
reuse a slot while keeping the cached prefix. So this walks a conversation to ~17K tokens with an
image in the first turn, then branches twice off an earlier point.

Note that running it leaves the server's slot without its block-key cache, which costs ~7% decode
at long context until the server is restarted. Do not take a timing measurement after it.

Run it against anything that touches sparse attention, M-RoPE or the KV cache:

    python tools/vision_stress.py                 # generated image and filler text
    python tools/vision_stress.py --text book.txt # real prose instead of filler

It is a crash probe, not a quality check: it fails if a turn errors or the server dies, and prints
STRESS OK otherwise. The image is generated here rather than shipped, so the only thing needed is a
running server with a projector loaded.
"""
import argparse
import base64
import json
import os
import struct
import sys
import time
import urllib.request
import zlib

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

W = H = 448
SHAPES = "a red circle, a green triangle and a blue square"


def png_with_shapes():
    """A white image with a red circle, a green triangle and a blue square, encoded by hand.

    No image library: the point is that this test runs anywhere, and a PNG of three flat shapes is
    a few lines of zlib.
    """
    rows = []
    for y in range(H):
        row = bytearray(b"\xff" * (W * 3))
        for x in range(W):
            r, g, b = 255, 255, 255
            if (x - 110) ** 2 + (y - 110) ** 2 <= 70 ** 2:              # circle, red
                r, g, b = 220, 40, 40
            elif 250 <= y <= 390 and abs(x - 120) <= (y - 250) // 2:     # triangle, green
                r, g, b = 40, 170, 60
            elif 280 <= x <= 400 and 60 <= y <= 180:                     # square, blue
                r, g, b = 50, 70, 210
            row[x * 3:x * 3 + 3] = bytes((r, g, b))
        rows.append(b"\x00" + bytes(row))                                # filter type 0 per scanline

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 6))
            + chunk(b"IEND", b""))


def filler(n_chars):
    """Prose-shaped text. A crash probe cares about token counts and cache behaviour, not meaning."""
    words = ("the quick sparse attention kernel gathers selected cells into a compact pair before "
             "the ordinary flash attention path reads them while positions advance past the image "
             "grid and the block key cache records which blocks are finished ").split()
    out, i = [], 0
    while sum(len(w) + 1 for w in out) < n_chars:
        out.append(words[i % len(words)])
        if i % 17 == 16:
            out.append("\n")
        i += 1
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("STRIX_TEST_PORT", "8080")))  # with_server sets it
    ap.add_argument("--text", help="a long text file to use instead of generated filler")
    ap.add_argument("--turns", type=int, default=7)
    # English runs ~6 characters per token here, so these are sized to pass the 12514-token mark
    # the aborts appeared at, and to finish near 17K
    ap.add_argument("--head", type=int, default=60000, help="characters of text in the first turn")
    ap.add_argument("--step", type=int, default=6000, help="characters added per later turn")
    args = ap.parse_args()

    need = args.head + args.turns * args.step
    if args.text:
        doc = open(args.text, encoding="utf-8", errors="replace").read()
        if len(doc) < need:
            sys.exit("%s is %d characters; this needs at least %d" % (args.text, len(doc), need))
    else:
        doc = filler(need)
    b64 = base64.b64encode(png_with_shapes()).decode()

    def chat(msgs, tag, n=60):
        body = {"messages": msgs, "max_tokens": n, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % args.port,
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))
            a = d["choices"][0]["message"]["content"]
            print("  %-10s ok %5.1fs prompt=%-6d %s" % (
                tag, time.time() - t0, d["usage"]["prompt_tokens"], a.replace("\n", " ")[:60]), flush=True)
            return a
        except Exception as e:
            print("  %-10s FAILED %.1fs: %s %s" % (tag, time.time() - t0, type(e).__name__, e), flush=True)
            return None

    base = [{"role": "user", "content": [
        {"type": "text", "text": doc[:args.head]},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
        {"type": "text", "text": "Name the shapes in the image. Be brief."}]}]
    a = chat(base, "turn1")
    if a is None:
        return 1
    msgs = base + [{"role": "assistant", "content": a}]
    # grow the conversation past 12K with extra text each turn, as the abort log's 12514 tokens did
    for i in range(2, args.turns + 1):
        msgs = msgs + [{"role": "user", "content": doc[args.head + (i - 2) * args.step:
                                                       args.head + (i - 1) * args.step] +
                        "\n\nIn one sentence, what colour is the circle in the image?"}]
        a = chat(msgs, "turn%d" % i)
        if a is None:
            return 1
        msgs = msgs + [{"role": "assistant", "content": a}]
    # a divergent branch: same prefix, different tail -> slot reuse with the cached prefix kept
    for k, q in enumerate(("Name the triangle's colour.", "Name the square's colour."), 1):
        branch = msgs[:-2] + [{"role": "user", "content": q}]
        if chat(branch, "branch%d" % k) is None:
            return 1
    print("STRESS OK  (the image holds %s)" % SHAPES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
