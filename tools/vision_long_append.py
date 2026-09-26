#!/usr/bin/env python3
"""A long conversation with many large images, then a large text append - the shape of issue #2.

Reported on 0.2.4 (github.com/rulith-dev/strixllama/issues/2): about 58K tokens of text, then 27 images of
1920x1080 over seven turns with the whole history kept (~114K tokens), a short text turn, then ~14K more
tokens of text. With batch = ubatch = 8192 the last request killed the server with "ROCm error: unspecified
launch failure"; at 1024 it completed, and so did the same lengths without images. This walks the same
shape with generated records and images, so it needs nothing but a running server with a projector loaded
and a single slot (image input needs one):

    python tools/vision_long_append.py                   # the reported sizes
    python tools/vision_long_append.py --images 8 --head 20000 --append 6000   # a quicker smaller variant

It prints each turn's prompt size and time and ends with LONG APPEND OK, or FAILED when a request errors or
the server stops answering. Requests are those of the report: temperature 0, seed 42, 64 tokens, no stream,
cache_prompt on, thinking off.
"""
import argparse
import base64
import json
import struct
import sys
import time
import urllib.request
import zlib

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def png(w, h, k):
    """A w x h RGB image, different for every k: diagonal colour bands and a frame number drawn as bars.
    Encoded by hand with zlib, as tools/vision_stress.py does - no image library."""
    base = ((37 * k) % 256, (91 * k + 60) % 256, (153 * k + 120) % 256)
    rows = []
    for y in range(h):
        row = bytearray(w * 3)
        for x in range(0, w, 8):
            band = ((x + y + 40 * k) // 64) % 4
            px = bytes(((base[0] + 50 * band) % 256, (base[1] + 30 * band) % 256, (base[2] + 70 * band) % 256)) * 8
            row[x * 3:x * 3 + 24] = px[:max(0, min(24, (w - x) * 3))]
        if 40 <= y < 120:                                         # the frame number as k white bars
            for b in range(k + 1):
                x0 = 40 + b * 24
                row[x0 * 3:(x0 + 12) * 3] = b"\xff" * 36
        rows.append(b"\x00" + bytes(row))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 6)) + chunk(b"IEND", b""))


def records(first, n):
    """Synthetic repetitive numbered records, as in the report."""
    return "".join("Record %06d: station=S%03d reading=%d.%02d status=%s shift=%s\n" % (
        i, i % 211, (i * 7919) % 1000, i % 100, ("ok", "warn", "ok", "check")[i % 4], ("day", "night")[i % 2])
        for i in range(first, first + n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--head", type=int, default=58000, help="approximate tokens of text in the first turn")
    ap.add_argument("--images", type=int, default=27)
    ap.add_argument("--groups", default="4,4,4,4,4,4,3", help="images per turn")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--append", type=int, default=14000, help="approximate tokens of text in the last turn")
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.split("x"))
    groups = [int(g) for g in args.groups.split(",")]
    while sum(groups) > args.images:
        groups[-1] -= 1
        if groups[-1] == 0:
            groups.pop()
    # tokens per record, from the server's own tokenizer, so --head and --append mean tokens
    req = urllib.request.Request("http://127.0.0.1:%d/tokenize" % args.port, data=json.dumps(
        {"content": records(0, 200)}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        per_record = len(json.loads(r.read())["tokens"]) / 200

    def chat(msgs, tag):
        body = {"messages": msgs, "max_tokens": 64, "temperature": 0, "seed": 42, "stream": False,
                "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % args.port,
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                d = json.loads(r.read().decode("utf-8", "replace"))
            a = d["choices"][0]["message"]["content"] or ""
            t = d.get("timings", {})
            print("  %-8s ok %6.1fs prompt=%-7d prefill %6.1f t/s  %s" % (
                tag, time.time() - t0, d["usage"]["prompt_tokens"], t.get("prompt_per_second", 0.0),
                a.replace("\n", " ")[:50]), flush=True)
            return a
        except Exception as e:
            print("  %-8s FAILED %.1fs: %s %s" % (tag, time.time() - t0, type(e).__name__, e), flush=True)
            return None

    n_head = int(args.head / per_record)
    msgs = [{"role": "user", "content": records(0, n_head) + "\nHow many records have status=warn? Estimate."}]
    a = chat(msgs, "text")
    if a is None:
        print("FAILED")
        return 1
    msgs.append({"role": "assistant", "content": a})
    k = 0
    for g, n in enumerate(groups, 1):
        content = []
        for _ in range(n):
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                                                                   base64.b64encode(png(w, h, k)).decode()}})
            k += 1
        content.append({"type": "text", "text": "Frames %d-%d. How many white bars does each frame show?" % (k - n, k - 1)})
        msgs.append({"role": "user", "content": content})
        a = chat(msgs, "images%d" % g)
        if a is None:
            print("FAILED")
            return 1
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": "Summarise the frames in one sentence."})
    a = chat(msgs, "short")
    if a is None:
        print("FAILED")
        return 1
    msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": records(n_head, int(args.append / per_record)) +
                 "\nWhich station appears most often in these new records?"})
    if chat(msgs, "append") is None:
        print("FAILED")
        return 1
    print("LONG APPEND OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
