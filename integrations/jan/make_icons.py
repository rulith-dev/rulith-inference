#!/usr/bin/env python3
"""Generate the strixllama application icon: SVG source, PNGs, a Windows .ico and a macOS .icns.

    python integrations/jan/make_icons.py          # writes integrations/jan/icons/

The mark is an owl's face inside a ring. `strix` is the owl genus and also the `strix` of Strix
Halo, the platform this targets; the ring is the halo. Two large eyes are the strongest shape an
icon can have at 16 px, which is the size that decides whether an icon works.

Everything is drawn here rather than rasterised from the SVG, because rasterising SVG needs a
library and nothing else in this repository has a dependency. The primitives are analytic
(rounded rectangle, ring, disc, triangle) and sampled 4x4 per pixel for antialiasing, which for
flat shapes is indistinguishable from a real renderer.

Below 24 px the ring is dropped and the eyes grow: at that size the ring is under one pixel wide
and only muddies the silhouette. Icons are hinted, not scaled.

The colours are Rulith's: the tile in Rulith green, the face in near-black, the pupils cut back to
the tile. Up to 0.2.4 it was a gold owl on a dark slate tile.
"""
import math
import os
import struct
import sys
import zlib

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "icons")

BG = (0x2F, 0xD6, 0x9A)                   # Rulith green (--rl-green on dark)
RING = (0x1F, 0xA8, 0x77)
FACE = (0x0F, 0x14, 0x11)
SS = 4                                    # supersampling factor per axis


def shapes(small):
    """The mark, in a 128-unit square. `small` is the hinted variant for 16-24 px."""
    s = [("rrect", 0, 0, 128, 128, 28, BG)]
    if small:
        # no ring, bigger eyes, heavier beak - the silhouette has to survive 16x16
        s += [("disc", 44, 57, 21, FACE), ("disc", 44, 57, 8.5, BG),
              ("disc", 84, 57, 21, FACE), ("disc", 84, 57, 8.5, BG),
              ("tri", (64, 74), (73, 92), (55, 92), FACE)]
    else:
        s += [("ring", 64, 60, 44, 5, RING),
              ("disc", 46, 58, 17, FACE), ("disc", 46, 58, 7, BG),
              ("disc", 82, 58, 17, FACE), ("disc", 82, 58, 7, BG),
              ("tri", (64, 72), (71, 85), (57, 85), FACE)]
    return s


def inside(shape, x, y):
    kind = shape[0]
    if kind == "rrect":
        _, x0, y0, w, h, r, _ = shape
        cx = min(max(x, x0 + r), x0 + w - r)
        cy = min(max(y, y0 + r), y0 + h - r)
        return (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    if kind == "disc":
        _, cx, cy, r, _ = shape
        return (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    if kind == "ring":
        _, cx, cy, r, w, _ = shape
        d = math.hypot(x - cx, y - cy)
        return r - w / 2 <= d <= r + w / 2
    if kind == "tri":
        _, a, b, c, _ = shape

        def side(p, q):
            return (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0])
        d1, d2, d3 = side(a, b), side(b, c), side(c, a)
        return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def render(size):
    """RGBA rows, top to bottom. Transparent wherever the rounded rectangle is not."""
    sh = shapes(size < 24)
    step = 128.0 / size
    rows = []
    for py in range(size):
        row = bytearray()
        for px in range(size):
            r = g = b = a = 0
            for sy in range(SS):
                for sx in range(SS):
                    x = (px + (sx + 0.5) / SS) * step
                    y = (py + (sy + 0.5) / SS) * step
                    col = None
                    for shape in sh:                       # last shape wins: painter's order
                        if inside(shape, x, y):
                            col = shape[-1]
                    if col is not None:
                        r += col[0]
                        g += col[1]
                        b += col[2]
                        a += 255
            n = SS * SS
            # premultiplied average would darken the edge against the transparent background, so
            # average the colour over covered samples only and carry coverage in alpha
            cov = a // 255
            if cov:
                row += bytes((r // cov, g // cov, b // cov, a // n))
            else:
                row += b"\0\0\0\0"
        rows.append(bytes(row))
    return rows


def png(size):
    rows = render(size)
    raw = b"".join(b"\x00" + r for r in rows)

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


def ico(images):
    """ICO with PNG payloads - supported by Windows Vista and later at every size."""
    head = struct.pack("<HHH", 0, 1, len(images))
    entries, blobs = b"", b""
    offset = len(head) + 16 * len(images)
    for size, data in images:
        entries += struct.pack("<BBBBHHII", size if size < 256 else 0, size if size < 256 else 0,
                               0, 0, 1, 32, len(data), offset)
        blobs += data
        offset += len(data)
    return head + entries + blobs


def icns(images):
    """ICNS with PNG payloads. Only the sizes macOS actually reads are emitted."""
    types = {16: b"icp4", 32: b"icp5", 64: b"icp6", 128: b"ic07", 256: b"ic08", 512: b"ic09"}
    body = b""
    for size, data in images:
        if size in types:
            body += types[size] + struct.pack(">I", len(data) + 8) + data
    return b"icns" + struct.pack(">I", len(body) + 8) + body


def hexc(c):
    return "#%02X%02X%02X" % c


def svg():
    """The source of truth for anyone who wants to redraw it properly."""
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" width="128" height="128">
  <title>strixllama</title>
  <desc>An owl's face inside a ring: strix, the owl genus and the strix of Strix Halo; the ring is the halo.</desc>
  <rect width="128" height="128" rx="28" fill="{bg}"/>
  <circle cx="64" cy="60" r="44" fill="none" stroke="{ring}" stroke-width="5"/>
  <circle cx="46" cy="58" r="17" fill="{face}"/><circle cx="46" cy="58" r="7" fill="{bg}"/>
  <circle cx="82" cy="58" r="17" fill="{face}"/><circle cx="82" cy="58" r="7" fill="{bg}"/>
  <path d="M64 72 L71 85 L57 85 Z" fill="{face}"/>
</svg>
""".format(bg=hexc(BG), ring=hexc(RING), face=hexc(FACE))


# what Tauri's bundle.icon list and the Windows Store tiles expect
PNG_SIZES = [16, 32, 44, 64, 71, 89, 107, 128, 142, 150, 256, 284, 310, 512]
NAMED = {32: "32x32.png", 64: "64x64.png", 128: "128x128.png", 256: "128x128@2x.png", 512: "icon.png"}
TILES = {30: "Square30x30Logo.png", 44: "Square44x44Logo.png", 71: "Square71x71Logo.png",
         89: "Square89x89Logo.png", 107: "Square107x107Logo.png", 142: "Square142x142Logo.png",
         150: "Square150x150Logo.png", 284: "Square284x284Logo.png", 310: "Square310x310Logo.png",
         50: "StoreLogo.png"}


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "strixllama.svg"), "w", encoding="utf-8") as f:
        f.write(svg())
    cache = {}

    def get(size):
        if size not in cache:
            cache[size] = png(size)
        return cache[size]

    for size, name in NAMED.items():
        open(os.path.join(OUT, name), "wb").write(get(size))
    for size, name in TILES.items():
        open(os.path.join(OUT, name), "wb").write(get(size))
    open(os.path.join(OUT, "icon.ico"), "wb").write(
        ico([(s, get(s)) for s in (16, 32, 48, 64, 128, 256)]))
    open(os.path.join(OUT, "icon.icns"), "wb").write(
        icns([(s, get(s)) for s in (16, 32, 64, 128, 256, 512)]))
    n = len(NAMED) + len(TILES) + 3
    print("wrote %d files to %s" % (n, os.path.relpath(OUT, os.path.dirname(HERE))))


if __name__ == "__main__":
    main()
