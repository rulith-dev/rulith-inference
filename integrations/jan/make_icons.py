#!/usr/bin/env python3
"""Generate the Rulith Inference application icon: SVG source, PNGs, a Windows .ico and a macOS .icns.

    python integrations/jan/make_icons.py [OUT_DIR]     # default: integrations/jan/icons/

The mark is a white capital R on a rounded square with a diagonal blue gradient, #3B82F6 at the
top left to #1E3A8A at the bottom right (picked 10-18; it replaced the green owl of 0.2.5-0.5.x).
White on blue keeps the contrast high in both the light and the dark theme, since the tile carries
its own background.

Every size is rendered from the same vector paths (100-unit viewBox) with 8x8 supersampling and a
box filter in premultiplied alpha, so a 16 px icon is as crisp as the 512 px one - no size is a
downscale of another. Needs Pillow and numpy.
"""
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))

TOP_LEFT = (0x3B, 0x82, 0xF6)
BOTTOM_RIGHT = (0x1E, 0x3A, 0x8A)
WHITE = (0xFF, 0xFF, 0xFF)
VIEW = 100.0
RADIUS = 22.0
SHIFT = (-3.0, 0.0)                       # the R sits optically centred with translate(-3 0)
SS = 8                                    # supersampling factor per axis

# the R, evenodd: the body with its counter as a hole, plus the leg
BODY = "M30 22H56C68 22 76 30 76 40C76 48.5 71 54.5 63.5 56.5L49.5 58H44V78H30Z"
COUNTER = "M44 33.5V47H55.5C59.5 47 62 44.3 62 40.25C62 36.2 59.5 33.5 55.5 33.5Z"
LEG = "M49.5 58L63.5 56.5L77 78H61.5Z"


def path_points(d, steps=64):
    """Polygon for an absolute M/L/H/V/C/Z path (the only commands the mark uses)."""
    import re
    toks = re.findall(r"[MLHVCZ]|-?\d+(?:\.\d+)?", d)
    pts, i, cmd = [], 0, None
    x = y = 0.0
    while i < len(toks):
        if toks[i].isalpha():
            cmd = toks[i]
            i += 1
            if cmd == "Z":
                continue
        if cmd in ("M", "L"):
            x, y = float(toks[i]), float(toks[i + 1])
            i += 2
            pts.append((x, y))
        elif cmd == "H":
            x = float(toks[i])
            i += 1
            pts.append((x, y))
        elif cmd == "V":
            y = float(toks[i])
            i += 1
            pts.append((x, y))
        elif cmd == "C":
            x1, y1, x2, y2, x3, y3 = (float(t) for t in toks[i:i + 6])
            i += 6
            for k in range(1, steps + 1):
                t = k / steps
                u = 1 - t
                pts.append((u ** 3 * x + 3 * u * u * t * x1 + 3 * u * t * t * x2 + t ** 3 * x3,
                            u ** 3 * y + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t ** 3 * y3))
            x, y = x3, y3
        else:
            raise ValueError("unsupported path command %r" % cmd)
    return pts


def render(size):
    """A size x size RGBA image of the icon."""
    n = size * SS
    k = n / VIEW

    def poly(d, dx=0.0, dy=0.0):
        return [((px + dx) * k, (py + dy) * k) for px, py in path_points(d)]

    tile = Image.new("L", (n, n), 0)
    ImageDraw.Draw(tile).rounded_rectangle((0, 0, n - 1, n - 1), radius=RADIUS * k, fill=255)
    glyph = Image.new("L", (n, n), 0)
    g = ImageDraw.Draw(glyph)
    g.polygon(poly(BODY, *SHIFT), fill=255)
    g.polygon(poly(COUNTER, *SHIFT), fill=0)
    g.polygon(poly(LEG, *SHIFT), fill=255)

    a = np.asarray(tile, dtype=np.float32) / 255.0
    w = np.asarray(glyph, dtype=np.float32) / 255.0
    # linearGradient x1=0 y1=0 x2=1 y2=1 on a square: t = (x + y) / 2 in unit coordinates
    c = (np.arange(n, dtype=np.float32) + 0.5) / n
    t = (c[None, :] + c[:, None]) / 2.0
    out = np.empty((n, n, 4), dtype=np.float32)
    for ch in range(3):
        grad = TOP_LEFT[ch] + (BOTTOM_RIGHT[ch] - TOP_LEFT[ch]) * t
        out[..., ch] = (grad * (1.0 - w) + WHITE[ch] * w) * a      # premultiplied
    out[..., 3] = a
    out = out.reshape(size, SS, size, SS, 4).mean(axis=(1, 3))
    alpha = out[..., 3:4]
    rgb = np.where(alpha > 0, out[..., :3] / np.maximum(alpha, 1e-6), 0.0)
    px = np.concatenate([rgb, alpha * 255.0], axis=2)
    return Image.fromarray(np.clip(np.rint(px), 0, 255).astype(np.uint8), "RGBA")


def png_bytes(size):
    import io
    buf = io.BytesIO()
    render(size).save(buf, "PNG", optimize=True)
    return buf.getvalue()


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


def hexc(col):
    return "#%02X%02X%02X" % col


def svg():
    """The source of truth; also what the web app shows in the sidebar and on the welcome page."""
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100" role="img">
  <title>Rulith Inference</title>
  <defs>
    <linearGradient id="rulith-icon-bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="{tl}"/>
      <stop offset="1" stop-color="{br}"/>
    </linearGradient>
  </defs>
  <rect width="100" height="100" rx="{r:g}" fill="url(#rulith-icon-bg)"/>
  <g transform="translate({dx:g} {dy:g})" fill="#FFFFFF">
    <path fill-rule="evenodd" d="{body}{counter}"/>
    <path d="{leg}"/>
  </g>
</svg>
""".format(tl=hexc(TOP_LEFT), br=hexc(BOTTOM_RIGHT), r=RADIUS, dx=SHIFT[0], dy=SHIFT[1],
           body=BODY, counter=COUNTER, leg=LEG)


# what Tauri's bundle.icon list and the Windows Store tiles expect
NAMED = {32: "32x32.png", 64: "64x64.png", 128: "128x128.png", 256: "128x128@2x.png", 512: "icon.png"}
TILES = {30: "Square30x30Logo.png", 44: "Square44x44Logo.png", 71: "Square71x71Logo.png",
         89: "Square89x89Logo.png", 107: "Square107x107Logo.png", 142: "Square142x142Logo.png",
         150: "Square150x150Logo.png", 284: "Square284x284Logo.png", 310: "Square310x310Logo.png",
         50: "StoreLogo.png"}
ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)
ICNS_SIZES = (16, 32, 64, 128, 256, 512)


def main():
    out = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "icons")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "rulith-icon.svg"), "w", encoding="utf-8", newline="\n") as f:
        f.write(svg())
    cache = {}

    def get(size):
        if size not in cache:
            cache[size] = png_bytes(size)
        return cache[size]

    for size, name in list(NAMED.items()) + list(TILES.items()):
        with open(os.path.join(out, name), "wb") as f:
            f.write(get(size))
    with open(os.path.join(out, "icon.ico"), "wb") as f:
        f.write(ico([(s, get(s)) for s in ICO_SIZES]))
    with open(os.path.join(out, "icon.icns"), "wb") as f:
        f.write(icns([(s, get(s)) for s in ICNS_SIZES]))
    print("wrote %d files to %s" % (len(NAMED) + len(TILES) + 3, out))


if __name__ == "__main__":
    main()
