"""Composite chroma mattes over a dark checkerboard so alpha is actually visible."""
import sys
sys.path.insert(0, "/root/workspace/Image-Deskew")
import numpy as np
from PIL import Image, ImageDraw


def checker(size, sq=24, c1=(70, 70, 80), c2=(105, 105, 118)):
    im = Image.new("RGB", size, c1)
    d = ImageDraw.Draw(im)
    for y in range(0, size[1], sq):
        for x in range(0, size[0], sq):
            if (x // sq + y // sq) % 2:
                d.rectangle([x, y, x + sq - 1, y + sq - 1], fill=c2)
    return im


for name in ("green", "magenta"):
    rgba = Image.open(f"/tmp/ck_{name}_rgba.png")
    comp = checker(rgba.size)
    comp.paste(rgba, (0, 0), rgba)
    comp.save(f"/tmp/vis_{name}_checker.png")
    # hair zoom: top area around head/hair strands
    w, h = rgba.size
    crop = rgba.crop((int(w * 0.35), int(h * 0.02), int(w * 0.80), int(h * 0.40)))
    zw, zh = crop.size[0] * 2, crop.size[1] * 2
    zc = checker((zw, zh))
    zc.paste(crop.resize((zw, zh), Image.NEAREST), (0, 0), crop.resize((zw, zh), Image.NEAREST))
    zc.save(f"/tmp/vis_{name}_hairzoom.png")
print("saved checker composites + hair zooms")
