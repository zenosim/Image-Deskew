"""Shine remover vs synthetic glare fixture: does it recover the underlying art?"""
import os
import sys
import time
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from deskew_pipeline.shine_remover import ShineRemover

FIX = os.path.join(ROOT, "tests", "fixtures")

sr = ShineRemover()
glare = Image.open(os.path.join(FIX, "glare_55.png")).convert("RGB")
clean = Image.open(os.path.join(ROOT, "Test_stickes", "diesentai.com_ada-kisscut_01_ASDSADASASD.png")).convert("RGB")
clean.thumbnail(glare.size)

t0 = time.time()
res = sr.remove_shine(glare, strength=60)
dt = time.time() - t0
out = np.array(res.cleaned_image).astype(np.float32)
g = np.array(glare).astype(np.float32)
c = np.array(clean).astype(np.float32)[: out.shape[0], : out.shape[1]]

# glare band (regen): center x = 0.62w - (y/h)*0.17w, half-width 110
h, w = out.shape[:2]
yy, xx = np.mgrid[0:h, 0:w]
band_dist = np.abs(xx - (w * 0.62 - (yy / h) * w * 0.17))
# clip all to common size (clean was thumbnailed to glare size, may differ by a pixel)
c = c[:h, :w]
canvas_band = (band_dist < 90) & (c.mean(axis=2) > 240)
char_band = (band_dist < 90) & (c.mean(axis=2) <= 240)
err_before = np.abs(g[char_band] - c[char_band]).mean() if char_band.any() else float("nan")
err_after = np.abs(out[char_band] - c[char_band]).mean() if char_band.any() else float("nan")
print(
    f"band px: canvas={int(canvas_band.sum())} char={int(char_band.sum())} | "
    f"glare band, character pixels: MAE before={err_before:.1f} after={err_after:.1f} ({dt:.1f}s)"
)
res.cleaned_image.save("/tmp/shine_removed.png")
glare.save("/tmp/shine_input.png")
