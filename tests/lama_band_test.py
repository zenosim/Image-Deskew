"""Wide glare band: ShineRemover detects core, LaMa inpaints the dilated band."""
import sys
import time
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image
from deskew_pipeline.shine_remover import ShineRemover
from deskew_pipeline.inpainter import BigLamaInpainter

g = Image.open("tests/fixtures/glare_55.png").convert("RGB")
sr = ShineRemover()

captured = {}
orig_cc = cv2.connectedComponentsWithStats


def spy(img, connectivity=8):
    r = orig_cc(img, connectivity)
    if img.dtype == np.uint8 and img.max() <= 1:
        captured["m"] = img.copy()
    return r


cv2.connectedComponentsWithStats = spy
res = sr.remove_shine(g, strength=75)
cv2.connectedComponentsWithStats = orig_cc

mask = cv2.dilate(captured["m"] * 255, np.ones((31, 31), np.uint8))
print("dilated mask px:", int((mask > 0).sum()))
inp = BigLamaInpainter(model_type="anime")
t0 = time.time()
out = inp.inpaint(g, Image.fromarray(mask), dilate_px=0)
print(f"lama band inpaint: {time.time() - t0:.1f}s")
out.save("/tmp/shine_lama.png")

clean = Image.open("Test_stickes/diesentai.com_ada-kisscut_01_ASDSADASASD.png").convert("RGB")
clean.thumbnail(g.size)
c = np.array(clean).astype(np.float32)
o = np.array(out.convert("RGB")).astype(np.float32)
gg = np.array(g).astype(np.float32)
h2, w2 = g.size[1], g.size[0]
yy, xx = np.mgrid[0:h2, 0:w2]
band = np.abs(xx - (w2 * 0.62 - (yy / h2) * w2 * 0.17)) < 110
b_before = float(np.abs(gg[band] - c[band]).mean())
b_after = float(np.abs(o[band] - c[band]).mean())
print(f"band MAE before: {b_before:.1f} after lama: {b_after:.1f} (recovered {100 * (1 - b_after / b_before):.0f}%)")
