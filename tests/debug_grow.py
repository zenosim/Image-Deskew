"""Verify region-grow candidate logic on the real detected core."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image
from deskew_pipeline.shine_remover import ShineRemover

g_img = Image.open("tests/fixtures/glare_55.png").convert("RGB")
g = np.array(g_img)
hsv = cv2.cvtColor(g, cv2.COLOR_RGB2HSV)
v = hsv[:, :, 2].astype(np.float32)
s = hsv[:, :, 1].astype(np.float32)
h2, w2 = v.shape
mf = np.ones((h2, w2), np.float32)
den = cv2.blur(mf, (61, 61)) + 1e-6
v_loc = cv2.GaussianBlur(v * mf, (61, 61), 0) / den

# get the real detected core
cap = {}
occ = cv2.connectedComponentsWithStats


def spy(img, connectivity=8):
    r = occ(img, connectivity)
    if img.dtype == np.uint8 and img.max() <= 1:
        cap["m"] = img.copy()
    return r


cv2.connectedComponentsWithStats = spy
sr = ShineRemover()
sr.remove_shine(g_img, strength=75)
cv2.connectedComponentsWithStats = occ
core = cap["m"] > 0
print("real core px:", int(core.sum()))

grown = cv2.dilate(core.astype(np.uint8), np.ones((51, 51), np.uint8)) > 0
cand = grown & ((v - v_loc) > 20) & (s < 60)
print("grown candidates:", int(cand.sum()))
band_truth = np.zeros_like(core)
yy, xx = np.mgrid[0:h2, 0:w2]
band_truth[np.abs(xx - (w2 * 0.62 - (yy / h2) * w2 * 0.17)) < 110] = True
print("truth band px:", int(band_truth.sum()), "cand∩truth:", int((cand & band_truth).sum()))
