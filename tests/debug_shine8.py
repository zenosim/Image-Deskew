"""Debug why glare_mask is empty in integrated run (spot attenuation worked earlier?)."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image
from deskew_pipeline.shine_remover import ShineRemover

g = Image.open("tests/fixtures/glare_55.png").convert("RGB")
sr = ShineRemover()

captured = {}
orig_cc = cv2.connectedComponentsWithStats


def spy(img, connectivity=8):
    r = orig_cc(img, connectivity)
    if img.dtype == np.uint8 and img.max() <= 1:
        captured["cc"] = img.copy()
    return r


# spy ALL calls to find which mask reaches the compaction stage
orig_inpaint = cv2.inpaint
calls = {"inpaint": 0}


def spy_inpaint(src, mask, r, f):
    calls["inpaint"] += 1
    calls["mask_px"] = int(np.count_nonzero(mask))
    return orig_inpaint(src, mask, r, f)


cv2.connectedComponentsWithStats = spy
cv2.inpaint = spy_inpaint
res = sr.remove_shine(g, strength=75)
cv2.connectedComponentsWithStats = orig_cc
cv2.inpaint = orig_inpaint
print("cc calls:", len(captured), "inpaint calls:", calls)
print("final mask px:", int((res.glare_mask > 0).sum()), "detected:", res.shine_detected)
