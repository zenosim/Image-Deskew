"""Debug: which compaction-filter condition drops the glare components."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image

g = np.array(Image.open("tests/fixtures/glare_55.png").convert("RGB"))
hsv = cv2.cvtColor(g, cv2.COLOR_RGB2HSV)
v, s = hsv[:, :, 2], hsv[:, :, 1]
opaque = np.ones(g.shape[:2], bool)
k = 61
mask_f = opaque.astype(np.float32)
denom = cv2.blur(mask_f, (k, k)) + 1e-6
v_local = cv2.GaussianBlur((v.astype(np.float32)) * mask_f, (k, k), 0) / denom
s_local = cv2.GaussianBlur((s.astype(np.float32)) * mask_f, (k, k), 0) / denom
glare = (v > 240) & (s < 22) & (((v.astype(float) - v_local) > 12) | ((s_local - s.astype(float)) > 14)) & (s_local > 28) & opaque
bright = (v > 235) & (s < 30) & opaque
nb, blbl, bstats, _ = cv2.connectedComponentsWithStats(bright.astype(np.uint8), connectivity=8)
contains = np.bincount(blbl[glare], minlength=nb) > 0
area = bstats[:, cv2.CC_STAT_AREA]
max_area = max(40, int(0.10 * opaque.sum()))
ringl = cv2.dilate(blbl.astype(np.float32), np.ones((7, 7), np.uint8)).astype(np.int32)
ring = (ringl > 0) & (blbl == 0) & opaque & (v > 60)
rc = np.bincount(ringl[ring], minlength=nb)
rcc = np.bincount(ringl[ring & (s > 35)], minlength=nb)
cf = rcc / np.maximum(rc, 1)
keep = contains & (area <= max_area) & (rc > 0) & (cf >= 0.7)
print("glare px before keep:", int(glare.sum()), "after:", int((glare & keep[blbl]).sum()))
dropped = contains & ~keep
for i in np.nonzero(dropped)[0][:8]:
    print(f"comp {i}: area={area[i]} maxallowed={max_area} ring_colored={cf[i]:.2f} ring_count={rc[i]}")
