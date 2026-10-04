"""Debug compaction with relaxed gate — why keep kills everything."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image

g = np.array(Image.open("tests/fixtures/glare_55.png").convert("RGB"))
hsv = cv2.cvtColor(g, cv2.COLOR_RGB2HSV)
v_chan, s_chan = hsv[:, :, 2], hsv[:, :, 1]
opaque_mask = np.ones(g.shape[:2], bool)
k = 61
mask_f = opaque_mask.astype(np.float32)
denom = cv2.blur(mask_f, (k, k)) + 1e-6
v_local = cv2.GaussianBlur((v_chan.astype(np.float32)) * mask_f, (k, k), 0) / denom
s_local = cv2.GaussianBlur((s_chan.astype(np.float32)) * mask_f, (k, k), 0) / denom
glare = (
    (v_chan > 240)
    & (s_chan < 22)
    & (((v_chan.astype(float) - v_local) > 12) | ((s_local - s_chan.astype(float)) > 14))
    & ((s_local > 28) | ((v_chan.astype(float) - v_local) > 40))
    & opaque_mask
)
print("core:", int(glare.sum()))
border = np.vstack([
    g[:15].reshape(-1, 3),
    g[-15:].reshape(-1, 3),
    g[:, :15].reshape(-1, 3),
    g[:, -15:].reshape(-1, 3),
])
bg = np.median(border, axis=0)
is_canvas = np.linalg.norm(g.astype(float) - bg, axis=2) < 30
bright = (v_chan > 235) & (s_chan < 30) & opaque_mask & (~is_canvas)
nb, blbl, bstats, _ = cv2.connectedComponentsWithStats(bright.astype(np.uint8), connectivity=8)
contains = np.bincount(blbl[glare], minlength=nb) > 0
area = bstats[:, cv2.CC_STAT_AREA]
maxarea = max(40, int(0.10 * opaque_mask.sum()))
ringl = cv2.dilate(blbl.astype(np.float32), np.ones((7, 7), np.uint8)).astype(np.int32)
ring = (ringl > 0) & (blbl == 0) & opaque_mask & (v_chan > 60) & (~is_canvas)
rc = np.bincount(ringl[ring], minlength=nb)
rcc = np.bincount(ringl[ring & (s_chan > 35)], minlength=nb)
cf = rcc / np.maximum(rc, 1)
keep = contains & (area <= maxarea) & (rc > 0) & (cf >= 0.7)
print("after keep:", int((glare & keep[blbl]).sum()))
dropped = np.nonzero(contains & ~keep)[0]
for i in dropped[:6]:
    print(f"comp {i}: area={area[i]} max={maxarea} cf={cf[i]:.2f} rc={rc[i]}")
