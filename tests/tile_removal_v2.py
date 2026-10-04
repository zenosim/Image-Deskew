"""Tile-locked watermark removal v2: dual-threshold (bg strict, char relaxed) + skin guard."""
import sys
import time
sys.path.insert(0, ".")
import numpy as np
import cv2
import glob
from PIL import Image
from deskew_pipeline.inpainter import BigLamaInpainter

p = glob.glob("Test_stickes/nekodecal.com_4th-wall*")[0]
orig = Image.open(p).convert("RGB")
orig.thumbnail((1400, 1400))
arr = np.array(orig)
gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
tmpl = gray[758:812, 630:770]
res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)

hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
border = np.vstack([
    arr[:15].reshape(-1, 3),
    arr[-15:].reshape(-1, 3),
    arr[:, :15].reshape(-1, 3),
    arr[:, -15:].reshape(-1, 3),
])
bg_med = np.median(border, axis=0)
dist_bg = np.linalg.norm(arr.astype(float) - bg_med, axis=2)
is_pinkbg = (dist_bg < 60) & (hsv[:, :, 1] > 40)

mask = np.zeros(gray.shape, np.uint8)
mask[: res.shape[0], : res.shape[1]][res > 0.55] = 255
# char hits: lower threshold, restricted to on_char (res is smaller than gray by the tile size)
hits_char = (res > 0.42) & is_pinkbg[: res.shape[0], : res.shape[1]]
mask[: res.shape[0], : res.shape[1]][hits_char] = 255

ys, xs = np.nonzero(mask)
full = np.zeros_like(mask)
th, tw = tmpl.shape
for y, x in zip(ys, xs):
    full[y : y + th, x : x + tw] = 255
mask = cv2.dilate(full, np.ones((5, 5), np.uint8))

h_, w_ = mask.shape
raw_skin = ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] > 20) & (hsv[:, :, 2] > 70)
ns, ls, ss, _ = cv2.connectedComponentsWithStats(raw_skin.astype(np.uint8))
skin = np.zeros((h_, w_), bool)
for i in range(1, ns):
    if ss[i, cv2.CC_STAT_AREA] >= 80:
        skin[ls == i] = True
fz = cv2.dilate(skin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) > 0
mask[fz] = 0
print("combined mask px:", int((mask > 0).sum()))
vis = arr.copy()
vis[mask > 0] = [255, 0, 0]
Image.fromarray(vis).save("/tmp/tmpl_v2_vis.png")
inp = BigLamaInpainter(model_type="anime")
t0 = time.time()
out = inp.inpaint(Image.fromarray(arr), Image.fromarray(mask), dilate_px=0)
print(f"inpainted {time.time() - t0:.1f}s")
out.save("/tmp/wolf_tile_v2.png")
