"""Replicate exact v2 protection chain to debug candidate count."""
import numpy as np
import cv2
from PIL import Image

img = Image.open("tests/fixtures/watermarked_a52_tiles.png").convert("RGB")
np_rgb = np.array(img)
h, w, _ = np_rgb.shape
bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
t_th = max(18, int(54 - 0.6 * 36))
t_bh = max(24, int(64 - 0.6 * 40))
_, m1 = cv2.threshold(th, t_th, 255, cv2.THRESH_BINARY)
_, m2 = cv2.threshold(bh, t_bh, 255, cv2.THRESH_BINARY)
cand = m1 | m2
print("cand:", int((cand > 0).sum()))

hsv = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2HSV)
is_chroma = (hsv[:, :, 1] > 25) & (hsv[:, :, 2] > 30)
border = np.vstack([
    np_rgb[:10].reshape(-1, 3),
    np_rgb[-10:].reshape(-1, 3),
    np_rgb[:, :10].reshape(-1, 3),
    np_rgb[:, -10:].reshape(-1, 3),
])
bg = np.median(border, axis=0)
is_canvas = np.linalg.norm(np_rgb.astype(float) - bg, axis=2) < 22.0
raw_skin = ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] > 20) & (hsv[:, :, 2] > 70) & (~is_canvas)
ns, ls, ss, _ = cv2.connectedComponentsWithStats(raw_skin.astype(np.uint8))
skin = np.zeros((h, w), bool)
for i in range(1, ns):
    if ss[i, cv2.CC_STAT_AREA] >= 80:
        skin[ls == i] = True
fz = cv2.dilate(skin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))) > 0
c = cand.copy()
c[fz] = 0
print("after face:", int((c > 0).sum()))
c[is_chroma] = 0
print("after chroma:", int((c > 0).sum()))

is_dark = (gray < 115).astype(np.uint8)
nd, ld, sd, _ = cv2.connectedComponentsWithStats(is_dark)
lda = np.zeros((h, w), bool)
for i in range(1, nd):
    if sd[i, cv2.CC_STAT_AREA] > 300:
        lda[ld == i] = True
kern = np.ones((31, 31), np.float32)
dd = cv2.filter2D(lda.astype(np.float32), -1, kern, borderType=cv2.BORDER_REPLICATE) / kern.sum()
on_art = dd > 0.70
db = cv2.dilate(lda.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
ldbo = db & ~on_art
c[ldbo] = 0
print("after dark-boundary:", int((c > 0).sum()))
c[on_art & (m2 == 0)] = 0
print("after on-art tophat kill:", int((c > 0).sum()))
print("need >=", max(60, int(w * h * 0.003)))
