"""Replicate compaction with the binary-dilate ring fix."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image, ImageDraw

img = Image.new("RGB", (120, 200), color=(245, 180, 170))
d = ImageDraw.Draw(img)
d.rectangle([40, 80, 80, 120], fill=(255, 255, 255))
d.line([(30, 10), (30, 190)], fill=(30, 20, 20), width=2)
rgb = np.array(img)
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
v2, s2 = hsv[:, :, 2], hsv[:, :, 1]
h2, w2 = v2.shape
mf = np.ones((h2, w2), np.float32)
kb = max(31, int(min(h2, w2) * 0.20) | 1)
den = cv2.GaussianBlur(mf, (kb, kb), 0) + 1e-6
v_loc = cv2.GaussianBlur(v2.astype(np.float32) * mf, (kb, kb), 0) / den
s_loc = cv2.GaussianBlur(s2.astype(np.float32) * mf, (kb, kb), 0) / den
core = (v2 > 240) & (s2 < 22) & (((v2.astype(float) - v_loc) > 12) | ((s_loc - s2.astype(float)) > 14)) & ((s_loc > 28) | ((v2.astype(float) - v_loc) > 40))
border = np.vstack([
    rgb[:15].reshape(-1, 3),
    rgb[-15:].reshape(-1, 3),
    rgb[:, :15].reshape(-1, 3),
    rgb[:, -15:].reshape(-1, 3),
])
bg = np.median(border, axis=0)
is_c = np.linalg.norm(rgb.astype(float) - bg, axis=2) < 30
bright = (v2 > 235) & (s2 < 30) & (~is_c)
nb, blbl, bstats, _ = cv2.connectedComponentsWithStats(bright.astype(np.uint8), connectivity=8)
contains = np.bincount(blbl[core], minlength=nb) > 0
area = bstats[:, cv2.CC_STAT_AREA]
maxa = max(40, int(0.10 * h2 * w2))
ring_lbl = cv2.dilate(bright.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
ring = ring_lbl & (blbl == 0) & (v2 > 60) & (~is_c)
rc = np.bincount(ring_lbl[ring], minlength=nb)
rcc = np.bincount(ring_lbl[ring & (s2 > 35)], minlength=nb)
cf = rcc / np.maximum(rc, 1)
keep = contains & (area <= maxa) & (rc > 0) & (cf >= 0.7)
final = core & (keep[blbl] | (blbl == 0))
print("core", int(core.sum()), "after keep", int(final.sum()))
for i in np.nonzero(contains)[0]:
    print(f"comp {i}: area={area[i]} cf={cf[i]:.2f} rc={rc[i]}")
