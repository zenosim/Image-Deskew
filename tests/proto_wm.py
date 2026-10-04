"""Prototype: watermark detector based on background residual instead of tophat strokes.

Watermark model: low-saturation text layer blended over flat canvas OR over artwork.
- On canvas: gray text (deviates from canvas color by 20-90 delta, sat < 30)
- Detection = |pixel - canvas_color| in a narrow band + low saturation + not part of art
"""
import numpy as np
import cv2
from PIL import Image

img = Image.open("tests/fixtures/watermarked_a52_tiles.png").convert("RGB")
np_rgb = np.array(img)
h, w, _ = np_rgb.shape

# canvas color from borders
border = np.vstack([
    np_rgb[:10].reshape(-1, 3),
    np_rgb[-10:].reshape(-1, 3),
    np_rgb[:, :10].reshape(-1, 3),
    np_rgb[:, -10:].reshape(-1, 3),
])
bg = np.median(border, axis=0)
delta = np.linalg.norm(np_rgb.astype(float) - bg, axis=2)

hsv = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2HSV)
sat = hsv[:, :, 1]
val = hsv[:, :, 2]

# is canvas-ish: close to canvas color OR watermark-gray
is_grayish = sat < 40
is_bg_or_wm = (delta < 90)  # watermark at delta ~57 (255-198)
# actual background has delta < 20; watermark band is delta 25..90
wm_band = (delta > 22) & (delta < 95) & is_grayish

# protect chroma art
art = (sat >= 40) | (delta > 95)
cand = (wm_band & ~art).astype(np.uint8) * 255
print("band candidates:", int((cand > 0).sum()))

# text-stroke shaping: letters are thin structures — remove large blobs (shadow gradients)
num, lbl, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
clean = np.zeros_like(cand)
for i in range(1, num):
    area = stats[i, cv2.CC_STAT_AREA]
    cw, ch = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    # letter components: compact, not huge
    if 20 <= area <= (w * h) * 0.02 and cw < w * 0.5 and ch < h * 0.25:
        clean[lbl == i] = 255
print("after component filter:", int((clean > 0).sum()))

# check overlap with TRUE watermark: rebuild the ground truth from fixture recipe
# (tiles at 3x4 grid, text at (120,120,120) alpha 52/255 => delta = (255-198)=57)
gt = np.zeros((h, w), np.uint8)
step_x, step_y = w // 3, h // 4
gray_only = (np.abs(np_rgb[:, :, 0] - np_rgb[:, :, 1]) < 12) & (np.abs(np_rgb[:, :, 1] - np_rgb[:, :, 2]) < 12)
gt_band = (delta > 30) & (delta < 85) & gray_only
print("gt-band px:", int((gt_band > 0).sum()))
inter = ((clean > 0) & gt_band).sum()
print("detector∩gt:", int(inter), "precision:", round(inter / max(1, (clean > 0).sum()), 3),
      "recall:", round(inter / max(1, (gt_band > 0).sum()), 3))

vis = np_rgb.copy()
vis[clean > 0] = [255, 0, 0]
Image.fromarray(vis).save("/tmp/wm_proto_vis.png")
