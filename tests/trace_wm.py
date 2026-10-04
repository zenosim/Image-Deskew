"""Trace which protection stage kills watermark candidates."""
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
cand = ((th > 36) | (bh > 44)).astype(np.uint8) * 255
print("cand px:", int((cand > 0).sum()))

hsv = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2HSV)
is_chroma = (hsv[:, :, 1] > 25) & (hsv[:, :, 2] > 30)
c2 = cand.copy()
c2[is_chroma] = 0
print("after chroma kill:", int((c2 > 0).sum()))

border = np.vstack([
    np_rgb[:10].reshape(-1, 3),
    np_rgb[-10:].reshape(-1, 3),
    np_rgb[:, :10].reshape(-1, 3),
    np_rgb[:, -10:].reshape(-1, 3),
])
bg = np.median(border, axis=0)
is_canvas = np.linalg.norm(np_rgb.astype(float) - bg, axis=2) < 22.0
raw_skin = ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] > 20) & (hsv[:, :, 2] > 70) & (~is_canvas)
num_s, lbls_s, stats_s, _ = cv2.connectedComponentsWithStats(raw_skin.astype(np.uint8))
skin = np.zeros((h, w), bool)
for i in range(1, num_s):
    if stats_s[i, cv2.CC_STAT_AREA] >= 80:
        skin[lbls_s == i] = True
fz = cv2.dilate(skin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))) > 0
c3 = c2.copy()
c3[fz] = 0
print("after face-zone kill:", int((c3 > 0).sum()))

is_dark = (gray < 115).astype(np.uint8)
num_d, lbls_d, stats_d, _ = cv2.connectedComponentsWithStats(is_dark)
lda = np.zeros((h, w), bool)
for i in range(1, num_d):
    if stats_d[i, cv2.CC_STAT_AREA] > 300:
        lda[lbls_d == i] = True
# Light text ON TOP of dark art (a white watermark over hair/dress) sits inside large-dark
# regions; the tophat stroke fires on it there. Only kill candidates that merely TOUCH the
# boundary of dark art from outside, not ones fully enclosed by it (enclosed = overlay text).
dark_dilated = cv2.dilate(lda.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
# enclosed = candidate adjacent to dark art on most sides but surrounded (not on outer edge).
# Practical test: candidate pixel whose 15px neighborhood is >=70% dark art = overlay on art.
kern = np.ones((31, 31), np.float32)
dark_density = cv2.filter2D(lda.astype(np.float32), -1, kern, borderType=cv2.BORDER_REPLICATE) / kern.sum()
on_art = dark_density > 0.70
lda_touch = dark_dilated & ~on_art  # outside boundary strokes only
c4 = c3.copy()
c4[lda_touch] = 0
# candidates fully inside dark art are watermark-over-art: keep them but only where the art is
# genuinely dark under the stroke (bh fires on light-on-dark)
keep_on_art = (c3 > 0) & on_art & (bh > 44)
c4[keep_on_art] = 255
print("after large-dark kill:", int((c4 > 0).sum()))

# component filtering
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(c4, connectivity=8)
kept = 0
max_char_dim = max(35, int(min(h, w) * 0.35))
max_comp_area = max(500, int((w * h) * 0.05))
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    cw, ch = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
    if 8 <= area <= max_comp_area and cw < max_char_dim and ch < max_char_dim:
        if max(cw, ch) / max(1, min(cw, ch)) < 15:
            kept += area
print("after component filter:", kept)
print("min_detection_px:", max(60, int((w * h) * 0.003)))
