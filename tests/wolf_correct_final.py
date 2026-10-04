"""Proper final pass: inpaint the tile/oval ghost regions, THEN segment away the background.

Root cause of the regressed 'final' images: the inpainter returned an RGB image (alpha=255
everywhere), wiping the segmentation alpha. Correct order:
1. inpaint ghost tiles on the RGB (bg included — LaMa needs context)
2. run background segmentation (rembg) on the inpainted image
3. postprocess (defringe)
"""
import os
import sys
import time
sys.path.insert(0, ".")
import numpy as np
import cv2
import glob
from PIL import Image
from deskew_pipeline.inpainter import BigLamaInpainter
from deskew_pipeline.postprocessor import PostProcessor
from rembg import remove, new_session

p = glob.glob("Test_stickes/nekodecal.com_4th-wall*")[0]
orig = Image.open(p).convert("RGB")  # 900x900
arr = np.array(orig)
h, w = arr.shape[:2]

# ---- pass 1: chest ovals + chest/towel-row tiles ----
grid = np.zeros((h, w), np.uint8)
for cy, cx in [(525, 745)]:
    pass
for cy in [525, 745]:
    for cx in [520, 740]:
        y0, x0 = max(0, cy - 35), max(0, cx - 80)
        y1, x1 = min(h, cy + 35), min(w, cx + 80)
        grid[y0:y1, x0:x1] = 255
cv2.ellipse(grid, (545, 555), (85, 75), 0, 0, 360, 255, -1)
cv2.ellipse(grid, (655, 545), (75, 70), 0, 0, 360, 255, -1)

hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
raw_skin = ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] > 20) & (hsv[:, :, 2] > 70)
ns, ls, ss, _ = cv2.connectedComponentsWithStats(raw_skin.astype(np.uint8))
skin = np.zeros((h, w), bool)
for i in range(1, ns):
    if ss[i, cv2.CC_STAT_AREA] >= 500:
        skin[ls == i] = True
face_zone = cv2.dilate(skin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
face_zone[450:, :] = False  # chest skin treated; face above 450 protected

s = new_session("isnet-anime")
char = np.array(remove(arr, session=s))
char_mask = char[:, :, 3] > 40

mask1 = (grid > 0) & char_mask & ~face_zone
mask1 = cv2.dilate(mask1.astype(np.uint8), np.ones((3, 3), np.uint8)) * 255
inp = BigLamaInpainter(model_type="anime")
arr1 = np.array(inp.inpaint(Image.fromarray(arr), Image.fromarray(mask1), dilate_px=0))
print("pass1 done, mask px:", int((mask1 > 0).sum()))

# ---- pass 2: hem/tail rows (block centers in 900-space) ----
grid2 = np.zeros((h, w), np.uint8)
for cy, cx in [(730, 495), (730, 655), (615, 620), (715, 830)]:
    y0, x0 = max(0, cy - 32), max(0, cx - 80)
    y1, x1 = min(h, cy + 32), min(w, cx + 80)
    grid2[y0:y1, x0:x1] = 255
mask2 = (grid2 > 0) & char_mask & ~face_zone
mask2 = cv2.dilate(mask2.astype(np.uint8), np.ones((3, 3), np.uint8)) * 255
arr2 = np.array(inp.inpaint(Image.fromarray(arr1), Image.fromarray(mask2), dilate_px=0))
print("pass2 done, mask px:", int((mask2 > 0).sum()))

# ---- pass 3: background removal on the CLEANED rgb, then defringe ----
clean_cut = remove(Image.fromarray(arr2), session=s)
pp = PostProcessor(defringe=True)
res = pp.process(clean_cut.convert("RGBA"))
final = res.rgba

os.makedirs("/tmp/wolf_correct", exist_ok=True)
final.save("/tmp/wolf_correct.png")
a = np.array(final)[:, :, 3]
print("final alpha: 0:", int((a == 0).sum()), "opaque:", int((a >= 250).sum()))
print("saved /tmp/wolf_correct.png")
