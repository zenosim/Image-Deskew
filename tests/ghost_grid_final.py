"""Final ghost removal: grid-restricted local-residual mask + LaMa."""
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
h, w = arr.shape[:2]
gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
loc_med = cv2.medianBlur(gray, 31)
resid = gray.astype(np.float32) - loc_med.astype(np.float32)
hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
sat = hsv[:, :, 1].astype(np.float32)
lap = cv2.Laplacian(gray, cv2.CV_32F)
ghost = (np.abs(resid) > 7) & (np.abs(resid) < 80) & (np.abs(lap) < 45) & (sat < 90)
grid = np.zeros((h, w), np.uint8)
for cy in [525, 745]:
    for cx in [520, 740]:
        y0, x0 = max(0, cy - 30), max(0, cx - 75)
        y1, x1 = min(h, cy + 30), min(w, cx + 75)
        grid[y0:y1, x0:x1] = 255
ghost_mask = (ghost & (grid > 0)).astype(np.uint8) * 255
raw_skin = ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] > 20) & (hsv[:, :, 2] > 70)
ns, ls, ss, _ = cv2.connectedComponentsWithStats(raw_skin.astype(np.uint8))
skin = np.zeros((h, w), bool)
for i in range(1, ns):
    if ss[i, cv2.CC_STAT_AREA] >= 80:
        skin[ls == i] = True
fz = cv2.dilate(skin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) > 0
ghost_mask[fz] = 0
ghost_mask = cv2.dilate(ghost_mask, np.ones((3, 3), np.uint8))
print("grid-restricted ghost mask px:", int((ghost_mask > 0).sum()))
inp = BigLamaInpainter(model_type="anime")
t0 = time.time()
out = inp.inpaint(Image.fromarray(arr), Image.fromarray(ghost_mask), dilate_px=0)
print(f"inpainted {time.time() - t0:.1f}s")
out.save("/tmp/wolf_grid_ghost.png")
print("saved /tmp/wolf_grid_ghost.png")
