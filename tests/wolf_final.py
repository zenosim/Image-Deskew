"""Final watermark-tile removal: grid blocks + chest ovals + tile rows, face protected."""
import sys
import time
sys.path.insert(0, ".")
import numpy as np
import cv2
import glob
from PIL import Image
from deskew_pipeline.inpainter import BigLamaInpainter
from rembg import remove, new_session

p = glob.glob("Test_stickes/nekodecal.com_4th-wall*")[0]
orig = Image.open(p).convert("RGB")
orig.thumbnail((1400, 1400))
arr = np.array(orig)
h, w = arr.shape[:2]
s = new_session("isnet-anime")
char = np.array(remove(arr, session=s))
char_mask = char[:, :, 3] > 40

grid = np.zeros((h, w), np.uint8)
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
face_zone[450:, :] = False
mask = (grid > 0) & char_mask & ~face_zone
mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) * 255
print("final mask px:", int((mask > 0).sum()))
inp = BigLamaInpainter(model_type="anime")
t0 = time.time()
out = inp.inpaint(Image.fromarray(arr), Image.fromarray(mask), dilate_px=0)
print(f"inpainted {time.time() - t0:.1f}s")
out.save("/tmp/wolf_final.png")
print("saved /tmp/wolf_final.png")
