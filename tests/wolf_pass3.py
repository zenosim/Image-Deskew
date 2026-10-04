"""Pass 3b: hem ghost blocks in true 900x900 space."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
import glob
from PIL import Image
from deskew_pipeline.inpainter import BigLamaInpainter
from rembg import remove, new_session

p = glob.glob("Test_stickes/nekodecal.com_4th-wall*")[0]
orig = Image.open(p).convert("RGB")  # 900x900 natively
arr = np.array(orig)
h, w = arr.shape[:2]
cur = Image.open("/tmp/wolf_final2.png").resize((w, h))
s = new_session("isnet-anime")
char = np.array(remove(arr, session=s))
char_mask = char[:, :, 3] > 40
grid = np.zeros((h, w), np.uint8)
for cy, cx in [(730, 495), (730, 655), (615, 620)]:
    y0, x0 = max(0, cy - 32), max(0, cx - 80)
    y1, x1 = min(h, cy + 32), min(w, cx + 80)
    grid[y0:y1, x0:x1] = 255
hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
raw_skin = ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] > 20) & (hsv[:, :, 2] > 70)
ns, ls, ss, _ = cv2.connectedComponentsWithStats(raw_skin.astype(np.uint8))
skin = np.zeros((h, w), bool)
for i in range(1, ns):
    if ss[i, cv2.CC_STAT_AREA] >= 300:
        skin[ls == i] = True
fz = cv2.dilate(skin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) > 0
fz[:400, :] = False
mask = (grid > 0) & char_mask & ~fz
mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) * 255
print("pass3b mask px:", int((mask > 0).sum()))
inp = BigLamaInpainter(model_type="anime")
out = inp.inpaint(cur, Image.fromarray(mask), dilate_px=0)
out.save("/tmp/wolf_final3.png")
print("saved /tmp/wolf_final3.png")
