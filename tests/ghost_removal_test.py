"""End-to-end ghost-watermark removal: soft-flat detector + skin/face protection + LaMa."""
import sys
import time
sys.path.insert(0, ".")
import numpy as np
import cv2
import glob
from PIL import Image
from deskew_pipeline.inpainter import BigLamaInpainter

p = glob.glob("Test_stickes/nekodecal.com_4th-wall*")[0]
PILim = Image.open(p).convert("RGB")
PILim.thumbnail((1400, 1400))
arr = np.array(PILim)
h, w = arr.shape[:2]
gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
loc_med = cv2.medianBlur(gray, 31)
resid = gray.astype(np.float32) - loc_med.astype(np.float32)
hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
sat = hsv[:, :, 1].astype(np.float32)
lap = cv2.Laplacian(gray, cv2.CV_32F)
flat = np.abs(loc_med.astype(np.float32) - cv2.blur(loc_med.astype(np.float32), (15, 15))) < 12
ghost = (np.abs(resid) > 6) & (np.abs(resid) < 45) & (np.abs(lap) < 12) & (sat < 60) & flat
near_hard = cv2.dilate((np.abs(lap) > 25).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
ghost = ghost & ~near_hard

raw_skin = ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] > 20) & (hsv[:, :, 2] > 70)
ns, ls, ss, _ = cv2.connectedComponentsWithStats(raw_skin.astype(np.uint8))
skin = np.zeros((h, w), bool)
for i in range(1, ns):
    if ss[i, cv2.CC_STAT_AREA] >= 80:
        skin[ls == i] = True
fz = cv2.dilate(skin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
ghost = ghost & ~fz

num, lbl, st, _ = cv2.connectedComponentsWithStats(ghost.astype(np.uint8), connectivity=8)
clean = np.zeros_like(ghost)
for i in range(1, num):
    if st[i, cv2.CC_STAT_AREA] >= 25:
        clean[lbl == i] = True
mask = cv2.dilate(clean.astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
print("final ghost mask px:", int((mask > 0).sum()))

inp = BigLamaInpainter(model_type="anime")
t0 = time.time()
out = inp.inpaint(Image.fromarray(arr), Image.fromarray(mask), dilate_px=0)
print(f"inpainted {time.time() - t0:.1f}s")
out.save("/tmp/waifu_ghost_removed.png")
print("saved /tmp/waifu_ghost_removed.png")
