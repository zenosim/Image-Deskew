"""Ghost detector v2: soft residual only, excluding hard linework neighborhoods."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
import glob
from PIL import Image

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
grad_mag = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
flat = np.abs(loc_med.astype(np.float32) - cv2.blur(loc_med.astype(np.float32), (15, 15))) < 12
ghost_soft = (np.abs(resid) > 6) & (np.abs(resid) < 45) & (grad_mag < 12) & (sat < 60) & flat
near_hard = cv2.dilate((grad_mag > 25).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
ghost_clean = ghost_soft & ~near_hard
print("soft:", int(ghost_soft.sum()), "clean:", int(ghost_clean.sum()))
vis = arr.copy()
vis[ghost_clean] = [255, 0, 0]
Image.fromarray(vis).save("/tmp/ghost_vis2.png")
print("saved")
