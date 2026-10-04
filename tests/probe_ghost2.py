"""Prototype flat-region ghost detection for white-on-white watermark."""
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
flat = np.abs(loc_med.astype(np.float32) - cv2.blur(loc_med.astype(np.float32), (15, 15))) < 12
ghost = (np.abs(resid) > 6) & (np.abs(resid) < 45) & (sat < 60) & flat
print("ghost candidates:", int(ghost.sum()), "of", h * w)
vis = arr.copy()
vis[ghost] = [255, 0, 0]
Image.fromarray(vis).save("/tmp/ghost_vis.png")
print("saved /tmp/ghost_vis.png")
