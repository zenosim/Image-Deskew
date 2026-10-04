"""Probe ghost-watermark residual vs local median on the waifu towel."""
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
towel = resid[int(h * 0.45):int(h * 0.62), int(w * 0.55):int(w * 0.8)]
print("towel residual percentiles (50/90/97):",
      np.percentile(towel, [50, 90, 97]).round(1))
print("abs residual > 8 frac:", round(float((np.abs(towel) > 8).mean()), 3))
