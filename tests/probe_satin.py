"""Measure sat values: chest skin vs towel, to build a paint-restriction gate."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
import glob
from PIL import Image

p = glob.glob("Test_stickes/nekodecal.com_4th-wall*")[0]
img = np.array(Image.open(p).convert("RGB"))
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
print("chest skin sat median:", float(np.median(hsv[400:430, 500:600, 1])))
print("towel sat median:", float(np.median(hsv[560:620, 420:700, 1])))
# also hue
print("chest skin hue median:", float(np.median(hsv[400:430, 500:600, 0])))
print("towel hue median:", float(np.median(hsv[560:620, 420:700, 0])))
