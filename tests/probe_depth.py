"""Depth-profile of fringe: how deep does the bg-colored margin go inside the mask?"""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image

img = np.array(Image.open("/tmp/wolf_eroded3.png").convert("RGBA"))
a = img[:, :, 3]
hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2HSV)
core = (a >= 250).astype(np.uint8)
dist_in = cv2.distanceTransform(core, cv2.DIST_L2, 5)
for lo, hi in [(0, 3), (3, 8), (8, 15), (15, 30), (30, 60)]:
    ring = core & (dist_in >= lo) & (dist_in < hi)
    hh = hsv[:, :, 0][ring]
    ss = hsv[:, :, 1][ring]
    green = int(((hh >= 90) & (hh <= 110) & (ss > 50)).sum())
    print(f"depth {lo}-{hi}: hue median {float(np.median(hh)):.0f} green px {green}/{int(ring.sum())}")
