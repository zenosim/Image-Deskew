"""Mask-shrink approach: erode the opaque mask 3px to cut the bg-colored fringe ring."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image

img = np.array(Image.open("/tmp/wolf_auto.png").convert("RGBA"))
a = img[:, :, 3]
core = (a >= 250).astype(np.uint8)
er = cv2.erode(core, np.ones((7, 7), np.uint8))
feather = cv2.GaussianBlur(er * 255, (5, 5), 0)
new_a = (a.astype(np.float32) * (feather / 255.0)).astype(np.uint8)
new_a[er > 0] = a[er > 0]
out = img.copy()
out[:, :, 3] = new_a
hsv = cv2.cvtColor(out[:, :, :3], cv2.COLOR_RGB2HSV)
core2 = new_a >= 250
dist_in = cv2.distanceTransform(core2.astype(np.uint8), cv2.DIST_L2, 5)
ring5 = core2 & (dist_in <= 5)
print("new ring5 hue median:", round(float(np.median(hsv[:, :, 0][ring5])), 1), "(was 101 green)")
print("area loss:", round(float(1 - core2.sum() / core.sum()) * 100, 1), "%")
Image.fromarray(out).save("/tmp/wolf_eroded.png")
print("saved /tmp/wolf_eroded.png")
