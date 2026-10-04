"""Full fix: erode the ENTIRE soft alpha (not just opaque), zeroing everything outside.

Alpha >= 1 pixels outside the eroded core are background remnant; set them to 0. Interior
alpha untouched. Then re-feather the edge with a 3px blur for antialiasing.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image

img = np.array(Image.open("/tmp/wolf_auto.png").convert("RGBA"))
a = img[:, :, 3]
core = (a >= 250).astype(np.uint8)
er = cv2.erode(core, np.ones((7, 7), np.uint8))
# keep only alpha inside the eroded core; everything else 0
new_a = np.where(er > 0, a, 0).astype(np.uint8)
# feather: 3px blur of the new alpha edge
new_a = cv2.GaussianBlur(new_a, (5, 5), 0)
new_a[er > 0] = np.maximum(new_a[er > 0], a[er > 0])  # interior stays solid
out = img.copy()
out[:, :, 3] = new_a
Image.fromarray(out).save("/tmp/wolf_eroded2.png")
hsv = cv2.cvtColor(out[:, :, :3], cv2.COLOR_RGB2HSV)
core2 = new_a >= 250
dist_in = cv2.distanceTransform(core2.astype(np.uint8), cv2.DIST_L2, 5)
ring5 = core2 & (dist_in <= 5)
print("ring5 hue:", round(float(np.median(hsv[:, :, 0][ring5])), 1),
      "area loss:", round(float(1 - core2.sum() / core.sum()) * 100, 1), "%")
print("saved /tmp/wolf_eroded2.png")
