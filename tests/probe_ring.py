"""Measure opaque-ring vs core hue on the wolf cutout."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image

img = np.array(Image.open("/tmp/wolf_auto.png").convert("RGBA"))
a = img[:, :, 3]
hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2HSV)
core = a >= 250
dist_in = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 5)
ring5 = core & (dist_in <= 5)
ring20 = core & (dist_in > 20)
print("ring5 hue median:", float(np.median(hsv[:, :, 0][ring5])),
      "sat", float(np.median(hsv[:, :, 1][ring5])))
print("core20 hue median:", float(np.median(hsv[:, :, 0][ring20])),
      "sat", float(np.median(hsv[:, :, 1][ring20])))
