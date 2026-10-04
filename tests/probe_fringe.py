"""Analyze the wolf-girl fringe: hue/sat distribution in the alpha band."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image

img = np.array(Image.open("/tmp/wolf_auto.png").convert("RGBA"))
a = img[:, :, 3]
band = (a > 10) & (a < 250)
print("band px:", int(band.sum()))
hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2HSV)
ys, xs = np.nonzero(band)
top = ys < img.shape[0] * 0.35
print("top-band hue median:", float(np.median(hsv[:, :, 0][ys[top], xs[top]])),
      "sat median:", float(np.median(hsv[:, :, 1][ys[top], xs[top]])))
# core art hue near hair (opaque, top):
core = (a >= 250) & (np.arange(img.shape[0])[:, None] < img.shape[0] * 0.35)
print("top-core hue median:", float(np.median(hsv[:, :, 0][core])),
      "sat median:", float(np.median(hsv[:, :, 1][core])))
