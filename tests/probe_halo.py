"""Measure pink fringe on isnet-anime cutout edges."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image

img = np.array(Image.open("/tmp/bg_isnet-anime.png").convert("RGBA"))
rgb = img[:, :, :3].astype(np.int16)
a = img[:, :, 3]
band = (a > 40) & (a < 250)
hsv = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2HSV)
pink = (hsv[:, :, 0] >= 130) & (hsv[:, :, 0] <= 180) & band
print("pink-ish edge band px:", int(pink.sum()), "of", int(band.sum()))
