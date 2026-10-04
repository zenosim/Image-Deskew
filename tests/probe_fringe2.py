"""Trace deep opaque fringe px beyond the current ring reach."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image

img = Image.open("/tmp/wolf_auto.png").convert("RGBA")
arr = np.array(img)
alpha = arr[:, :, 3]
band = (alpha > 40) & (alpha < 250)
edge_ring = cv2.dilate(band.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
band2 = band | (edge_ring & (alpha >= 250))
hsv = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2HSV)
sat = hsv[:, :, 1]
far_ring = cv2.dilate(band.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
deep = far_ring & (alpha >= 250)
hued = np.abs(hsv[:, :, 0].astype(np.float32) - 13)
hued = np.minimum(hued, 180 - hued)
deep_fringe = deep & (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 110) & (sat > 50)
print("deep opaque green fringe px:", int(deep_fringe.sum()))
# also: how many green fringe px exist at alpha>=250 ANYWHERE (not just near band)?
anywhere = (alpha >= 250) & (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 110) & (sat > 50)
print("green opaque px anywhere:", int(anywhere.sum()))
