"""Locate the pink opaque islands in the left strip."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image

out = np.array(Image.open("/tmp/wolf_eroded3.png").convert("RGBA"))
strip = out[:, :60]
m = strip[:, :, 3] > 128
hsv = cv2.cvtColor(strip[:, :, :3], cv2.COLOR_RGB2HSV)
pink = m & (hsv[:, :, 0] >= 160) & (hsv[:, :, 1] > 50)
print("pink opaque px in left strip:", int(pink.sum()))
ys, xs = np.nonzero(pink)
if len(ys):
    print("pink bbox y:", ys.min(), ys.max(), "x:", xs.min(), xs.max())
# full image pink islands:
hsvF = cv2.cvtColor(out[:, :, :3], cv2.COLOR_RGB2HSV)
pinkF = (out[:, :, 3] > 128) & (hsvF[:, :, 0] >= 160) & (hsvF[:, :, 0] <= 179) & (hsvF[:, :, 1] > 50)
print("pink opaque px full image:", int(pinkF.sum()))
