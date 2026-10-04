"""Probe extended glare detection: catch partial glare (v 200-240) too."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image

g = np.array(Image.open("tests/fixtures/glare_55.png").convert("RGB"))
hsv = cv2.cvtColor(g, cv2.COLOR_RGB2HSV)
v = hsv[:, :, 2].astype(np.float32)
s = hsv[:, :, 1].astype(np.float32)
h2, w2 = g.shape[:2]
mask_f = np.ones((h2, w2), np.float32)
denom = cv2.blur(mask_f, (61, 61)) + 1e-6
v_loc = cv2.GaussianBlur(v * mask_f, (61, 61), 0) / denom
s_loc = cv2.GaussianBlur(s * mask_f, (61, 61), 0) / denom
yy, xx = np.mgrid[0:h2, 0:w2]
d = np.abs(xx - (w2 * 0.62 - (yy / h2) * w2 * 0.17))
band = d < 110
cand = (v > 200) & (s < 40) & ((v - v_loc) > 25) & (s_loc > 28)
print("extended cand in band:", int((cand & band).sum()), "of", int(band.sum()))
print("extended cand outside band:", int((cand & ~band).sum()))
