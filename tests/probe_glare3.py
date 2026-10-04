"""Probe: relax the s_loc gate for wide glare bands (band core has unsaturated surroundings)."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image

g = np.array(Image.open("tests/fixtures/glare_55.png").convert("RGB"))
hsv = cv2.cvtColor(g, cv2.COLOR_RGB2HSV)
v = hsv[:, :, 2].astype(np.float32)
s = hsv[:, :, 1].astype(np.float32)
h2, w2 = v.shape
mf = np.ones((h2, w2), np.float32)
den = cv2.blur(mf, (61, 61)) + 1e-6
v_loc = cv2.GaussianBlur(v * mf, (61, 61), 0) / den
s_loc = cv2.GaussianBlur(s * mf, (61, 61), 0) / den
yy, xx = np.mgrid[0:h2, 0:w2]
band = np.abs(xx - (w2 * 0.62 - (yy / h2) * w2 * 0.17)) < 110
old = (v > 240) & (s < 22) & (((v - v_loc) > 12) | ((s_loc - s) > 14)) & (s_loc > 28)
new = (v > 240) & (s < 22) & (((v - v_loc) > 12) | ((s_loc - s) > 14)) & ((s_loc > 28) | ((v - v_loc) > 40))
print("old core in band:", int((old & band).sum()), "new core in band:", int((new & band).sum()))
print("new core outside band:", int((new & ~band).sum()))
