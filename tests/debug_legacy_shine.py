"""Debug legacy shine test on 120x200 synthetic."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image, ImageDraw

img = Image.new("RGB", (120, 200), color=(245, 180, 170))
d = ImageDraw.Draw(img)
d.rectangle([40, 80, 80, 120], fill=(255, 255, 255))
arr = np.array(img)
hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
v, s = hsv[:, :, 2].astype(np.float32), hsv[:, :, 1].astype(np.float32)
h2, w2 = v.shape
mf = np.ones((h2, w2), np.float32)
den = cv2.blur(mf, (61, 61)) + 1e-6
v_loc = cv2.GaussianBlur(v * mf, (61, 61), 0) / den
s_loc = cv2.GaussianBlur(s * mf, (61, 61), 0) / den
core = (v > 240) & (s < 22) & (((v - v_loc) > 12) | ((s_loc - s) > 14)) & ((s_loc > 28) | ((v - v_loc) > 40))
print("core px:", int(core.sum()),
      "s_loc at patch:", round(float(s_loc[100, 60]), 1),
      "v-v_loc:", round(float(v[100, 60] - v_loc[100, 60]), 1))
