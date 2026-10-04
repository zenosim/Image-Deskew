"""Deeper: why does the integrated remover miss the white patch?"""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image, ImageDraw
from deskew_pipeline.shine_remover import ShineRemover

img = Image.new("RGB", (120, 200), color=(245, 180, 170))
d = ImageDraw.Draw(img)
d.rectangle([40, 80, 80, 120], fill=(255, 255, 255))
d.line([(30, 10), (30, 190)], fill=(30, 20, 20), width=2)
np_rgb = np.array(img)
hsv = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2HSV)
v, s = hsv[:, :, 2], hsv[:, :, 1]
print("patch rgb", np_rgb[100, 60], "s:", s[100, 60], "v:", v[100, 60])
print("BGR vs RGB check: remover converts to BGR then HSV — same values for gray")
res = ShineRemover().remove_shine(img, strength=70)
print("detected:", res.shine_detected)
# replicate exact remover math including its own mask_f semantics
bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
hsv2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
v2, s2 = hsv2[:, :, 2].astype(np.float32), hsv2[:, :, 1].astype(np.float32)
h2, w2 = v2.shape
mf = np.ones((h2, w2), np.float32)
den = cv2.blur(mf, (61, 61)) + 1e-6
v_loc = cv2.GaussianBlur(v2 * mf, (61, 61), 0) / den
s_loc = cv2.GaussianBlur(s2 * mf, (61, 61), 0) / den
core = (v2 > 240) & (s2 < 22) & (((v2 - v_loc) > 12) | ((s_loc - s2) > 14)) & ((s_loc > 28) | ((v2 - v_loc) > 40))
print("replicated core:", int(core.sum()))
