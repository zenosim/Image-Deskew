"""Debug the 2 failing watermark tests."""
import os
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from deskew_pipeline.watermark_remover import WatermarkRemover

wm = WatermarkRemover()
img = Image.open("tests/fixtures/watermarked_a52_tiles.png").convert("RGB")
arr = np.array(img)
r = wm.remove(img, sensitivity=60)

hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
skin = ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] > 20) & (hsv[:, :, 2] > 70)
ys, xs = np.nonzero(skin)
y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
inside = int((r.watermark_mask[y0:y1, x0:x1] > 0).sum())
print("skin bbox:", x0, x1, y0, y1, "mask px inside:", inside)

vis = np.array(img).copy()
vis[r.watermark_mask > 0] = [255, 0, 0]
Image.fromarray(vis[y0:y1, x0:x1]).save("/tmp/face_mask_vis.png")

rgb = arr[0:300, 0:500].astype(np.int16)
sat = rgb.max(2) - rgb.min(2)
gray = rgb.mean(2)
before = float((((np.abs(gray - 140) < 45) & (sat < 26))).mean() * 100)
out = np.array(r.cleaned_image)
rgb2 = out[0:300, 0:500].astype(np.int16)
sat2 = rgb2.max(2) - rgb2.min(2)
gray2 = rgb2.mean(2)
after = float((((np.abs(gray2 - 140) < 45) & (sat2 < 26))).mean() * 100)
print("canvas residual before/after:", round(before, 3), round(after, 3))
