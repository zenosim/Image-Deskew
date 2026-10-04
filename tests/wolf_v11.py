"""v11 fixed: paint ovals white with per-pixel noise (broadcast-safe)."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
import glob
from PIL import Image

p = glob.glob("Test_stickes/nekodecal.com_4th-wall*")[0]
img = np.array(Image.open(p).convert("RGB"))
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
ovals = ((hsv[:, :, 1] < 60) & (gray > 150) & (gray < 235) & (lap < 20)).astype(np.uint8)
ovals = cv2.morphologyEx(ovals, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
ovals = cv2.dilate(ovals, np.ones((21, 21), np.uint8))
out = img.copy()
rng = np.random.default_rng(42)
noise = rng.normal(0, 2.5, size=img.shape)
towel = np.broadcast_to(np.array([250, 250, 252], dtype=np.float32), img.shape)
out[ovals > 0] = np.clip(towel[ovals > 0] + noise[ovals > 0], 0, 255).astype(np.uint8)
Image.fromarray(out).save("/tmp/wolf_v11.png")
print("saved, px:", int((ovals > 0).sum()))
crop = out[440:660, 380:740]
Image.fromarray(crop).resize((540, 330), Image.LANCZOS).save("/tmp/v11_chest.png")
print("saved crop")
