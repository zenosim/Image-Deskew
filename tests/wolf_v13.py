"""v13: towel-left ghost text cleanup."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image

img = np.array(Image.open("/tmp/wolf_v12.png").convert("RGB"))
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
region = np.zeros(img.shape[:2], np.uint8)
region[540:690, 70:260] = 255
text = (gray < 235) & (hsv[:, :, 1] < 50) & (region > 0)
text = cv2.dilate(text.astype(np.uint8), np.ones((5, 5), np.uint8))
out = img.copy()
rng = np.random.default_rng(7)
noise = rng.normal(0, 2.0, size=img.shape)
towel = np.broadcast_to(np.array([250, 250, 252], dtype=np.float32), img.shape)
out[text > 0] = np.clip(towel[text > 0] + noise[text > 0], 0, 255).astype(np.uint8)
Image.fromarray(out).save("/tmp/wolf_v13.png")
crop = out[520:760, 40:400]
Image.fromarray(crop).resize((540, 360), Image.LANCZOS).save("/tmp/v13_towelleft.png")
print("saved v13 + crop")
