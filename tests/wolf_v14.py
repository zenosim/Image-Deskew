"""v14: hem ghost text cleanup (targeted region)."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image

img = np.array(Image.open("/tmp/wolf_v13.png").convert("RGB"))
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
region = np.zeros(img.shape[:2], np.uint8)
region[640:770, 430:620] = 255
text = (gray < 240) & (hsv[:, :, 1] < 55) & (region > 0)
text = cv2.dilate(text.astype(np.uint8), np.ones((5, 5), np.uint8))
out = img.copy()
rng = np.random.default_rng(11)
noise = rng.normal(0, 2.0, size=img.shape)
towel = np.broadcast_to(np.array([250, 250, 252], dtype=np.float32), img.shape)
out[text > 0] = np.clip(towel[text > 0] + noise[text > 0], 0, 255).astype(np.uint8)
Image.fromarray(out).save("/tmp/wolf_v14.png")
crop = out[640:770, 430:620]
Image.fromarray(crop).resize((380, 260), Image.LANCZOS).save("/tmp/v14_hem.png")
print("saved v14 + crop")
