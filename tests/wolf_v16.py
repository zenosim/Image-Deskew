"""v16: tightened paint regions — ovals moved down, skin-protected, char-masked."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
import glob
from PIL import Image
from rembg import remove, new_session

p = glob.glob("Test_stickes/nekodecal.com_4th-wall*")[0]
orig = Image.open(p).convert("RGB")
arr = np.array(orig)
h, w = arr.shape[:2]
s = new_session("isnet-anime")
char = np.array(remove(Image.fromarray(arr), session=s))
char_mask = char[:, :, 3] > 128
paint = np.zeros((h, w), np.uint8)
cv2.ellipse(paint, (555, 545), (75, 55), 0, 0, 360, 255, -1)
cv2.ellipse(paint, (655, 535), (70, 50), 0, 0, 360, 255, -1)
paint[555:640, 100:260] = 255
paint[645:770, 430:620] = 255
hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
skin = ((hsv[:, :, 0] <= 25) & (hsv[:, :, 1] > 45)).astype(np.uint8)
skin = cv2.dilate(skin, np.ones((7, 7), np.uint8)) > 0
paint = cv2.dilate(paint, np.ones((9, 9), np.uint8))
paint = paint & char_mask & ~skin
out = arr.copy()
rng = np.random.default_rng(42)
noise = rng.normal(0, 2.5, size=arr.shape)
towel = np.broadcast_to(np.array([250, 250, 252], dtype=np.float32), arr.shape)
out[paint > 0] = np.clip(towel[paint > 0] + noise[paint > 0], 0, 255).astype(np.uint8)
Image.fromarray(out).save("/tmp/wolf_v16.png")
crop = out[430:800, 60:900]
Image.fromarray(crop).save("/tmp/v16_body.png")
print("saved v16, painted:", int((paint > 0).sum()))
