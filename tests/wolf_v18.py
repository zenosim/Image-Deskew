"""v18: dark-than-local ghost text removal (negative residual detection)."""
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
hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
loc_med = cv2.medianBlur(gray.astype(np.uint8), 51).astype(np.float32)
resid = gray - loc_med
ghost = (resid < -6) & (resid > -80) & (hsv[:, :, 1] < 60) & char_mask
region = np.zeros((h, w), np.uint8)
region[590:790, 380:900] = 255
ghost = ghost & (region > 0)
ghost = cv2.morphologyEx(ghost.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
ghost = cv2.dilate(ghost, np.ones((7, 7), np.uint8))
paint = ghost & char_mask
out = arr.copy()
rng = np.random.default_rng(42)
noise = rng.normal(0, 2.0, size=arr.shape)
towel = np.broadcast_to(np.array([250, 250, 252], dtype=np.float32), arr.shape)
out[paint > 0] = np.clip(towel[paint > 0] + noise[paint > 0], 0, 255).astype(np.uint8)
Image.fromarray(out).save("/tmp/wolf_v18.png")
Image.fromarray(out[600:750, 560:780]).resize((330, 225), Image.LANCZOS).save("/tmp/v18_tail.png")
Image.fromarray(out[640:790, 380:640]).resize((390, 225), Image.LANCZOS).save("/tmp/v18_hem.png")
print("painted px:", int((paint > 0).sum()))
