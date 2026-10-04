"""TRUE final: v16 (clean chest ovals) + v18 negative-residual pass for towel/hem/tail ghosts."""
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

# STEP 1 (from v16): paint chest ovals + towel-left + hem-left towel-white, skin-protected
paint = np.zeros((h, w), np.uint8)
cv2.ellipse(paint, (555, 545), (75, 55), 0, 0, 360, 255, -1)
cv2.ellipse(paint, (655, 535), (70, 50), 0, 0, 360, 255, -1)
paint[555:640, 100:260] = 255
paint[645:770, 430:620] = 255
hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
skin = ((hsv[:, :, 0] <= 25) & (hsv[:, :, 1] > 45)).astype(np.uint8)
skin = cv2.dilate(skin, np.ones((7, 7), np.uint8)) > 0
paint1 = cv2.dilate(paint, np.ones((9, 9), np.uint8)) & char_mask & ~skin
step1 = arr.copy()
rng = np.random.default_rng(42)
noise = rng.normal(0, 2.5, size=arr.shape)
towel = np.broadcast_to(np.array([250, 250, 252], dtype=np.float32), arr.shape)
step1[paint1 > 0] = np.clip(towel[paint1 > 0] + noise[paint1 > 0], 0, 255).astype(np.uint8)

# STEP 2 (from v18): negative-residual ghost pass on towel/tail band
gray = cv2.cvtColor(step1, cv2.COLOR_RGB2GRAY).astype(np.float32)
loc_med = cv2.medianBlur(gray.astype(np.uint8), 51).astype(np.float32)
resid = gray - loc_med
ghost = (resid < -6) & (resid > -80) & (hsv[:, :, 1] < 60) & char_mask
region = np.zeros((h, w), np.uint8)
region[590:790, 380:900] = 255
ghost = ghost & (region > 0)
ghost = cv2.morphologyEx(ghost.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
ghost = cv2.dilate(ghost, np.ones((7, 7), np.uint8))
paint2 = ghost & char_mask
step2 = step1.copy()
noise2 = rng.normal(0, 2.0, size=arr.shape)
step2[paint2 > 0] = np.clip(towel[paint2 > 0] + noise2[paint2 > 0], 0, 255).astype(np.uint8)
print("step1 px:", int((paint1 > 0).sum()), "step2 px:", int((paint2 > 0).sum()))

Image.fromarray(step2).save("/tmp/wolf_v19.png")
Image.fromarray(step2[600:750, 560:780]).resize((330, 225), Image.LANCZOS).save("/tmp/v19_tail.png")
Image.fromarray(step2[640:790, 380:640]).resize((390, 225), Image.LANCZOS).save("/tmp/v19_hem.png")
Image.fromarray(step2[440:640, 300:720]).resize((630, 300), Image.LANCZOS).save("/tmp/v19_chest.png")
print("saved v19 + crops")
