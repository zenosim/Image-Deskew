"""v20: feathered paint (no hard rectangle edges) — ovals + hem ghost, then segment."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
import glob
from PIL import Image
from rembg import remove, new_session
from deskew_pipeline.postprocessor import PostProcessor

p = glob.glob("Test_stickes/nekodecal.com_4th-wall*")[0]
orig = Image.open(p).convert("RGB")
arr = np.array(orig)
h, w = arr.shape[:2]
s = new_session("isnet-anime")
char = np.array(remove(Image.fromarray(arr), session=s))
char_mask = char[:, :, 3] > 128

paint1 = np.zeros((h, w), np.uint8)
cv2.ellipse(paint1, (555, 545), (75, 55), 0, 0, 360, 255, -1)
cv2.ellipse(paint1, (655, 535), (70, 50), 0, 0, 360, 255, -1)
paint1[555:640, 100:260] = 255
paint1[645:770, 430:620] = 255
hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
skin = ((hsv[:, :, 0] <= 25) & (hsv[:, :, 1] > 45)).astype(np.uint8)
skin = cv2.dilate(skin, np.ones((7, 7), np.uint8)) > 0
paint1 = cv2.dilate(paint1, np.ones((9, 9), np.uint8)) & char_mask & ~skin
p1f = cv2.GaussianBlur(paint1 * 255, (31, 31), 0).astype(np.float32) / 255

rng = np.random.default_rng(42)
noise = rng.normal(0, 2.5, size=arr.shape)
towel = np.broadcast_to(np.array([250, 250, 252], dtype=np.float32), arr.shape)
painted = np.clip(towel + noise, 0, 255).astype(np.float32)

step1 = arr.astype(np.float32) * (1 - p1f[..., None]) + painted * p1f[..., None]

gray = cv2.cvtColor(step1.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
loc_med = cv2.medianBlur(gray.astype(np.uint8), 51).astype(np.float32)
resid = gray - loc_med
ghost = ((resid < -6) & (resid > -80) & (hsv[:, :, 1] < 60) & char_mask).astype(np.uint8)
region = np.zeros((h, w), np.uint8)
region[590:790, 380:900] = 255
ghost = ghost & (region > 0)
ghost = cv2.morphologyEx(ghost, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
ghost = cv2.dilate(ghost, np.ones((7, 7), np.uint8))
p2f = cv2.GaussianBlur(ghost * 255, (21, 21), 0).astype(np.float32) / 255
step2 = step1 * (1 - p2f[..., None]) + painted * p2f[..., None]

final_rgb = step2.astype(np.uint8)
cut = remove(Image.fromarray(final_rgb), session=s)
pp = PostProcessor(defringe=True)
res = pp.process(cut.convert("RGBA"))
res.rgba.save("/tmp/wolf_v20.png")
u8 = step2.astype(np.uint8)
Image.fromarray(u8[600:750, 560:780]).resize((330, 225), Image.LANCZOS).save("/tmp/v20_tail.png")
Image.fromarray(u8[640:790, 380:640]).resize((390, 225), Image.LANCZOS).save("/tmp/v20_hem.png")
Image.fromarray(u8[440:640, 300:720]).resize((630, 300), Image.LANCZOS).save("/tmp/v20_chest.png")
print("saved v20 (feathered) + segmentation")
