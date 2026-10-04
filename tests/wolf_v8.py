"""v8: final two ghost tiles (towel-top-left 715,170 and towel-right 715,700)."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
import glob
from PIL import Image
from deskew_pipeline.inpainter import BigLamaInpainter
from deskew_pipeline.postprocessor import PostProcessor
from rembg import remove, new_session

p = glob.glob("Test_stickes/nekodecal.com_4th-wall*")[0]
orig = Image.open(p).convert("RGB")
arr = np.array(orig)
h, w = arr.shape[:2]
s = new_session("isnet-anime")
char = np.array(remove(Image.fromarray(arr), session=s))
char_mask = char[:, :, 3] > 40
alpha = char[:, :, 3:4].astype(np.float32) / 255
comp = (char[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
raw_skin = ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] > 20) & (hsv[:, :, 2] > 70)
ns, ls, ss, _ = cv2.connectedComponentsWithStats(raw_skin.astype(np.uint8))
skin = np.zeros((h, w), bool)
for i in range(1, ns):
    if ss[i, cv2.CC_STAT_AREA] >= 500:
        skin[ls == i] = True
face_zone = cv2.dilate(skin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
face_zone[450:, :] = False
grid = np.zeros((h, w), np.uint8)
for cy, cx in [(525, 745), (745, 520), (745, 740), (455, 140), (470, 230), (620, 515), (715, 170), (715, 700)]:
    y0, x0 = max(0, cy - 35), max(0, cx - 80)
    y1, x1 = min(h, cy + 35), min(w, cx + 80)
    grid[y0:y1, x0:x1] = 255
cv2.ellipse(grid, (545, 555), (85, 75), 0, 0, 360, 255, -1)
cv2.ellipse(grid, (655, 545), (75, 70), 0, 0, 360, 255, -1)
for cy, cx in [(730, 495), (730, 655), (615, 620), (715, 830)]:
    y0, x0 = max(0, cy - 32), max(0, cx - 80)
    y1, x1 = min(h, cy + 32), min(w, cx + 80)
    grid[y0:y1, x0:x1] = 255
tiles = cv2.dilate((grid > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) & char_mask & ~face_zone
mask = tiles * 255
inp = BigLamaInpainter(model_type="anime")
filled = inp.inpaint(Image.fromarray(comp), Image.fromarray(mask), dilate_px=0)
out = char.copy()
m = mask > 0
out[m, :3] = np.array(filled)[m, :3]
pp = PostProcessor(defringe=True)
res = pp.process(Image.fromarray(out))
res.rgba.save("/tmp/wolf_v8.png")
alpha2 = out[:, :, 3:4].astype(np.float32) / 255
comp2 = (out[:, :, :3] * alpha2 + 255 * (1 - alpha2)).astype(np.uint8)
Image.fromarray(comp2[500:640, 100:340]).resize((480, 280), Image.LANCZOS).save("/tmp/v8_towelleft.png")
Image.fromarray(comp2[500:640, 500:780]).resize((560, 280), Image.LANCZOS).save("/tmp/v8_towelright.png")
print("saved v8 + crops")
