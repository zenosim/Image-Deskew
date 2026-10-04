"""v6: v5 + towel-left-mid ghost tiles (row 455/470, col 140/230)."""
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
for cy, cx in [(525, 745), (745, 520), (745, 740), (455, 140), (470, 230)]:
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
res.rgba.save("/tmp/wolf_v6.png")
print("saved /tmp/wolf_v6.png")
