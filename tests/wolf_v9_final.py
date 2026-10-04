"""Final wolf-girl pass: ghost-blob mask + hem/tail grid rows, then segment + defringe."""
import sys
import time
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
gm = np.load("/tmp/ghost_mask3.npy")
s = new_session("isnet-anime")
char = np.array(remove(Image.fromarray(arr), session=s))
char_mask = char[:, :, 3] > 40
hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
raw_skin = ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] > 20) & (hsv[:, :, 2] > 70)
ns, ls, ss, _ = cv2.connectedComponentsWithStats(raw_skin.astype(np.uint8))
skin = np.zeros((h, w), bool)
for i in range(1, ns):
    if ss[i, cv2.CC_STAT_AREA] >= 500:
        skin[ls == i] = True
face_zone = cv2.dilate(skin.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
face_zone[:450, :] = False
hem = np.zeros((h, w), np.uint8)
hem[690:770, 380:880] = 255
hem[600:730, 760:890] = 255
mask = ((gm > 0) | (hem > 0)) & char_mask & ~face_zone
mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) * 255
print("final mask px:", int((mask > 0).sum()))
inp = BigLamaInpainter(model_type="anime")
t0 = time.time()
out = inp.inpaint(Image.fromarray(arr), Image.fromarray(mask), dilate_px=0)
print(f"inpainted {time.time() - t0:.1f}s")
clean_cut = remove(Image.fromarray(np.array(out)), session=s)
pp = PostProcessor(defringe=True)
res = pp.process(clean_cut.convert("RGBA"))
res.rgba.save("/tmp/wolf_v9.png")
print("saved /tmp/wolf_v9.png")
