"""Char-restricted gray-oval watermark detection on the wolf-girl."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
import glob
from PIL import Image
from rembg import remove, new_session

p = glob.glob("Test_stickes/nekodecal.com_4th-wall*")[0]
img = np.array(Image.open(p).convert("RGB"))
h, w = img.shape[:2]
s = new_session("isnet-anime")
char = np.array(remove(Image.fromarray(img), session=s))
char_mask = char[:, :, 3] > 128
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
ovals = (hsv[:, :, 1] < 60) & (gray > 150) & (gray < 235) & (lap < 8) & char_mask
num, lbl, st, _ = cv2.connectedComponentsWithStats(ovals.astype(np.uint8), connectivity=8)
mask = np.zeros((h, w), np.uint8)
kept = 0
for i in range(1, num):
    if st[i, cv2.CC_STAT_AREA] > 2000:
        mask[lbl == i] = 255
        kept += 1
mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))
print("blobs kept:", kept, "mask px:", int((mask > 0).sum()))
vis = img.copy()
vis[mask > 0] = [255, 0, 0]
Image.fromarray(vis).save("/tmp/ghostblob2_vis.png")
np.save("/tmp/ghost_mask2.npy", mask)
print("saved /tmp/ghostblob2_vis.png")
