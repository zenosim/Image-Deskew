"""Oval detection v3: morphological open to bridge text-speckled ovals into solid blobs."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
import glob
from PIL import Image

p = glob.glob("Test_stickes/nekodecal.com_4th-wall*")[0]
img = np.array(Image.open(p).convert("RGB"))
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
ovals = ((hsv[:, :, 1] < 60) & (gray > 150) & (gray < 235) & (lap < 20)).astype(np.uint8)
ovals = cv2.morphologyEx(ovals, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
ovals = cv2.dilate(ovals, np.ones((21, 21), np.uint8))
num, lbl, st, _ = cv2.connectedComponentsWithStats(ovals, connectivity=8)
mask = np.zeros_like(ovals)
kept = 0
for i in range(1, num):
    if st[i, cv2.CC_STAT_AREA] > 5000:
        mask[lbl == i] = 255
        kept += 1
print("blobs:", kept, "mask px:", int((mask > 0).sum()))
vis = img.copy()
vis[mask > 0] = [255, 0, 0]
Image.fromarray(vis).save("/tmp/ghostblob3_vis.png")
np.save("/tmp/ghost_mask3.npy", mask)
print("saved /tmp/ghostblob3_vis.png")
