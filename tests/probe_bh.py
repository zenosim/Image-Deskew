"""Check blackhat response for watermark text over dark art (dress area)."""
import numpy as np
import cv2
from PIL import Image

img = Image.open("tests/fixtures/watermarked_a52_tiles.png").convert("RGB")
np_rgb = np.array(img)
gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)[:, :, 0]
for ks in (5, 11, 21):
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    n = int((bh[780:900, 620:900] > 44).sum())
    print(f"kernel {ks}: dress patch bh>44 px = {n} of {120*280}")
