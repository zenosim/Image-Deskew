"""Why doesn't the generalized defringe fire on hint-mode fringe?"""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image

out = np.array(Image.open("/tmp/hint_extract4.png").convert("RGBA"))
rgb = out[:, :, :3]
hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
a = out[:, :, 3]
band = (a > 40) & (a < 250)
hue = hsv[:, :, 0].astype(np.float32)
sat = hsv[:, :, 1].astype(np.float32)
art_only = (a >= 250).astype(np.float32)
hw = hue * art_only * (sat / 255.0)
hue_med = cv2.blur(hw, (9, 9)) / np.maximum(cv2.blur(art_only * (sat / 255.0), (9, 9)), 1e-3)
sat_med = cv2.blur(sat * art_only, (9, 9)) / np.maximum(cv2.blur(art_only, (9, 9)), 1e-3)
hd = np.abs(hue_med - hue)
hd = np.minimum(hd, 180 - hd)
fix = band & (sat > 30) & (sat_med > 20) & (hd > 25)
print("defringe would fix:", int(fix.sum()), "band px:", int(band.sum()))
ys, xs = np.nonzero(fix)
if len(ys):
    i = 0
    print("sample fix px: hue", hue[ys[i], xs[i]], "sat", sat[ys[i], xs[i]], "hue_med", round(float(hue_med[ys[i], xs[i]]), 1))
else:
    ys2, xs2 = np.nonzero(band)
    if len(ys2):
        i = len(ys2) // 2
        print("sample band px: hue", hue[ys2[i], xs2[i]], "sat", sat[ys2[i], xs2[i]],
              "hue_med", round(float(hue_med[ys2[i], xs2[i]]), 1),
              "sat_med", round(float(sat_med[ys2[i], xs2[i]]), 1))
# what fraction of band is opaque?
print("band alpha==255 frac:", float((a[band] == 255).mean()))
