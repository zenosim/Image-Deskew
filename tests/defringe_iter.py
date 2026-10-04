"""Iterative defringe passes on the wolf-girl cutout."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image
from deskew_pipeline.postprocessor import PostProcessor

pp = PostProcessor(defringe=True)
img = Image.open("/tmp/wolf_auto.png").convert("RGBA")
arr = np.array(img)
for it in range(3):
    res = pp._defringe_colors(arr[:, :, :3].copy(), arr[:, :, 3])
    arr[:, :, :3] = res
a = arr[:, :, 3]
hsv = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2HSV)
far = cv2.dilate((a > 40).astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
deep = far & (a >= 250) & (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 110) & (hsv[:, :, 1] > 50)
ys, xs = np.nonzero((a > 10) & (a < 250))
top = ys < arr.shape[0] * 0.35
print("after 3 passes: deep green:", int(deep.sum()),
      "top band hue:", float(np.median(hsv[:, :, 0][ys[top], xs[top]])))
Image.fromarray(arr).save("/tmp/wolf_defringe3.png")
print("saved /tmp/wolf_defringe3.png")
