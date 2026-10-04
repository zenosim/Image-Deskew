"""Verify the 13px ring defringe kills the wolf-girl green fringe."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image
from deskew_pipeline.postprocessor import PostProcessor

pp = PostProcessor(defringe=True)
img = Image.open("/tmp/wolf_auto.png").convert("RGBA")
res = pp.process(img)
out = np.array(res.rgba)
a = out[:, :, 3]
band = (a > 10) & (a < 250)
hsv = cv2.cvtColor(out[:, :, :3], cv2.COLOR_RGB2HSV)
ys, xs = np.nonzero(band)
top = ys < out.shape[0] * 0.35
print("top band hue median after:", float(np.median(hsv[:, :, 0][ys[top], xs[top]])),
      "(was 93; hair is 13)")
far = cv2.dilate((a > 40).astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
deep = far & (a >= 250) & (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 110) & (hsv[:, :, 1] > 50)
print("deep green fringe px after:", int(deep.sum()), "(was 673)")
Image.fromarray(out).save("/tmp/wolf_defringed.png")
print("saved /tmp/wolf_defringed.png")
