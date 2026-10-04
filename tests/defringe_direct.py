"""Direct defringe test on hint-mode output (RGBA numpy path)."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image
from deskew_pipeline.postprocessor import PostProcessor

img = Image.open("/tmp/hint_extract4.png").convert("RGBA")
arr = np.array(img)
pp = PostProcessor(defringe=True)
rgb_fixed = pp._defringe_colors(arr[:, :, :3], arr[:, :, 3])
out = arr.copy()
out[:, :, :3] = rgb_fixed
Image.fromarray(out).save("/tmp/hint_extract5.png")
hsv = cv2.cvtColor(out[:, :, :3], cv2.COLOR_RGB2HSV)
a = out[:, :, 3]
band = (a > 40) & (a < 250)
print("band px after:", int(band.sum()))
