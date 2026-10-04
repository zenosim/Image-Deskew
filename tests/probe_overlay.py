"""Overlay-text-on-art detector probe: local saturation drop + blackhat response."""
import numpy as np
import cv2
from PIL import Image

img = Image.open("tests/fixtures/watermarked_a52_tiles.png").convert("RGB")
np_rgb = np.array(img)
hsv = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2HSV)
sat = hsv[:, :, 1].astype(np.float32)

sat_localmean = cv2.blur(sat, (25, 25))
sat_drop = sat_localmean - sat  # gray text patch sits below local mean saturation

gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)[:, :, 0]
k21 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k21)

overlay_text = (sat_drop > 18) & (bh > 30)
print("overlay-text px on art:", int(overlay_text.sum()))

vis = np_rgb.copy()
vis[overlay_text] = [255, 0, 0]
Image.fromarray(vis).save("/tmp/overlay_vis.png")
print("saved /tmp/overlay_vis.png")
