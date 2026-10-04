"""Check bright-component sizes in the band after canvas exclusion."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image

g = np.array(Image.open("tests/fixtures/glare_55.png").convert("RGB"))
hsv = cv2.cvtColor(g, cv2.COLOR_RGB2HSV)
v, s = hsv[:, :, 2], hsv[:, :, 1]
border = np.vstack([
    g[:15].reshape(-1, 3),
    g[-15:].reshape(-1, 3),
    g[:, :15].reshape(-1, 3),
    g[:, -15:].reshape(-1, 3),
])
bg = np.median(border, axis=0)
is_c = np.linalg.norm(g.astype(float) - bg, axis=2) < 30
bright = (v > 235) & (s < 30) & (~is_c)
nb, blbl, bstats, _ = cv2.connectedComponentsWithStats(bright.astype(np.uint8), 8)
sizes = bstats[:, 4]
print("largest comp excl canvas:", int(sizes[1:].max()) if nb > 1 else 0)
h, w = g.shape[:2]
yy, xx = np.mgrid[0:h, 0:w]
band = np.abs(xx - (w * 0.62 - (yy / h) * w * 0.17)) < 110
labels_in_band = np.unique(blbl[band & (blbl > 0)])
big = [(int(l), int(sizes[l])) for l in labels_in_band if sizes[l] > 50000]
print("huge comps in band:", big[:5])
print("num band comps:", len(labels_in_band))
