"""Spy on the real glare mask inside remove_shine to count band detections."""
import sys
sys.path.insert(0, ".")
import numpy as np
import cv2
from PIL import Image
import deskew_pipeline.shine_remover as SR

captured = {}
orig_cc = cv2.connectedComponentsWithStats


def spy(img, connectivity=8):
    ok, labels, stats, centroids = orig_cc(img, connectivity)
    if img.dtype == np.uint8 and img.max() <= 1:
        captured["labels"] = labels
        captured["stats"] = stats
        captured["in"] = img.copy()
    return ok, labels, stats, centroids


cv2.connectedComponentsWithStats = spy
sr = SR.ShineRemover()
g = Image.open("tests/fixtures/glare_55.png").convert("RGB")
res = sr.remove_shine(g, strength=75)
cv2.connectedComponentsWithStats = orig_cc

h, w = g.size[1], g.size[0]
yy, xx = np.mgrid[0:h, 0:w]
band = np.abs(xx - (w * 0.62 - (yy / h) * w * 0.17)) < 110
if "in" in captured:
    m = captured["in"] > 0
    print("mask total:", int(m.sum()), "in band:", int((m & band).sum()), "outside:", int((m & ~band).sum()))
changed = int((np.abs(np.array(g).astype(int) - np.array(res.cleaned_image).astype(int)).sum(2) > 20).sum())
print("changed px:", changed)
res.cleaned_image.save("/tmp/shine_removed.png")
