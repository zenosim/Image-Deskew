"""Measure glare-band MAE before/after removal with the ring fix."""
import sys
sys.path.insert(0, ".")
import numpy as np
from PIL import Image

clean = Image.open("Test_stickes/diesentai.com_ada-kisscut_01_ASDSADASASD.png").convert("RGB")
glare = Image.open("tests/fixtures/glare_55.png").convert("RGB")
clean.thumbnail(glare.size)
out = Image.open("/tmp/shine_removed.png")
h, w = glare.size[1], glare.size[0]
c = np.array(clean).astype(np.float32)
o = np.array(out).astype(np.float32)
g = np.array(glare).astype(np.float32)
yy, xx = np.mgrid[0:h, 0:w]
d = np.abs(xx - (w * 0.62 - (yy / h) * w * 0.17))
char_band = (d < 110) & (c.mean(2) <= 240)
before = float(np.abs(g[char_band] - c[char_band]).mean())
after = float(np.abs(o[char_band] - c[char_band]).mean())
print(f"band MAE before: {before:.1f} after: {after:.1f} (recovered {100*(1-after/before):.0f}%)")
