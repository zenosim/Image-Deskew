"""Debug empty char_band in shine probe."""
import sys
sys.path.insert(0, ".")
import numpy as np
from PIL import Image

clean = Image.open("Test_stickes/diesentai.com_ada-kisscut_01_ASDSADASASD.png").convert("RGB")
glare = Image.open("tests/fixtures/glare_55.png").convert("RGB")
c = np.array(clean).astype(np.float32)
g = np.array(glare).astype(np.float32)
print("clean:", c.shape, "glare:", g.shape)
h, w = g.shape[:2]
c = c[:h, :w]
yy, xx = np.mgrid[0:h, 0:w]
bd = np.abs((xx + yy * 0.35) - w * 0.35)
print("band<90 total:", int((bd < 90).sum()))
print("canvas band:", int(((bd < 90) & (c.mean(2) > 240)).sum()))
print("char band:", int(((bd < 90) & (c.mean(2) <= 240)).sum()))
print("band rows:", yy[bd < 90].min(), yy[bd < 90].max(), "cols:", xx[bd < 90].min(), xx[bd < 90].max())
