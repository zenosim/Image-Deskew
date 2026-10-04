"""Debug char_band emptiness after glare regen."""
import sys
sys.path.insert(0, ".")
import numpy as np
from PIL import Image

clean = Image.open("Test_stickes/diesentai.com_ada-kisscut_01_ASDSADASASD.png").convert("RGB")
glare = Image.open("tests/fixtures/glare_55.png").convert("RGB")
clean.thumbnail(glare.size)
c = np.array(clean).astype(np.float32)
h, w = glare.size[1], glare.size[0]
print("glare:", glare.size, "clean thumb:", clean.size, "c:", c.shape)
yy, xx = np.mgrid[0:h, 0:w]
center = w * 0.62 - (yy / h) * w * 0.17
d = np.abs(xx - center)
char_band = (d < 110) & (c.mean(2) <= 240)
print("char band px:", int(char_band.sum()), "canvas band px:", int(((d < 110) & (c.mean(2) > 240)).sum()))
