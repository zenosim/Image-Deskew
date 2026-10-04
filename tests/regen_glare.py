"""Regenerate glare fixture as physically-correct specular glare (blend toward white)."""
import os
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "tests", "fixtures")
os.makedirs(FIX, exist_ok=True)

img = Image.open(os.path.join(ROOT, "Test_stickes", "diesentai.com_ada-kisscut_01_ASDSADASASD.png")).convert("RGB")
img.thumbnail((1400, 1400))
w, h = img.size
arr = np.array(img).astype(np.float32)
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
center = w * 0.62 - (yy / h) * w * 0.17
d = np.abs(xx - center)
a = np.clip(1.0 - d / 110.0, 0, 1) ** 1.3 * 0.85  # specular blend alpha up to 0.85
glare = arr * (1 - a[:, :, None]) + 255 * a[:, :, None]
out = Image.fromarray(np.clip(glare, 0, 255).astype(np.uint8))
out.save(os.path.join(FIX, "glare_55.png"))
print("wrote glare_55.png (blend-to-white, max alpha 0.85)")
