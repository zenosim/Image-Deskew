"""Check canvas heuristic on the synthetic fixture."""
import sys
sys.path.insert(0, ".")
import numpy as np
from PIL import Image, ImageDraw

img = Image.new("RGB", (120, 200), color=(245, 180, 170))
d = ImageDraw.Draw(img)
d.rectangle([40, 80, 80, 120], fill=(255, 255, 255))
rgb = np.array(img)
border = np.vstack([
    rgb[:15].reshape(-1, 3),
    rgb[-15:].reshape(-1, 3),
    rgb[:, :15].reshape(-1, 3),
    rgb[:, -15:].reshape(-1, 3),
])
bg = np.median(border, axis=0)
d_skin = np.linalg.norm(rgb.astype(float) - bg, axis=2)
print("bg median:", bg)
print("skin px flagged as canvas frac:", round(float((d_skin < 30).mean()), 2))
