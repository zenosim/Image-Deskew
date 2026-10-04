"""A/B comparison: original vs current pipeline (v8) vs DENet on the towel ghost region."""
import numpy as np
import glob
from PIL import Image

region = (380, 480, 700, 660)  # x0, y0, x1, y1 in 900-space
rw, rh = region[2] - region[0], region[3] - region[1]
scale = 2

o = Image.open(glob.glob('Test_stickes/nekodecal.com_4th-wall*')[0]).convert('RGB')
o = o.resize((900, 900), Image.LANCZOS)

v8 = Image.open('/tmp/wolf_v8.png').convert('RGBA')
a = np.array(v8)[:, :, 3:4].astype(np.float32) / 255
v8c = (np.array(v8)[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)

dn = Image.open('/tmp/denet_out/nekodecal.com_4th-wall-waifu-stickers-1_.png').convert('RGB')

combo = Image.new('RGB', (rw * scale, (rh * scale + 10) * 3), 'white')
combo.paste(o.crop(region).resize((rw * scale, rh * scale), Image.LANCZOS), (0, 0))
combo.paste(Image.fromarray(v8c[region[1]:region[3], region[0]:region[2]]).resize((rw * scale, rh * scale), Image.LANCZOS), (0, rh * scale + 10))
combo.paste(dn.crop(region).resize((rw * scale, rh * scale), Image.LANCZOS), (0, (rh * scale + 10) * 2))
combo.save('/tmp/ab_towel.png')
print('saved: original top / v8 middle / denet bottom')
