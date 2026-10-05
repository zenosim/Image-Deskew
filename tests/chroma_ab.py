"""A/B: isnet-anime neural segmentation vs chroma-key matte on the SAME Grok green image."""
import sys
sys.path.insert(0, "/root/workspace/Image-Deskew")
import numpy as np
from PIL import Image
from deskew_pipeline.chroma_key import chroma_key_matte
from deskew_pipeline.segmentor import BackgroundSegmentor
from tests.chroma_vis import checker

SRC = Image.open("/tmp/grok_green_out.png").convert("RGB")

# 1) isnet-anime on the green image (what the pipeline would do without chroma)
seg = BackgroundSegmentor(model_name="isnet-anime")
res = seg.segment(SRC)
isnet_rgba = res.rgba
isnet_rgba.save("/tmp/ab_isnet_rgba.png")

# 2) chroma key on the same image
ck_rgba, _ = chroma_key_matte(SRC)
ck_rgba.save("/tmp/ab_ck_rgba.png")

for tag, rgba in (("isnet", isnet_rgba), ("ck", ck_rgba)):
    a = np.asarray(rgba)[:, :, 3] / 255.0
    comp = checker(rgba.size)
    comp.paste(rgba, (0, 0), rgba)
    comp.save(f"/tmp/ab_{tag}_checker.png")
    crop_box = (int(rgba.size[0] * 0.35), int(rgba.size[1] * 0.02),
                int(rgba.size[0] * 0.80), int(rgba.size[1] * 0.40))
    crop = rgba.crop(crop_box)
    zw, zh = crop.size[0] * 2, crop.size[1] * 2
    big = crop.resize((zw, zh), Image.NEAREST)
    zc = checker((zw, zh))
    zc.paste(big, (0, 0), big)
    zc.save(f"/tmp/ab_{tag}_hairzoom.png")
    print(f"{tag}: fg={(a > 0.5).mean():.3f} partial={((a > 0.02) & (a < 0.98)).sum()}")
print("A/B saved")
