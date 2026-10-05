"""Fresh-image robustness test: Grok green-chroma edit + chroma key on Shadowheart sticker."""
import sys
sys.path.insert(0, "/root/workspace/Image-Deskew")
from PIL import Image
from deskew_pipeline import grok_edit
from deskew_pipeline.chroma_key import chroma_key_matte
from tests.chroma_vis import checker

SRC_PATH = "/root/workspace/Image-Deskew/Test_stickes/cowf.ee_shadowheart-kiss-cut-1_03_shadowheart-kiss-cut-972301.jpg"
OUT = "/tmp/grok_sh_green_out.png"

src = Image.open(SRC_PATH).convert("RGB")
print(f"input: {src.size}")

key = grok_edit.load_grok_key()
assert key, "no grok key configured"
prompt = grok_edit.build_grok_prompt(chroma_bg="green")
edited, err = grok_edit.grok_edit_image(src, key, prompt=prompt, timeout_s=240)
if edited is None:
    print("GROK FAILED:", err)
    sys.exit(1)
edited.save(OUT)

matte, bg = chroma_key_matte(edited)
matte.save("/tmp/grok_sh_cut.png")
import numpy as np
a = np.asarray(matte)[:, :, 3] / 255.0
rgb = np.asarray(matte)[:, :, :3].astype(np.float32)
pm = (a > 0.05) & (a < 0.95)
spill = (rgb[..., 1] - np.maximum(rgb[..., 0], rgb[..., 2]))[pm]
print(f"bg sampled: {(bg * 255).round().astype(int)}")
print(f"fg={(a > 0.5).mean():.3f} partial={pm.sum()} mean_green_excess={spill.mean() if pm.sum() else 0:.1f} frac_green={(spill > 20).mean() if pm.sum() else 0:.3f}")

comp = checker(matte.size)
comp.paste(matte, (0, 0), matte)
comp.save("/tmp/grok_sh_checker.png")
w, h = matte.size
crop = matte.crop((int(w * 0.25), int(h * 0.02), int(w * 0.85), int(h * 0.45)))
zw, zh = crop.size[0] * 2, crop.size[1] * 2
big = crop.resize((zw, zh), Image.NEAREST)
zc = checker((zw, zh))
zc.paste(big, (0, 0), big)
zc.save("/tmp/grok_sh_hairzoom.png")
print("saved /tmp/grok_sh_green_out.png + cut + zooms")
