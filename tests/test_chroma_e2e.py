"""E2E: full StickerPipeline run with Grok call mocked to the saved green output.

Verifies the chroma path end to end: chroma extraction -> skip isnet -> deskew ->
border -> final RGBA, without spending an API call.
"""
import sys
sys.path.insert(0, "/root/workspace/Image-Deskew")
import numpy as np
from PIL import Image

GREEN = Image.open("/tmp/grok_green_out.png").convert("RGB")
SRC = Image.open(
    "/root/workspace/Image-Deskew/Test_stickes/cowf.ee_frieren-pout-kiss-cut_02_frieren-pout-kiss-cut-201601.jpg"
).convert("RGB")

from deskew_pipeline import StickerPipeline
from deskew_pipeline import grok_edit as grok_mod

calls = {"n": 0}


def fake_edit(image, api_key, prompt="", timeout_s=180):
    calls["n"] += 1
    assert "GREEN" in prompt.upper() or "#00FF00" in prompt, "prompt must request green bg"
    return GREEN, ""


grok_mod.grok_edit_image = fake_edit

pipe = StickerPipeline(
    segmentor_model="isnet-anime",
    grok_ai_edit=True,
    grok_api_key="dummy",
)
result = pipe.process(
    SRC,
    progress_callback=lambda p, m: print(f"  [{p:3d}%] {m}"),
)
result.save_all_stages("/tmp/e2e_chroma_out")

print("\n--- assertions ---")
assert calls["n"] == 1, f"grok edit called {calls['n']} times"
print("grok edit called exactly once: OK")
final = result.segmented_sticker if hasattr(result, "segmented_sticker") else result.final_sticker
print("final size:", final.size, "mode:", final.mode)
a = np.asarray(final)[:, :, 3] if final.mode == "RGBA" else None
assert a is not None, "final must be RGBA"
frac = (a > 128).mean()
print(f"opaque fraction: {frac:.3f}")
assert 0.05 < frac < 0.95, "sanity: subject occupies plausible area"
print("final is RGBA with plausible subject area: OK")
import os
files = sorted(os.listdir("/tmp/e2e_chroma_out"))
print("outputs:", files)
print("\nE2E CHROMA PATH PASS")
