"""Test: residual green (drop shadows on the green screen) is keyed out, while
genuinely green art (green hair) is protected by the 25% safety valve."""
import sys
sys.path.insert(0, "/root/workspace/Image-Deskew")
import numpy as np
from PIL import Image
from deskew_pipeline.chroma_key import chroma_key_matte

# Case 1: red blob + sharp dark-green drop shadow offset to the right
arr = np.zeros((512, 512, 3), dtype=np.float32)
arr[..., 0] = 0.05; arr[..., 1] = 0.92; arr[..., 2] = 0.08   # pure green bg
arr[120:360, 120:300] = (0.75, 0.35, 0.30)                    # red character
arr[150:390, 310:380] = (0.18, 0.55, 0.22)                    # dark-green SHADOW
img = Image.fromarray((arr * 255).astype(np.uint8))
matte, _ = chroma_key_matte(img)
a = np.asarray(matte)[:, :, 3] / 255.0
char_opaque = (a[120:360, 120:300] > 0.9).mean()
shadow_opaque = (a[150:390, 310:380] > 0.5).mean()
print(f"case1 character opaque frac: {char_opaque:.3f} (want ~1.0)")
print(f"case1 shadow opaque frac:    {shadow_opaque:.3f} (want ~0.0)")
assert char_opaque > 0.95, "character must survive"
assert shadow_opaque < 0.05, "sharp green shadow must be keyed out"

# Case 2: GREEN-HAIRED character (green-dominant art) must survive via safety valve
arr2 = np.zeros((512, 512, 3), dtype=np.float32)
arr2[..., 0] = 0.05; arr2[..., 1] = 0.92; arr2[..., 2] = 0.08
arr2[60:220, 150:360] = (0.15, 0.75, 0.20)   # big green hair block (green-dominant)
arr2[220:420, 200:320] = (0.75, 0.35, 0.30)  # red body
img2 = Image.fromarray((arr2 * 255).astype(np.uint8))
matte2, _ = chroma_key_matte(img2)
a2 = np.asarray(matte2)[:, :, 3] / 255.0
hair_kept = (a2[60:220, 150:360] > 0.5).mean()
body_kept = (a2[220:420, 200:320] > 0.9).mean()
print(f"case2 green hair kept frac:  {hair_kept:.3f} (safety valve wants high)")
print(f"case2 red body kept frac:    {body_kept:.3f} (want ~1.0)")
assert body_kept > 0.95
assert hair_kept > 0.60, f"safety valve should keep green-dominant art, got {hair_kept:.3f}"

print("SHADOW REJECTION PASS")
