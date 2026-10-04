"""E2E: /api/process with remove_watermarks on/off — stage outputs and timing."""
import json
import io
import base64
import time
import urllib.request
from PIL import Image

BASE = "http://127.0.0.1:8080"


def process(payload, timeout=580):
    req = urllib.request.Request(
        BASE + "/api/process",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


img = Image.open("tests/fixtures/watermarked_a52_tiles.png").convert("RGB")
img.thumbnail((1200, 1200))
buf = io.BytesIO()
img.save(buf, "JPEG", quality=85)
b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

t0 = time.time()
res_off = process({"image_base64": b64, "remove_watermarks": False, "deskew_mode": "auto"})
t_off = time.time() - t0

t0 = time.time()
res_on = process({"image_base64": b64, "remove_watermarks": True, "watermark_sensitivity": 60, "deskew_mode": "auto"})
t_on = time.time() - t0

print(f"OFF: success={res_off.get('success')} {t_off:.1f}s stages={sorted(res_off.get('stages', {}).keys())}")
print(f"ON:  success={res_on.get('success')} {t_on:.1f}s stages={sorted(res_on.get('stages', {}).keys())}")
print("watermark_cleaned present:", "watermark_cleaned" in res_on.get("stages", {}))
print("wm detection metadata:", json.dumps(res_on.get("watermark", res_on.get("metadata", {})), default=str)[:200])
