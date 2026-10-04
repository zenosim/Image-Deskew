"""Functional audit of every studio API endpoint the frontend calls."""
import json
import io
import base64
import urllib.request
import urllib.error
from PIL import Image

BASE = "http://127.0.0.1:8080"


def post(path, payload, timeout=240):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()[:180]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:180]
    except Exception as e:
        return type(e).__name__, str(e)[:150]


def get(path, timeout=30):
    try:
        with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
            return r.status, r.read().decode()[:120]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:150]
    except Exception as e:
        return type(e).__name__, str(e)[:150]


im = Image.open("Test_stickes/diesentai.com_ada-kisscut_01_ASDSADASASD.png").convert("RGB")
im.thumbnail((900, 900))
buf = io.BytesIO()
im.save(buf, "JPEG", quality=80)
b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

for p in ["/api/progress", "/api/lora-presets", "/api/web"]:
    print("GET", p, get(p))

print("POST /api/process", post("/api/process", {"image_base64": b64, "deskew_mode": "auto"})[0])
print("POST /api/auto-cel", post("/api/auto-cel", {"image_base64": b64}))
print("POST /api/auto-fill-holes", post("/api/auto-fill-holes", {"image_base64": b64}))
print("POST /api/inpaint", post("/api/inpaint", {"image_base64": b64, "mask_base64": b64, "model_type": "anime"})[0])
print("POST /api/download-lora-presets", post("/api/download-lora-presets", {})[0])
