"""Full UX smoke test: load page, pick test sticker, process with toggle states, download."""
import json
import io
import base64
import time
import urllib.request
from PIL import Image

BASE = "http://127.0.0.1:8080"
passed, failed = [], []


def check(name, cond, detail=""):
    (passed if cond else failed).append(f"{name} {detail}")
    print(("PASS" if cond else "FAIL"), name, detail)


# 1. page loads with all controls
html = urllib.request.urlopen(BASE + "/", timeout=15).read().decode()
for ctl in ["toggle-watermark-remover", "slider-watermark-sensitivity",
            "toggle-shine-remover", "toggle-color-pop", "btn-process-main",
            "btn-header-download", "btn-toggle-roi-mode", "btn-extract-char"]:
    check(f"UI has {ctl}", ctl in html)

appjs = urllib.request.urlopen(BASE + "/app.js", timeout=15).read().decode()
check("app.js serves watermark wiring", "toggleWatermarkRemover" in appjs)

# 2. process cycle: default settings (what a first-time user runs)
img = Image.open("Test_stickes/cowf.ee_frieren-pout-kiss-cut_02_frieren-pout-kiss-cut-201601.jpg").convert("RGB")
img.thumbnail((1200, 1200))
buf = io.BytesIO()
img.save(buf, "JPEG", quality=88)
b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def process(payload):
    req = urllib.request.Request(BASE + "/api/process", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


t0 = time.time()
r = process({"image_base64": b64, "deskew_mode": "auto"})
dt = time.time() - t0
check("default process succeeds", r.get("success") is True, f"{dt:.1f}s")
check("all 5 stages returned", set(r.get("stages", {})) >= {"raw", "segmented", "character_art", "deskewed", "final"})
check("fast enough for UX (<10s)", dt < 10, f"{dt:.1f}s")
check("final image non-empty", len(r["stages"].get("final", "")) > 1000)

# 3. with watermark toggle on
t0 = time.time()
r2 = process({"image_base64": b64, "remove_watermarks": True, "watermark_sensitivity": 60, "deskew_mode": "auto"})
dt2 = time.time() - t0
check("toggle-on process succeeds", r2.get("success") is True, f"{dt2:.1f}s")
check("watermark stage present", "watermark_cleaned" in r2.get("stages", {}))

# 4. progress endpoint reflects readiness
pr = json.loads(urllib.request.urlopen(BASE + "/api/progress", timeout=10).read())
check("progress endpoint ok", pr.get("success") is True and pr.get("percent", 0) >= 0)

print(f"\n{len(passed)} passed, {len(failed)} failed")
if failed:
    print("FAILURES:", failed)
    raise SystemExit(1)
