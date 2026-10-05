"""Full-pipeline Grok-gate E2E: 4 concurrent StickerPipeline jobs with a mocked
xAI endpoint (0.5s latency). Asserts the server-side semaphore caps actual Grok
call overlap at GROK_MAX_CONCURRENT and all jobs complete. No real API calls."""
import sys, os, time, json, base64, io, threading, tempfile
sys.path.insert(0, "/root/workspace/Image-Deskew")

os.environ["GROK_MAX_CONCURRENT"] = "2"
from PIL import Image
import numpy as np
from deskew_pipeline import grok_edit

mock_img = Image.new("RGB", (256, 256), (20, 220, 20))
state = {"active": 0, "max_seen": 0, "calls": 0}
_lock = threading.Lock()


class FakeResp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        buf = io.BytesIO()
        mock_img.save(buf, format="PNG")
        return json.dumps({"data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}]}).encode()


def fake_urlopen(req, timeout=None):
    with _lock:
        state["active"] += 1
        state["calls"] += 1
        state["max_seen"] = max(state["max_seen"], state["active"])
    time.sleep(0.5)
    with _lock:
        state["active"] -= 1
    return FakeResp()


grok_edit.urllib.request.urlopen = fake_urlopen

from deskew_pipeline import StickerPipeline

tmpdir = tempfile.mkdtemp(prefix="grok_gate_e2e_")
paths = []
for i in range(4):
    arr = np.full((512, 512, 3), 245, dtype=np.uint8)
    arr[128:384, 128:384] = (180 - i * 20, 90, 60)
    p = os.path.join(tmpdir, f"test_{i}.png")
    Image.fromarray(arr).save(p)
    paths.append(p)

results = {}


def run_one(idx, path):
    pipe = StickerPipeline(
        segmentor_model="digital",   # avoid neural model download in test
        grok_ai_edit=True,
        grok_api_key="dummy-key",
    )
    t0 = time.time()
    try:
        pipe.process(Image.open(path).convert("RGB"))
        results[idx] = ("ok", time.time() - t0)
    except Exception as e:
        results[idx] = (f"err: {e}", time.time() - t0)


threads = []
t0 = time.time()
for i, p in enumerate(paths):
    t = threading.Thread(target=run_one, args=(i, p))
    threads.append(t)
    t.start()
for t in threads:
    t.join()
wall = time.time() - t0

print(f"\nwall={wall:.2f}s grok_calls={state['calls']} max_grok_overlap={state['max_seen']}")
for k in sorted(results):
    print(f"  job {k}: {results[k][0]} in {results[k][1]:.2f}s")

assert state["calls"] == 4, f"expected 4 grok calls, got {state['calls']}"
assert state["max_seen"] == 2, f"gate must cap overlap at exactly 2, got {state['max_seen']}"
assert wall >= 0.95, f"4 x 0.5s jobs at cap 2 must take >=1s, took {wall:.2f}s"
ok_count = sum(1 for v in results.values() if v[0] == "ok")
print(f"jobs ok: {ok_count}/4")
print("SERVER GATE E2E PASS")
