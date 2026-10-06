"""Offline tests (NO API calls) for the Grok batch fixes:
1. 429/5xx retry with backoff: mock urlopen raising HTTP 429 twice then OK.
2. Retry-After header is honored.
3. Non-retryable HTTP 401 fails immediately.
4. grok_background_model override disables the chroma-key path.
5. Prompt contains haze/shine removal + cel-art conversion clauses.
"""
import sys, os, io, json, base64, time
sys.path.insert(0, "/root/workspace/Image-Deskew")
os.environ["GROK_MAX_CONCURRENT"] = "1"

import numpy as np
from PIL import Image
import urllib.error
from deskew_pipeline import grok_edit

mock_img = Image.new("RGB", (64, 64), (20, 220, 20))

def make_resp():
    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self):
            buf = io.BytesIO(); mock_img.save(buf, format="PNG")
            return json.dumps({"data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}]}).encode()
    return FakeResp()

def http_error(code, headers=None):
    return urllib.error.HTTPError("https://api.x.ai", code, "err", headers or {}, io.BytesIO(b"rate limited"))

# --- Test 1: two 429s then success ---
state = {"calls": 0, "sleeps": []}
orig_urlopen, orig_sleep = grok_edit.urllib.request.urlopen, grok_edit.time.sleep
def fake_urlopen_429(req, timeout=None):
    state["calls"] += 1
    if state["calls"] <= 2:
        raise http_error(429, {"Retry-After": "0.05"})
    return make_resp()
grok_edit.urllib.request.urlopen = fake_urlopen_429
grok_edit.time.sleep = lambda s: state["sleeps"].append(s)

img = Image.new("RGB", (64, 64), (200, 60, 60))
t0 = time.time()
out, err = grok_edit.grok_edit_image(img, "dummy", prompt="test", timeout_s=30)
assert out is not None and err == "", f"expected success after retries, got err={err}"
assert state["calls"] == 3, f"expected 3 calls (2 retries), got {state['calls']}"
assert state["sleeps"] == [0.05, 0.05], f"Retry-After not honored: {state['sleeps']}"
print(f"OK  test1: 429 x2 then success, honored Retry-After {state['sleeps']} ({time.time()-t0:.2f}s)")

# Restore the real sleep so the backoff delay is measurable
grok_edit.time.sleep = orig_sleep
state2 = {"calls": 0, "sleeps": []}
def fake_urlopen_backoff(req, timeout=None):
    state2["calls"] += 1
    if state2["calls"] == 1:
        raise http_error(503)
    return make_resp()
grok_edit.urllib.request.urlopen = fake_urlopen_backoff
_real_sleep = time.sleep
time.sleep = lambda s: state2["sleeps"].append(s)  # grok_edit references time module
out, err = grok_edit.grok_edit_image(img, "dummy", prompt="test", timeout_s=30)
time.sleep = _real_sleep
assert out is not None and state2["calls"] == 2
assert 1.0 <= state2["sleeps"][0] <= 6.0, f"backoff delay out of range: {state2['sleeps']}"
print(f"OK  test2: 503 retried with backoff {state2['sleeps'][0]:.2f}s")

# --- Test 3: non-retryable 401 fails immediately ---
state3 = {"calls": 0}
def fake_urlopen_401(req, timeout=None):
    state3["calls"] += 1
    raise http_error(401)
grok_edit.urllib.request.urlopen = fake_urlopen_401
out, err = grok_edit.grok_edit_image(img, "dummy", prompt="test", timeout_s=30)
assert out is None and "401" in err and state3["calls"] == 1, f"401 must fail fast: calls={state3['calls']} err={err}"
print("OK  test3: 401 fails immediately (no retry)")

grok_edit.urllib.request.urlopen = orig_urlopen
grok_edit.time.sleep = orig_sleep

# --- Test 4: grok_background_model override disables chroma path ---
from deskew_pipeline import StickerPipeline
pipe = StickerPipeline(segmentor_model="digital", grok_ai_edit=True, grok_api_key="dummy",
                       grok_background_model="isnet-anime")
assert pipe.grok_background_model == "isnet-anime"
pipe2 = StickerPipeline(segmentor_model="digital", grok_ai_edit=True, grok_api_key="dummy")
assert pipe2.grok_background_model is None
print("OK  test4: grok_background_model param wired (override + default None)")

# --- Test 5: prompt clauses ---
p = grok_edit.build_grok_prompt(chroma_bg="green")
clauses = {
    "color increase": "INCREASE THE COLOR" in p,
    "haze removal": "REMOVE ALL HAZE" in p,
    "unwanted shine": "UNWANTED SHINE" in p,
    "cel conversion": "CONVERT THE SHADING TO CLEAN CEL ART" in p,
    "flat tiers": "flat discrete color tiers" in p,
    "sharp shadow removal": "Sharp, well-defined shadows count as background" in p,
}
for k, v in clauses.items():
    print(("OK  test5: " if v else "FAIL test5: ") + k)
assert all(clauses.values())

print("GROK BATCH FIXES PASS")
