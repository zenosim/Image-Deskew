"""E2E: grok_model + grok_custom_prompt reach the xAI payload; parallel jobs on
different models still share the concurrency gate. Mocked endpoint, no API cost."""
import sys, os, time, json, base64, io, threading, tempfile
sys.path.insert(0, "/root/workspace/Image-Deskew")

os.environ["GROK_MAX_CONCURRENT"] = "2"
from PIL import Image
import numpy as np
from deskew_pipeline import grok_edit

mock_img = Image.new("RGB", (256, 256), (20, 220, 20))
seen = []          # (model_id, prompt_tail) per call
state = {"active": 0, "max_seen": 0}
_lock = threading.Lock()


class FakeResp:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self):
        buf = io.BytesIO(); mock_img.save(buf, format="PNG")
        return json.dumps({"data": [{"b64_json": base64.b64encode(buf.getvalue()).decode()}]}).encode()


def fake_urlopen(req, timeout=None):
    body = json.loads(req.data.decode())
    with _lock:
        state["active"] += 1
        state["max_seen"] = max(state["max_seen"], state["active"])
        seen.append((body["model"], body["prompt"]))
    time.sleep(0.4)
    with _lock:
        state["active"] -= 1
    return FakeResp()


grok_edit.urllib.request.urlopen = fake_urlopen

from deskew_pipeline import StickerPipeline

tmpdir = tempfile.mkdtemp(prefix="grok_opts_e2e_")
cfgs = [
    # (model, custom_prompt)
    ("imagine-2", "Keep the fox tail and lantern exactly as drawn"),
    ("imagine-1", ""),
    ("imagine-2", None),
]
results = {}


def run_one(idx, model, custom):
    arr = np.full((512, 512, 3), 245, dtype=np.uint8)
    arr[128:384, 128:384] = (170, 90, 60)
    p = os.path.join(tmpdir, f"t{idx}.png")
    Image.fromarray(arr).save(p)
    pipe = StickerPipeline(
        segmentor_model="digital",
        grok_ai_edit=True,
        grok_api_key="dummy",
        grok_model=model,
        grok_custom_prompt=custom or "",
    )
    t0 = time.time()
    try:
        pipe.process(Image.open(p).convert("RGB"))
        results[idx] = ("ok", time.time() - t0)
    except Exception as e:
        results[idx] = (f"err: {e}", time.time() - t0)


threads = []
t0 = time.time()
for i, (m, c) in enumerate(cfgs):
    t = threading.Thread(target=run_one, args=(i, m, c))
    threads.append(t); t.start()
for t in threads:
    t.join()
wall = time.time() - t0

print(f"wall={wall:.2f}s max_grok_overlap={state['max_seen']} calls={len(seen)}")
for i, (model_id, prompt) in enumerate(seen):
    tail = prompt[-70:].replace("\n", " ")
    has_user = "ADDITIONAL USER INSTRUCTIONS" in prompt
    has_green = "#00FF00" in prompt
    print(f"  call {i}: model={model_id} green={has_green} user_block={has_user} tail=...{tail}")

# --- assertions ---
assert len(seen) == 3
models = [m for m, _ in seen]
assert sorted(models) == ["grok-imagine-image", "grok-imagine-image-2.0", "grok-imagine-image-2.0"], models
user_blocks = [p for _, p in seen if "ADDITIONAL USER INSTRUCTIONS" in p]
assert len(user_blocks) == 1, f"only the custom-prompt job should carry the user block: {len(user_blocks)}"
assert "Keep the fox tail and lantern exactly as drawn" in user_blocks[0]
for _, p in seen:
    assert "#00FF00" in p, "chroma green must be in every prompt"
assert state["max_seen"] == 2, f"gate must cap at 2 across mixed models, got {state['max_seen']}"
ok = sum(1 for v in results.values() if v[0] == "ok")
print(f"jobs ok: {ok}/3")
assert ok == 3
print("GROK OPTIONS E2E PASS")
