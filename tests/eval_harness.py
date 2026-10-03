"""Eval harness: run the StickerPipeline over fixtures + real stickers,
score results, save stage outputs. Used both as pytest module and CLI report.

Metrics:
- seg_quality: fraction semi-transparent edge pixels (lower = crisper)
- deskew_error_deg: |recovered angle - known angle| on rotated fixtures
- watermark_residual: edge density of watermark-colored pixels remaining
- runtime: per-stage and total seconds
"""
import os
import sys
import json
import time
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from deskew_pipeline import StickerPipeline

FIXTURES = os.path.join(ROOT, "tests", "fixtures")
OUTDIR = os.path.join(ROOT, "eval", "runs")
os.makedirs(OUTDIR, exist_ok=True)

# ground truth for deskew fixtures
GT_ANGLES = {
    "rotated_6.png": 6.0,
    "rotated_-3.png": -3.5,
    "shadowed_4.png": 4.0,
}


def seg_metrics(rgba: Image.Image):
    a = np.array(rgba)[:, :, 3]
    return {
        "semi_pct": float(((a > 10) & (a < 245)).mean() * 100),
        "opaque_pct": float((a >= 245).mean() * 100),
        "empty_pct": float((a <= 10).mean() * 100),
    }


def watermark_residual(rgba: Image.Image):
    """Count medium-gray pixels (watermark color range) that survived inside opaque region."""
    arr = np.array(rgba)
    if arr.shape[2] < 4:
        return 0.0
    rgb = arr[:, :, :3].astype(np.int16)
    a = arr[:, :, 3]
    gray = rgb.mean(axis=2)
    # watermark stamped at ~(120,120,120) w/ alpha blend -> grayish, low saturation
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    wm = (a > 200) & (np.abs(gray - 140) < 45) & (sat < 26)
    return float(wm.mean() * 100)


def run_case(name, path, deskew=True, save_stages=False):
    pipe = StickerPipeline(
        segmentor_model="isnet-anime",
        extract_mode="character_only",
        deskew_mode="auto" if deskew else "none",
    )
    t0 = time.time()
    try:
        result = pipe.process(path)
        dt = time.time() - t0
    except Exception as e:
        return {"name": name, "error": f"{type(e).__name__}: {e}", "runtime_s": round(time.time() - t0, 1)}

    out_img = getattr(result, "final_rgba", None)
    if out_img is None:
        attrs = [a for a in dir(result) if not a.startswith("_")]
        return {"name": name, "error": f"no output image; attrs={attrs}", "runtime_s": round(dt, 1)}

    m = seg_metrics(out_img.convert("RGBA"))
    m["wm_residual_pct"] = watermark_residual(out_img)
    meta = getattr(result, "metadata", None) or {}
    rec_angle = meta.get("deskew_angle", meta.get("applied_rotation"))
    gt = GT_ANGLES.get(os.path.basename(path))
    m["deskew_err_deg"] = (abs(float(rec_angle) - gt) if rec_angle is not None and gt else None)
    m["recovered_angle"] = rec_angle
    m["runtime_s"] = round(dt, 2)
    m["name"] = name
    out_img.save(os.path.join(OUTDIR, f"{name}.png"))
    if save_stages and hasattr(result, "save_all_stages"):
        result.save_all_stages(os.path.join(OUTDIR, f"{name}_stages"))
    return m


def main(paths=None, save_stages=False):
    cases = paths or sorted(
        [os.path.join(FIXTURES, f) for f in os.listdir(FIXTURES) if f.endswith(".png")]
    )
    results = []
    for p in cases:
        name = os.path.basename(p).rsplit(".", 1)[0]
        r = run_case(name, p, save_stages=save_stages)
        results.append(r)
        print(json.dumps(r, default=str))
    with open(os.path.join(OUTDIR, "report.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args if args else None, save_stages="--stages" in args)
