"""Grok (xAI) image-edit pre-processing stage.

When enabled, the raw input image is first sent to grok-imagine-image (image EDIT
endpoint) with a single expert instruction prompt that asks the model to, in ONE
edit pass:
  1. cut the main character (+ anything they hold/wear) out of the scene,
  2. place it on a plain solid #FFFFFF white background,
  3. erase every watermark / text stamp / tiled overlay,
  4. clean shadows/glare left by the removed background,
  5. gently pop the colors (vibrant, true to the original art style).

Grok does NOT output transparency, so downstream the normal pipeline still runs:
isnet-anime segmentation pulls the character off the white canvas, then
deskew/border/color finishing applies exactly as usual.

The API key is stored OUTSIDE the repository in ~/.config/sticker-studio/grok_api_key.json
(chmod 600) via the /api/grok-key endpoints — it can never be committed/pushed. The
XAI_API_KEY environment variable takes precedence when set. Never hard-coded, never logged.
"""

import os
import io
import json
import base64
import random
import threading
import time
import urllib.request
import urllib.error
from typing import Optional, Tuple
from PIL import Image
from .chroma_key import CHROMA_BG_INSTRUCTIONS

XAI_EDIT_URL = "https://api.x.ai/v1/images/edits"
XAI_EDIT_MODEL = "grok-imagine-image-2.0"

# Grok edit models available on the xAI API (user-selectable in the studio UI).
GROK_MODELS = {
    "imagine-1": "grok-imagine-image",
    "imagine-2": "grok-imagine-image-2.0",
}
GROK_DEFAULT_MODEL = "imagine-2"

# Maximum number of Grok edit API calls allowed to run at the same time.
# All studio requests share this one global gate: callers beyond the cap block
# here (in their own thread) until a slot frees up, so N parallel jobs from the
# UI still hit the xAI API at most GROK_MAX_CONCURRENT at once.
# Override with the GROK_MAX_CONCURRENT env var (min 1).
_env_conc = None
try:
    _env_conc = int(os.environ.get("GROK_MAX_CONCURRENT", ""))
except (TypeError, ValueError):
    _env_conc = None
GROK_MAX_CONCURRENT = max(1, _env_conc if _env_conc else 2)
_grok_slot = threading.BoundedSemaphore(GROK_MAX_CONCURRENT)
# Key store lives OUTSIDE the repository (user config dir) so it can never be
# committed/pushed by accident. Env var XAI_API_KEY is the primary source; the
# file is the dashboard-managed fallback.
KEY_STORE_DIR = os.path.join(os.path.expanduser("~"), ".config", "sticker-studio")
KEY_STORE_PATH = os.path.join(KEY_STORE_DIR, "grok_api_key.json")

# Enhancement IS Grok's job: when the AI edit succeeds, every local color stage
# (color pop, cel restore, shine removal) is force-disabled downstream, so this
# sentence is the one place image enhancement is requested.
GROK_ENHANCE_INSTRUCTION = (
    "4. COLOR ENHANCEMENT & CEL CLEANUP (your edit is the ONLY enhancement this artwork receives — "
    "no other tool will process it afterwards): INCREASE THE COLOR of the image — clearly boost vibrance "
    "and saturation so every hue is rich, lively and print-ready, and deepen contrast slightly so "
    "line art stays crisp and blacks stay clean. REMOVE ALL HAZE AND UNWANTED SHINE: any foggy, milky "
    "or washed-out overlay, atmospheric haze, and any specular shine, gloss, sheen or glare highlight "
    "that does not belong to the original artwork (plastic-sleeve reflections, scanner sheen, photo "
    "hotspots) must be cleaned away. CONVERT THE SHADING TO CLEAN CEL ART: replace soft/airbrushed or "
    "photographic-looking shading with crisp cel-shaded tones — flat discrete color tiers, hard clean "
    "shadow edges, uniform fills — matching the artwork's own cel style. Keep skin tones natural. "
    "This is a finishing touch, not a re-style: do NOT "
    "re-style, re-paint, re-render, stylize, or reinterpret the artwork. Preserve the original "
    "artist's exact drawing style, line weight, proportions, facial features, pose, expression "
    "and palette identity. The character at the end must be unmistakably the SAME character "
    "from the source image, just cleaned with stronger, more vivid cel-shaded color."
)

# Single consolidated instruction prompt. This is THE prompt — everything the edit
# must do is specified here; no other prompt exists in the codebase.
GROK_EDIT_PROMPT = (
    "You are performing a precise professional image restoration edit on this anime/manga-style "
    "sticker artwork. Follow EVERY instruction below exactly, in this single edit pass, without "
    "adding anything that was not requested.\n\n"
    "YOUR TASK — this source image is a collectible sticker/character sheet that must be prepared "
    "for clean die-cut printing:\n\n"
    "1. SUBJECT ISOLATION: Identify the ONE main character — the central illustrated figure the "
    "artwork is about. Isolate that character completely: include their entire body, head, hair, "
    "ears/tail/wings/horns or any non-human features, their clothing and accessories, and "
    "CRITICALLY everything they are holding, carrying, wearing or interacting with (props, "
    "weapons, food, signs, plushies, tools, bags, effects emanating from their hands). If a "
    "smaller secondary subject is physically attached to or held by the main character, keep it. "
    "Do NOT crop any part of the character or their held items — the full silhouette must remain "
    "inside the frame with a comfortable margin.\n\n"
    "2. BACKGROUND REPLACEMENT: Remove the entire original background — every scenic element, "
    "pattern, gradient, texture, text panel, logo, decorative shape and border. CRITICALLY: remove "
    "EVERY shadow on the background — soft drop shadows, sharp/hard-edged shadow silhouettes cast "
    "behind or beside the character, contact shadows, ground shadows, even faint gray or tinted "
    "shadow shapes. Sharp, well-defined shadows count as background, not art — erase them "
    "completely so the character floats cleanly with nothing beneath, behind or beside it. "
    "Replace all of that with "
    "{bg_mid} filling the rest of the "
    "canvas. No gradients, no vignettes, no shadows of any kind on the background, no remnants "
    "of the old scene. The character should look like a clean sticker scan on a perfectly even, "
    "solid-color backdrop.\n\n"
    "3. WATERMARK ELIMINATION: Erase ALL watermarks, text, logos, URLs, usernames, signatures "
    "and tiled/repeating overlay patterns — including faint translucent 'ghost' text and "
    "semi-transparent tiled watermark grids that overlap the character's hair, skin, clothing "
    "and any cloth/props. Reconstruct the artwork underneath each watermark so it looks like the "
    "watermark never existed: continue line art, shading, fabric texture and color seamlessly. "
    "The result must contain zero readable or semi-readable text of any kind.\n\n"
    "4. SHADOW & GLARE CLEANUP: Remove any photographic defects that came from the original "
    "background or from the sticker being photographed/scaned: any remaining shadows on or around "
    "the character, specular glare/hotspots on shiny printed surfaces, moiré patterns, and color "
    "casts from ambient lighting. The character should have the flat, even lighting of native "
    "digital art.\n\n"
    + GROK_ENHANCE_INSTRUCTION + "\n\n"
    "OUTPUT REQUIREMENTS: Output the edited image at the same aspect ratio and resolution "
    "class as the input. Full-bleed {bg_end}, character fully in frame, no text "
    "anywhere, no borders, no frames, no rounded corners, no mockups — just the cleaned, "
    "enhanced artwork on the plain solid-color backdrop with no shadows."
)


def load_grok_key() -> str:
    env_key = os.environ.get("XAI_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        with open(KEY_STORE_PATH, "r", encoding="utf-8") as f:
            return (json.load(f).get("api_key") or "").strip()
    except Exception:
        return ""


def save_grok_key(api_key: str) -> None:
    os.makedirs(os.path.dirname(KEY_STORE_PATH), exist_ok=True)
    tmp = KEY_STORE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"api_key": api_key.strip()}, f)
    os.replace(tmp, KEY_STORE_PATH)
    try:
        os.chmod(KEY_STORE_PATH, 0o600)
    except Exception:
        pass


def clear_grok_key() -> None:
    try:
        os.remove(KEY_STORE_PATH)
    except FileNotFoundError:
        pass


def grok_key_configured() -> bool:
    return bool(load_grok_key())


def build_grok_prompt(chroma_bg: str = "white", custom_prompt: str = "") -> str:
    """Returns THE Grok edit prompt with the requested background color instruction.

    chroma_bg='green' swaps the white-background instruction for chroma-key green
    so deskew_pipeline.chroma_key.chroma_key_matte can extract the character with
    a hard-edged distance matte (measured better than isnet-anime on hair).

    custom_prompt: optional user extra instructions appended after the OUTPUT
    REQUIREMENTS block. The user's text may only ADD requirements (franchise
    style, accessories to keep, framing hints...); the core isolation/cleanup
    rules above stay authoritative.
    """
    bg_mid, bg_end = CHROMA_BG_INSTRUCTIONS.get(chroma_bg, CHROMA_BG_INSTRUCTIONS["white"])
    prompt = GROK_EDIT_PROMPT.replace("{bg_mid}", bg_mid).replace("{bg_end}", bg_end)
    # Guard: no unfilled placeholders may ever reach the model
    assert "{bg_mid}" not in prompt and "{bg_end}" not in prompt
    extra = (custom_prompt or "").strip()
    if extra:
        prompt += (
            "\n\nADDITIONAL USER INSTRUCTIONS (higher priority than anything above; "
            "follow them exactly, they never override the rules they do not mention):\n"
            f"{extra}"
        )
    return prompt


def grok_edit_image(
    image: Image.Image,
    api_key: str,
    prompt: str = GROK_EDIT_PROMPT,
    timeout_s: int = 180,
    model: Optional[str] = None,
) -> Tuple[Optional[Image.Image], str]:
    """Calls the xAI image-edit endpoint. Returns (edited_image | None, error_message).

    model: 'imagine-1' | 'imagine-2' (default) or a raw model id. The concurrency
    gate below applies to every model — calls beyond GROK_MAX_CONCURRENT wait
    here until a slot frees; the wait time is logged so saturation is visible.
    """
    if not api_key:
        return None, "No Grok API key configured."
    model_id = GROK_MODELS.get(model, model) if model else GROK_MODELS[GROK_DEFAULT_MODEL]
    t_gate = time.time()
    with _grok_slot:
        waited = time.time() - t_gate
        if waited > 0.5:
            print(f"[GrokEdit] concurrency gate: waited {waited:.1f}s for a free slot "
                  f"(max {GROK_MAX_CONCURRENT} concurrent)")
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        payload = json.dumps({
            "model": model_id,
            "prompt": prompt,
            "image": {
                "url": f"data:image/png;base64,{b64}",
                "type": "image_url",
            },
            # Zero-Data-Retention accounts reject URL-format responses (the URL would require
            # xAI to store the generated image). Always request base64 output.
            "response_format": "b64_json",
        }).encode("utf-8")
        req = urllib.request.Request(
            XAI_EDIT_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        t0 = time.time()
        # Rate-limit aware retry: batch jobs (multiple images at once) regularly
        # hit xAI 429s when several edits land together. Retry 429/5xx with
        # exponential backoff + jitter, honoring Retry-After when present, so
        # the job eventually succeeds instead of failing the whole batch.
        max_attempts = int(os.environ.get("GROK_MAX_RETRIES", "4"))
        attempt = 0
        while True:
            attempt += 1
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8")[:500]
                except Exception:
                    pass
                retryable = e.code in (429, 500, 502, 503, 504)
                if retryable and attempt < max_attempts:
                    delay = None
                    try:
                        delay = float(e.headers.get("Retry-After", ""))
                    except (TypeError, ValueError, AttributeError):
                        delay = None
                    if delay is None:
                        delay = min(30.0, (2 ** attempt) * 1.5) + random.uniform(0.0, 1.5)
                    print(f"[GrokEdit] HTTP {e.code} (attempt {attempt}/{max_attempts}), "
                          f"retrying in {delay:.1f}s: {detail[:120]}")
                    time.sleep(delay)
                    # Rebuild the request: a Request object can be reused, but
                    # be explicit so headers/state are always fresh.
                    req = urllib.request.Request(
                        XAI_EDIT_URL,
                        data=payload,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}",
                        },
                        method="POST",
                    )
                    continue
                return None, f"Grok API HTTP {e.code}: {detail}"
            except Exception as e:
                # Network-level transient failures also deserve one retry
                if attempt < max_attempts and isinstance(e, (urllib.error.URLError, TimeoutError, ConnectionError)):
                    delay = min(30.0, (2 ** attempt) * 1.5) + random.uniform(0.0, 1.5)
                    print(f"[GrokEdit] transient {type(e).__name__} (attempt {attempt}/{max_attempts}), "
                          f"retrying in {delay:.1f}s")
                    time.sleep(delay)
                    req = urllib.request.Request(
                        XAI_EDIT_URL,
                        data=payload,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}",
                        },
                        method="POST",
                    )
                    continue
                return None, f"Grok API request failed: {e}"

        data = body.get("data") or []
        if not data:
            return None, f"Grok API returned no image data: {str(body)[:300]}"
        item = data[0]
        img = None
        url = item.get("url")
        b64_out = item.get("b64_json")
        if b64_out:
            img = Image.open(io.BytesIO(base64.b64decode(b64_out)))
        elif url:
            with urllib.request.urlopen(url, timeout=timeout_s) as r:
                img = Image.open(io.BytesIO(r.read()))
        if img is None:
            return None, "Grok API response contained neither url nor b64_json."
        print(f"[GrokEdit] Edited image received in {time.time() - t0:.1f}s ({img.size[0]}x{img.size[1]})")
        return img.convert("RGB"), ""
