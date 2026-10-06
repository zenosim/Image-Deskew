"""Local Web Studio Server for Image Deskew & Character Extraction Pipeline.
Provides REST API endpoints and serves the visual studio UI.
"""

import os
import io
import json
import base64
import hashlib
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import urllib.request
import gzip
import zlib
import re
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = 300_000_000

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deskew_pipeline import (
    BackgroundSegmentor,
    CharacterExtractor,
    PerspectiveDeskewer,
    PostProcessor,
    BigLamaInpainter,
    ShineRemover,
    ColorEnhancer,
    StickerPipeline
)
from deskew_pipeline.color_enhancer import (
    get_all_presets,
    get_lora_preset_dirs,
    seed_safetensors_lora_presets,
    serialize_presets_for_json,
    AI_LORA_PRESETS,
    REAL_SAFETENSORS_DEFINITIONS
)
from deskew_pipeline.grok_edit import load_grok_key, save_grok_key, clear_grok_key, grok_key_configured
from deskew_pipeline import grok_edit as grok_mod

PORT = 8080
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
DEFAULT_OUTPUT_DIR = os.path.join(ROOT_DIR, "output_results")

_INPAINTERS = {}

def get_inpainter(model_type="anime"):
    global _INPAINTERS
    if model_type not in _INPAINTERS:
        _INPAINTERS[model_type] = BigLamaInpainter(model_type=model_type)
    return _INPAINTERS[model_type]


class CharacterArtCache:
    """Bounded cache for character art previews (capped to 2 entries to prevent RAM bloat)."""
    def __init__(self, max_entries: int = 2):
        self.max_entries = max_entries
        self.cache = {}
        self.order = []

    def set(self, key: str, img: Image.Image):
        if not key or img is None:
            return
        if key in self.cache:
            self.order.remove(key)
        elif len(self.order) >= self.max_entries:
            oldest = self.order.pop(0)
            self.cache.pop(oldest, None)
        self.cache[key] = img
        self.order.append(key)

    def get(self, key: str):
        return self.cache.get(key)


_CHARACTER_CACHE = CharacterArtCache(max_entries=2)
_PIPELINE_PROGRESS = {
    "percent": 0,
    "status": "Ready",
    "detail": ""
}

def set_pipeline_progress(percent: int, status: str, detail: str = ""):
    global _PIPELINE_PROGRESS
    _PIPELINE_PROGRESS["percent"] = max(0, min(100, int(percent)))
    _PIPELINE_PROGRESS["status"] = status
    _PIPELINE_PROGRESS["detail"] = detail


def open_folder_in_explorer(folder_path: str):
    """Opens a local folder in the native OS file explorer (Windows Explorer, Finder, or xdg-open)."""
    os.makedirs(folder_path, exist_ok=True)
    if sys.platform == "win32":
        try:
            os.startfile(folder_path)
            return True
        except Exception:
            import subprocess
            subprocess.Popen(["explorer.exe", os.path.normpath(folder_path)])
            return True
    elif sys.platform == "darwin":
        import subprocess
        subprocess.Popen(["open", folder_path])
        return True
    else:
        import subprocess
        subprocess.Popen(["xdg-open", folder_path])
        return True


def pil_to_base64_png(img: Image.Image, compress_level: int = 1) -> str:
    if img is None:
        return ""
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=compress_level)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


class StudioHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/progress":
            body = json.dumps({
                "success": True,
                "percent": _PIPELINE_PROGRESS["percent"],
                "status": _PIPELINE_PROGRESS["status"],
                "detail": _PIPELINE_PROGRESS["detail"]
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except Exception:
                pass
            return

        if path == "/api/test-images":
            images = []
            if os.path.exists(TEST_IMG_DIR):
                for f in sorted(os.listdir(TEST_IMG_DIR)):
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        images.append({
                            "name": f,
                            "path": os.path.join(TEST_IMG_DIR, f)
                        })
            body = json.dumps(images).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except Exception:
                pass
            return

        if path == "/api/lora-presets":
            all_p = get_all_presets()
            body = json.dumps({"success": True, "presets": serialize_presets_for_json(all_p)}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except Exception:
                pass
            return

        if path == "/api/web-proxy":
            query_params = urllib.parse.parse_qs(parsed.query)
            target_url = query_params.get("url", [None])[0]
            if not target_url:
                err = json.dumps({"error": "No url parameter provided"}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                try: self.wfile.write(err)
                except Exception: pass
                return

            target_url = urllib.parse.unquote(target_url).strip()
            if not target_url.startswith(("http://", "https://")):
                target_url = "https://" + target_url

            try:
                req = urllib.request.Request(
                    target_url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                        "Accept-Encoding": "gzip, deflate",
                        "Sec-Ch-Ua": '"Chromium";v="128", "Not;A=Brand";v="24", "Google Chrome";v="128"',
                        "Sec-Ch-Ua-Mobile": "?0",
                        "Sec-Ch-Ua-Platform": '"Windows"',
                        "Sec-Fetch-Dest": "document",
                        "Sec-Fetch-Mode": "navigate",
                        "Sec-Fetch-Site": "cross-site",
                        "Upgrade-Insecure-Requests": "1"
                    }
                )
                import ssl
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE

                with urllib.request.urlopen(req, context=ctx, timeout=12) as resp:
                    content = resp.read()
                    content_encoding = resp.headers.get("Content-Encoding", "").lower()
                    if "gzip" in content_encoding or content[:2] == b'\x1f\x8b':
                        try:
                            content = gzip.decompress(content)
                        except Exception:
                            pass
                    elif "deflate" in content_encoding:
                        try:
                            content = zlib.decompress(content)
                        except Exception:
                            pass

                    content_type = resp.headers.get("Content-Type", "text/html; charset=utf-8")

                    if "text/html" in content_type.lower():
                        try:
                            html_str = content.decode("utf-8", errors="replace")
                            # Strip CSP meta tags and X-Frame-Options meta tags
                            html_str = re.sub(
                                r'<meta[^>]+http-equiv=["\']?(?:Content-Security-Policy|X-Frame-Options)["\']?[^>]*>',
                                '',
                                html_str,
                                flags=re.IGNORECASE
                            )
                            # Strip integrity hashes that break when injected
                            html_str = re.sub(r'\sintegrity=["\'][^"\']*["\']', '', html_str, flags=re.IGNORECASE)

                            # Shim frame busters and inject base href
                            anti_framebuster = """<script>
try {
  Object.defineProperty(window, 'top', { get: function() { return window.self; } });
  Object.defineProperty(window, 'parent', { get: function() { return window.self; } });
} catch(e) {}
</script>"""
                            base_tag = f'{anti_framebuster}<base href="{target_url}">'
                            if "<head>" in html_str:
                                html_str = html_str.replace("<head>", f"<head>{base_tag}", 1)
                            elif "<HEAD>" in html_str:
                                html_str = html_str.replace("<HEAD>", f"<HEAD>{base_tag}", 1)
                            else:
                                html_str = base_tag + html_str
                            content = html_str.encode("utf-8")
                        except Exception:
                            pass

                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(content)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
                    self.send_header("Cache-Control", "public, max-age=300")
                    self.end_headers()
                    try: self.wfile.write(content)
                    except Exception: pass
                    return
            except Exception as e:
                err_html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Web Proxy Notice</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 32px; line-height: 1.6; }}
.card {{ max-width: 600px; margin: 40px auto; background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 24px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
h2 {{ color: #f43f5e; margin-top: 0; font-size: 1.3rem; }}
p {{ color: #94a3b8; font-size: 0.95rem; }}
code {{ background: #0f172a; padding: 2px 6px; border-radius: 4px; color: #38bdf8; font-family: monospace; word-break: break-all; }}
.tip {{ background: rgba(56, 189, 248, 0.1); border-left: 4px solid #38bdf8; padding: 12px; border-radius: 6px; margin-top: 16px; color: #bae6fd; font-size: 0.9rem; }}
.snip-now-btn {{ display: inline-flex; align-items: center; gap: 8px; margin-top: 14px; padding: 10px 18px; background: linear-gradient(135deg, #0284c7, #2563eb); color: #fff; font-weight: 600; border: none; border-radius: 8px; cursor: pointer; font-size: 0.95rem; transition: transform 0.15s ease; }}
.snip-now-btn:hover {{ transform: translateY(-1px); filter: brightness(1.1); }}
</style>
</head>
<body>
<div class="card">
  <h2>🌐 External Site Restricts Direct Embedding</h2>
  <p>Target URL: <code>{target_url}</code></p>
  <p><strong>Notice:</strong> This website is protected by Cloudflare bot protection or cross-origin iframe security policies.</p>
  <div class="tip">
    <div>💡 <strong>1-Click Solution:</strong> You can snip directly from this website or any other open tab using the native browser snipper with zero restrictions!</div>
    <button class="snip-now-btn" onclick="try {{ window.parent.postMessage({{action: 'start_screen_snip'}}, '*'); }} catch(e) {{}}">
      📸 Snip This Tab / Window Now
    </button>
  </div>
</div>
</body>
</html>""".encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(err_html)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                try: self.wfile.write(err_html)
                except Exception: pass
                return

        # Serve output results if requested
        if path.startswith("/output_results/"):
            img_rel = urllib.parse.unquote(path[len("/output_results/"):])
            img_path = os.path.join(DEFAULT_OUTPUT_DIR, img_rel)
            if os.path.exists(img_path) and os.path.isfile(img_path):
                ext = os.path.splitext(img_path)[1].lower()
                mime = "image/png" if ext == ".png" else "image/jpeg"
                with open(img_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    self.wfile.write(content)
                except Exception:
                    pass
                return

        # Direct static file serving
        clean = path.lstrip("/")
        if clean == "" or clean == "index.html":
            file_to_serve = os.path.join(STATIC_DIR, "index.html")
            content_type = "text/html; charset=utf-8"
        elif clean == "app.js":
            file_to_serve = os.path.join(STATIC_DIR, "app.js")
            content_type = "application/javascript; charset=utf-8"
        elif clean == "style.css":
            file_to_serve = os.path.join(STATIC_DIR, "style.css")
            content_type = "text/css; charset=utf-8"
        elif clean.endswith(".png"):
            file_to_serve = os.path.join(STATIC_DIR, clean)
            content_type = "image/png"
        elif clean.endswith((".jpg", ".jpeg")):
            file_to_serve = os.path.join(STATIC_DIR, clean)
            content_type = "image/jpeg"
        elif clean.endswith(".svg"):
            file_to_serve = os.path.join(STATIC_DIR, clean)
            content_type = "image/svg+xml"
        else:
            file_to_serve = os.path.join(STATIC_DIR, "index.html")
            content_type = "text/html; charset=utf-8"

        if os.path.exists(file_to_serve) and os.path.isfile(file_to_serve):
            try:
                with open(file_to_serve, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.end_headers()
                self.wfile.write(content)
                return
            except Exception as e:
                pass

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/process":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))
            set_pipeline_progress(2, "Initializing pipeline...", "Reading image and settings...")

            # Load image from base64 or file path
            image_path = data.get("image_path")
            image_base64 = data.get("image_base64")

            raw_img = None
            img_source_id = None

            if image_base64:
                header, encoded = image_base64.split(",", 1) if "," in image_base64 else ("", image_base64)
                raw_img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
                import hashlib
                img_source_id = hashlib.md5(encoded[:2000].encode("utf-8")).hexdigest()
            elif image_path:
                clean_img_path = urllib.parse.unquote(str(image_path)).strip()
                resolved_path = None
                if os.path.isabs(clean_img_path) and os.path.exists(clean_img_path):
                    resolved_path = clean_img_path
                elif os.path.exists(os.path.join(ROOT_DIR, clean_img_path)):
                    resolved_path = os.path.join(ROOT_DIR, clean_img_path)
                elif os.path.exists(os.path.join(TEST_IMG_DIR, os.path.basename(clean_img_path))):
                    resolved_path = os.path.join(TEST_IMG_DIR, os.path.basename(clean_img_path))
                elif os.path.exists(clean_img_path):
                    resolved_path = clean_img_path

                if resolved_path and os.path.exists(resolved_path):
                    raw_img = Image.open(resolved_path).convert("RGB")
                    img_source_id = resolved_path

            if raw_img is not None:
                # Protect against OOM crashes on huge images (>4096px)
                max_dim = max(raw_img.size)
                if max_dim > 4096:
                    scale = 4096.0 / max_dim
                    new_w = max(1, int(round(raw_img.size[0] * scale)))
                    new_h = max(1, int(round(raw_img.size[1] * scale)))
                    raw_img = raw_img.resize((new_w, new_h), Image.LANCZOS)

            # If no image was sent, do NOT arbitrarily substitute an old sticker!
            if raw_img is None:
                err_body = json.dumps({
                    "success": False,
                    "error": "No image loaded. Please upload an image or choose a test sticker."
                }).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                try:
                    self.wfile.write(err_body)
                except Exception:
                    pass
                return

            # Extract parameters
            model_name = data.get("model_name", "isnet-anime")
            extract_mode = data.get("extract_mode", "character_only")
            deskew_mode = data.get("deskew_mode", "auto")
            margin_inset_px = int(data.get("margin_inset_px", 0))
            padding_px = int(data.get("padding_px", 12))
            defringe = bool(data.get("defringe", True))
            ensure_connected = bool(data.get("ensure_connected", True))
            connectivity_mode = data.get("connectivity_mode", "bridge")
            bridge_thickness_px = int(data.get("bridge_thickness_px", 10))

            add_diecut_border = bool(data.get("add_diecut_border", False))
            border_thickness_px = int(data.get("border_thickness_px", 14))
            border_mode = str(data.get("border_mode", "solid"))
            border_smoothing = int(data.get("border_smoothing", 50))
            border_color_hex = str(data.get("border_color", "#FFFFFF")).lstrip("#")
            try:
                if len(border_color_hex) == 6:
                    border_color = tuple(int(border_color_hex[i:i+2], 16) for i in (0, 2, 4))
                else:
                    border_color = (255, 255, 255)
            except Exception:
                border_color = (255, 255, 255)
            highlight_glow_radius = int(data.get("highlight_glow_radius", 20))
            highlight_box_padding = int(data.get("highlight_box_padding", 16))

            enhance_resolution = bool(data.get("enhance_resolution", False))
            enhance_scale = int(data.get("enhance_scale", 2))
            enhance_model = str(data.get("enhance_model", "ultrasharp_lite"))
            preserve_colors = bool(data.get("preserve_colors", True))

            # Character ROI bounding box prompt
            character_bbox = data.get("character_bbox")
            parsed_bbox = None
            if character_bbox and isinstance(character_bbox, (list, tuple)) and len(character_bbox) == 4:
                try:
                    parsed_bbox = tuple(int(round(float(v))) for v in character_bbox)
                except Exception:
                    parsed_bbox = None

            # Character Highlighter prompt mask (drawn freehand by user)
            character_highlight_mask = data.get("character_highlight_mask")
            parsed_hint_mask = None
            if character_highlight_mask:
                try:
                    if isinstance(character_highlight_mask, str) and character_highlight_mask.startswith("data:"):
                        h_header, h_encoded = character_highlight_mask.split(",", 1)
                        h_img = Image.open(io.BytesIO(base64.b64decode(h_encoded)))
                        h_arr = np.array(h_img)
                        if h_arr.ndim == 3:
                            if h_arr.shape[2] == 4:
                                parsed_hint_mask = (h_arr[:, :, 3] > 30).astype(np.uint8) * 255
                            else:
                                parsed_hint_mask = (h_arr.max(axis=2) > 30).astype(np.uint8) * 255
                        else:
                            parsed_hint_mask = (h_arr > 30).astype(np.uint8) * 255
                except Exception as e:
                    print(f"[Server] Warning: Failed to decode character_highlight_mask: {e}")
                    parsed_hint_mask = None

            # Character Negative Exclusion Mask (drawn to strictly exclude unwanted parts)
            character_negative_mask = data.get("character_negative_mask")
            parsed_negative_mask = None
            if character_negative_mask:
                try:
                    if isinstance(character_negative_mask, str) and character_negative_mask.startswith("data:"):
                        n_header, n_encoded = character_negative_mask.split(",", 1)
                        n_img = Image.open(io.BytesIO(base64.b64decode(n_encoded)))
                        n_arr = np.array(n_img)
                        if n_arr.ndim == 3:
                            if n_arr.shape[2] == 4:
                                parsed_negative_mask = (n_arr[:, :, 3] > 30).astype(np.uint8) * 255
                            else:
                                parsed_negative_mask = (n_arr.max(axis=2) > 30).astype(np.uint8) * 255
                        else:
                            parsed_negative_mask = (n_arr > 30).astype(np.uint8) * 255
                except Exception as e:
                    print(f"[Server] Warning: Failed to decode character_negative_mask: {e}")
                    parsed_negative_mask = None

            auto_fill_holes = bool(data.get("auto_fill_holes", False))

            pca_align = bool(data.get("pca_align", False))
            pca_align_mode = data.get("pca_align_mode", "neutral")

            pitch = float(data.get("pitch_deg", 0.0))
            yaw = float(data.get("yaw_deg", 0.0))
            roll = float(data.get("roll_deg", 0.0))
            manual_rotation = float(data.get("manual_rotation_deg", 0.0))
            horizontal_skew = float(data.get("horizontal_skew_deg", 0.0))
            vertical_skew = float(data.get("vertical_skew_deg", 0.0))
            flip_horizontal = bool(data.get("flip_horizontal", False))
            flip_vertical = bool(data.get("flip_vertical", False))
            clean_shadow_smudges = bool(data.get("clean_shadow_smudges", True))
            smudge_sensitivity = int(data.get("smudge_sensitivity", 50))
            alpha_threshold = int(data.get("alpha_threshold", 15))
            edge_inset_px = int(data.get("edge_inset_px", 0))

            # Pre-Processing: Watermark & Anime De-Shine
            remove_watermarks = bool(data.get("remove_watermarks", False))
            watermark_sensitivity = int(data.get("watermark_sensitivity", 50))
            watermark_region = str(data.get("watermark_region", "full"))
            grok_ai_edit = bool(data.get("grok_ai_edit", False))
            grok_model = str(data.get("grok_model", "imagine-2") or "imagine-2")
            grok_custom_prompt = str(data.get("grok_custom_prompt", "") or "")[:2000]
            remove_shine = bool(data.get("remove_shine", False))
            shine_strength = int(data.get("shine_strength", 60))
            color_tiers = int(data.get("color_tiers", 40))
            flat_cel_look = bool(data.get("flat_cel_look", False))

            # Hair gap cleanup & Color Pop
            clean_hair_gaps = bool(data.get("clean_hair_gaps", True))
            color_pop_preset = str(data.get("color_pop_preset", "off"))
            # Manual background-removal override on top of the Grok edit
            # ('' or None = default chroma key; model name = force that segmentor)
            grok_background_model = str(data.get("grok_background_model", "") or "").strip() or None
            ai_lora_enabled = bool(data.get("ai_lora_enabled", False))
            ai_lora_preset = str(data.get("ai_lora_preset", "lora_shinkai"))
            if ai_lora_enabled and color_pop_preset != "off":
                color_pop_preset = ai_lora_preset

            if color_pop_preset != "off":
                color_pop_vibrance = float(data.get("color_pop_vibrance", 1.0)) if data.get("color_pop_vibrance") is not None else 1.0
                color_pop_clarity = float(data.get("color_pop_clarity", 0.0)) if data.get("color_pop_clarity") is not None else 0.0
            else:
                color_pop_vibrance = None
                color_pop_clarity = None
            save_to_local = bool(data.get("save_to_local", False))
            output_dir = str(data.get("output_dir", "output_results")).strip()
            output_filename = data.get("output_filename")
            lightweight_response = bool(data.get("lightweight_response", False))

            try:
                # Execute pipeline
                pipeline = StickerPipeline(
                    segmentor_model=model_name,
                    extract_mode=extract_mode,
                    deskew_mode=deskew_mode,
                    padding_px=padding_px,
                    defringe=defringe,
                    ensure_connected=ensure_connected,
                    connectivity_mode=connectivity_mode,
                    bridge_thickness_px=bridge_thickness_px,
                    pca_align=pca_align,
                    pca_align_mode=pca_align_mode,
                    add_diecut_border=add_diecut_border,
                    border_thickness_px=border_thickness_px,
                    border_color=border_color,
                    border_mode=border_mode,
                    border_smoothing=border_smoothing,
                    highlight_glow_radius=highlight_glow_radius,
                    highlight_box_padding=highlight_box_padding,
                    enhance_resolution=enhance_resolution,
                    enhance_scale=enhance_scale,
                    enhance_model=enhance_model,
                    preserve_colors=preserve_colors,
                    manual_rotation_deg=manual_rotation,
                    horizontal_skew_deg=horizontal_skew,
                    vertical_skew_deg=vertical_skew,
                    flip_horizontal=flip_horizontal,
                    flip_vertical=flip_vertical,
                    alpha_threshold=alpha_threshold,
                    clean_shadow_smudges=clean_shadow_smudges,
                    smudge_sensitivity=smudge_sensitivity,
                    edge_inset_px=edge_inset_px,
                    clean_hair_gaps=clean_hair_gaps,
                    remove_watermarks=remove_watermarks,
                    watermark_sensitivity=watermark_sensitivity,
                    watermark_region=watermark_region,
                    grok_ai_edit=grok_ai_edit,
                    grok_api_key=(load_grok_key() if grok_ai_edit else ""),
                    grok_model=grok_model,
                    grok_custom_prompt=grok_custom_prompt,
                    grok_background_model=grok_background_model,
                    remove_shine=remove_shine,
                    shine_strength=shine_strength,
                    color_tiers=color_tiers,
                    flat_cel_look=flat_cel_look,
                    color_pop_preset=color_pop_preset,
                    color_pop_vibrance=color_pop_vibrance,
                    color_pop_clarity=color_pop_clarity,
                    character_bbox=parsed_bbox,
                    character_hint_mask=parsed_hint_mask,
                    character_negative_mask=parsed_negative_mask,
                    auto_fill_holes=auto_fill_holes
                )
                # Override margin inset if custom
                pipeline.character_extractor.margin_inset_px = margin_inset_px

                set_pipeline_progress(5, "Pipeline initialized. Processing started...")
                result = pipeline.process(
                    raw_img,
                    parametric_pitch=pitch,
                    parametric_yaw=yaw,
                    parametric_roll=roll,
                    manual_rotation=manual_rotation,
                    horizontal_skew=horizontal_skew,
                    vertical_skew=vertical_skew,
                    precomputed_segmentation=None,  # always run every stage fresh from the original image
                    character_bbox=parsed_bbox,
                    character_hint_mask=parsed_hint_mask,
                    character_negative_mask=parsed_negative_mask,
                    progress_callback=set_pipeline_progress
                )

                # Store into bounded character cache for instant preview/auto-cel
                if result.character_art is not None:
                    _CHARACTER_CACHE.set(img_source_id, result.character_art)
                    _CHARACTER_CACHE.set("latest", result.character_art)

                saved_local_path = None
                if save_to_local:
                    target_out_dir = os.path.join(ROOT_DIR, output_dir) if not os.path.isabs(output_dir) else output_dir
                    os.makedirs(target_out_dir, exist_ok=True)
                    if output_filename:
                        base_name = os.path.splitext(os.path.basename(str(output_filename)))[0]
                    elif image_path:
                        base_name = os.path.splitext(os.path.basename(str(image_path)))[0]
                    elif data.get("name"):
                        base_name = os.path.splitext(os.path.basename(str(data.get("name"))))[0]
                    else:
                        base_name = f"sticker_{int(time.time() * 1000)}"

                    clean_fname = f"{base_name}_clean.png"
                    saved_local_path = os.path.join(target_out_dir, clean_fname)
                    counter = 1
                    while os.path.exists(saved_local_path):
                        saved_local_path = os.path.join(target_out_dir, f"{base_name}_clean_{counter}.png")
                        counter += 1
                    result.final_rgba.save(saved_local_path, format="PNG")

                set_pipeline_progress(96, "Packaging previews & asset...", "Encoding")

                def _preview_img(im: Image.Image, max_dim: int = 1280) -> Image.Image:
                    if im is None:
                        return None
                    w, h = im.size
                    if max(w, h) > max_dim:
                        factor = max_dim / max(w, h)
                        nw, nh = max(1, int(w * factor)), max(1, int(h * factor))
                        return im.resize((nw, nh), Image.Resampling.BILINEAR)
                    return im

                if lightweight_response:
                    stages_dict = {
                        "final": pil_to_base64_png(result.final_rgba, compress_level=1)
                    }
                else:
                    stages_dict = {
                        "raw": pil_to_base64_png(_preview_img(result.raw_image), compress_level=1),
                        "segmented": pil_to_base64_png(_preview_img(result.segmented_sticker), compress_level=1),
                        "character_art": pil_to_base64_png(_preview_img(result.character_art), compress_level=1),
                        "deskewed": pil_to_base64_png(_preview_img(result.deskewed_image), compress_level=1),
                        "final": pil_to_base64_png(result.final_rgba, compress_level=1)
                    }
                    if result.watermark_cleaned is not None:
                        stages_dict["watermark_cleaned"] = pil_to_base64_png(_preview_img(result.watermark_cleaned), compress_level=1)
                    if result.grok_edited is not None:
                        stages_dict["grok_edited"] = pil_to_base64_png(_preview_img(result.grok_edited), compress_level=1)
                    if result.cel_restored is not None:
                        stages_dict["cel_restored"] = pil_to_base64_png(_preview_img(result.cel_restored), compress_level=1)
                    if result.color_enhanced is not None:
                        stages_dict["color_enhanced"] = pil_to_base64_png(_preview_img(result.color_enhanced), compress_level=1)

                set_pipeline_progress(100, "Asset ready!", "Completed")
                response = {
                    "success": True,
                    "stages": stages_dict,
                    "metadata": result.metadata
                }
                if saved_local_path:
                    response["saved_path"] = saved_local_path
                    response["saved_filename"] = os.path.basename(saved_local_path)

                res_body = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(res_body)))
                self.end_headers()
                try:
                    self.wfile.write(res_body)
                except Exception:
                    pass

                # Free large arrays and run garbage collector
                del stages_dict
                del result
                del raw_img
                import gc
                gc.collect()
                return
            except Exception as e:
                import traceback
                traceback.print_exc()
                err_body = json.dumps({"success": False, "error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                try:
                    self.wfile.write(err_body)
                except Exception:
                    pass
                return

        elif parsed.path == "/api/inpaint":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8"))

            image_base64 = data.get("image_base64")
            image_path = data.get("image_path")
            mask_base64 = data.get("mask_base64")
            dilate_px = int(data.get("dilate_px", 8))
            model_type = str(data.get("model_type", "anime"))  # 'anime' (default) or 'general'

            if not mask_base64:
                err_body = json.dumps({"success": False, "error": "No inpainting mask provided."}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                try:
                    self.wfile.write(err_body)
                except Exception:
                    pass
                return

            raw_img = None
            if image_base64:
                header, encoded = image_base64.split(",", 1) if "," in image_base64 else ("", image_base64)
                raw_img = Image.open(io.BytesIO(base64.b64decode(encoded)))
            elif image_path:
                clean_img_path = urllib.parse.unquote(str(image_path)).strip()
                if os.path.isabs(clean_img_path) and os.path.exists(clean_img_path):
                    raw_img = Image.open(clean_img_path)
                elif os.path.exists(os.path.join(ROOT_DIR, clean_img_path)):
                    raw_img = Image.open(os.path.join(ROOT_DIR, clean_img_path))
                elif os.path.exists(os.path.join(TEST_IMG_DIR, os.path.basename(clean_img_path))):
                    raw_img = Image.open(os.path.join(TEST_IMG_DIR, os.path.basename(clean_img_path)))

            if raw_img is None:
                err_body = json.dumps({"success": False, "error": "No image loaded to inpaint."}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                try:
                    self.wfile.write(err_body)
                except Exception:
                    pass
                return

            # Decode mask
            m_header, m_encoded = mask_base64.split(",", 1) if "," in mask_base64 else ("", mask_base64)
            mask_img = Image.open(io.BytesIO(base64.b64decode(m_encoded)))

            try:
                inpainter = get_inpainter(model_type=model_type)
                inpainted_result = inpainter.inpaint(
                    image=raw_img,
                    mask=mask_img,
                    dilate_px=dilate_px
                )
                res_b64 = pil_to_base64_png(inpainted_result)
                res_body = json.dumps({
                    "success": True,
                    "image_base64": res_b64
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(res_body)))
                self.end_headers()
                try:
                    self.wfile.write(res_body)
                except Exception:
                    pass
                return
            except Exception as e:
                import traceback
                traceback.print_exc()
                err_body = json.dumps({"success": False, "error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                try:
                    self.wfile.write(err_body)
                except Exception:
                    pass
                return

        elif parsed.path == "/api/auto-cel":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8")) if body else {}

            image_base64 = data.get("image_base64")
            image_path = data.get("image_path")

            raw_img = None
            clean_img_path = urllib.parse.unquote(str(image_path)).strip() if image_path else ""
            if image_base64:
                header, encoded = image_base64.split(",", 1) if "," in image_base64 else ("", image_base64)
                raw_img = Image.open(io.BytesIO(base64.b64decode(encoded)))
            elif clean_img_path:
                if os.path.isabs(clean_img_path) and os.path.exists(clean_img_path):
                    raw_img = Image.open(clean_img_path)
                elif os.path.exists(os.path.join(ROOT_DIR, clean_img_path)):
                    raw_img = Image.open(os.path.join(ROOT_DIR, clean_img_path))
                elif os.path.exists(os.path.join(TEST_IMG_DIR, os.path.basename(clean_img_path))):
                    raw_img = Image.open(os.path.join(TEST_IMG_DIR, os.path.basename(clean_img_path)))

            if raw_img is None:
                err_body = json.dumps({"success": False, "error": "No image loaded to analyze."}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                try: self.wfile.write(err_body)
                except Exception: pass
                return

            try:
                shiner = ShineRemover()
                # Check for cached extracted character art to analyze real character, not background
                cached_char = _CHARACTER_CACHE.get(clean_img_path or image_path) or _CHARACTER_CACHE.get("latest")
                target_img = raw_img
                char_mask = None
                if cached_char is not None:
                    target_img = cached_char
                    if target_img.mode == "RGBA":
                        char_mask = (np.array(target_img)[:, :, 3] > 15).astype(np.uint8) * 255
                elif raw_img.mode == "RGBA":
                    char_mask = (np.array(raw_img)[:, :, 3] > 15).astype(np.uint8) * 255

                report = shiner.analyze_for_auto_cel(target_img, character_mask=char_mask)
                rec_str = int(report.get("recommended_strength", 60))
                rec_tiers = int(report.get("recommended_tiers", 40))

                # Immediately generate the rendered preview so user sees changes applied instantly!
                preview_img = target_img
                w, h = preview_img.size
                if max(w, h) > 768:
                    scale = 768.0 / max(w, h)
                    preview_img = preview_img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

                preview_res = shiner.remove_shine(preview_img, strength=rec_str, color_tiers=rec_tiers, flat_cel_look=False)
                report["preview_base64"] = pil_to_base64_png(preview_res.cleaned_image)

                res_body = json.dumps(report).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(res_body)))
                self.end_headers()
                try: self.wfile.write(res_body)
                except Exception: pass
                return
            except Exception as e:
                err_body = json.dumps({"success": False, "error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                try: self.wfile.write(err_body)
                except Exception: pass
                return

        elif parsed.path == "/api/cel-preview":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8")) if body else {}

            image_base64 = data.get("image_base64")
            image_path = data.get("image_path")
            strength = int(data.get("strength", 60))
            color_tiers = int(data.get("color_tiers", 40))
            flat_cel_look = bool(data.get("flat_cel_look", False))

            raw_img = None
            clean_img_path = urllib.parse.unquote(str(image_path)).strip() if image_path else ""
            if image_base64:
                header, encoded = image_base64.split(",", 1) if "," in image_base64 else ("", image_base64)
                raw_img = Image.open(io.BytesIO(base64.b64decode(encoded)))
            elif clean_img_path:
                if os.path.isabs(clean_img_path) and os.path.exists(clean_img_path):
                    raw_img = Image.open(clean_img_path)
                elif os.path.exists(os.path.join(ROOT_DIR, clean_img_path)):
                    raw_img = Image.open(os.path.join(ROOT_DIR, clean_img_path))
                elif os.path.exists(os.path.join(TEST_IMG_DIR, os.path.basename(clean_img_path))):
                    raw_img = Image.open(os.path.join(TEST_IMG_DIR, os.path.basename(clean_img_path)))

            if raw_img is None:
                err_body = json.dumps({"success": False, "error": "No image loaded for preview."}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                try: self.wfile.write(err_body)
                except Exception: pass
                return

            try:
                # Use cached extracted character if available so preview reflects final character output
                cached_char = _CHARACTER_CACHE.get(clean_img_path or image_path) or _CHARACTER_CACHE.get("latest")
                preview_img = cached_char if cached_char is not None else raw_img
                # Downsample preview if large (> 768px) for sub-20ms instant interactivity
                w, h = preview_img.size
                if max(w, h) > 768:
                    scale = 768.0 / max(w, h)
                    preview_img = preview_img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

                shiner = ShineRemover()
                res = shiner.remove_shine(preview_img, strength=strength, color_tiers=color_tiers, flat_cel_look=flat_cel_look)
                res_b64 = pil_to_base64_png(res.cleaned_image)
                res_body = json.dumps({"success": True, "image_base64": res_b64}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(res_body)))
                self.end_headers()
                try: self.wfile.write(res_body)
                except Exception: pass
                return
            except Exception as e:
                err_body = json.dumps({"success": False, "error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()

        elif parsed.path == "/api/color-pop-preview":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8")) if body else {}

            image_base64 = data.get("image_base64")
            image_path = data.get("image_path")
            preset = str(data.get("preset", "anime_pop"))
            ai_lora_enabled = bool(data.get("ai_lora_enabled", False))
            ai_lora_preset = str(data.get("ai_lora_preset", "lora_shinkai"))
            target_preset = ai_lora_preset if ai_lora_enabled else preset
            vibrance = float(data.get("vibrance", 1.0))
            clarity = float(data.get("clarity", 0.0))

            raw_img = None
            clean_img_path = urllib.parse.unquote(str(image_path)).strip() if image_path else ""
            if image_base64:
                header, encoded = image_base64.split(",", 1) if "," in image_base64 else ("", image_base64)
                raw_img = Image.open(io.BytesIO(base64.b64decode(encoded)))
            elif clean_img_path:
                if os.path.isabs(clean_img_path) and os.path.exists(clean_img_path):
                    raw_img = Image.open(clean_img_path)
                elif os.path.exists(os.path.join(ROOT_DIR, clean_img_path)):
                    raw_img = Image.open(os.path.join(ROOT_DIR, clean_img_path))
                elif os.path.exists(os.path.join(TEST_IMG_DIR, os.path.basename(clean_img_path))):
                    raw_img = Image.open(os.path.join(TEST_IMG_DIR, os.path.basename(clean_img_path)))

            if raw_img is None:
                err_body = json.dumps({"success": False, "error": "No image loaded for preview."}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                try: self.wfile.write(err_body)
                except Exception: pass
                return

            try:
                cached_char = _CHARACTER_CACHE.get(clean_img_path or image_path) or _CHARACTER_CACHE.get("latest")
                preview_img = cached_char if cached_char is not None else raw_img
                w, h = preview_img.size
                if max(w, h) > 768:
                    scale = 768.0 / max(w, h)
                    preview_img = preview_img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

                enhancer = ColorEnhancer()
                res = enhancer.enhance(preview_img, preset=target_preset, vibrance=vibrance, clarity=clarity)
                res_b64 = pil_to_base64_png(res.enhanced_image)
                res_body = json.dumps({"success": True, "image_base64": res_b64, "metadata": res.metadata}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(res_body)))
                self.end_headers()
                try: self.wfile.write(res_body)
                except Exception: pass
                return
            except Exception as e:
                err_body = json.dumps({"success": False, "error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
        elif parsed.path == "/api/auto-fill-holes":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode("utf-8")) if body else {}

            image_base64 = data.get("image_base64")
            image_path = data.get("image_path")
            model_type = data.get("model_type", "anime")

            raw_img = None
            if image_base64:
                header, encoded = image_base64.split(",", 1) if "," in image_base64 else ("", image_base64)
                raw_img = Image.open(io.BytesIO(base64.b64decode(encoded)))
            elif image_path:
                clean_img_path = urllib.parse.unquote(str(image_path)).strip()
                if os.path.isabs(clean_img_path) and os.path.exists(clean_img_path):
                    raw_img = Image.open(clean_img_path)
                elif os.path.exists(os.path.join(ROOT_DIR, clean_img_path)):
                    raw_img = Image.open(os.path.join(ROOT_DIR, clean_img_path))
                elif os.path.exists(os.path.join(TEST_IMG_DIR, os.path.basename(clean_img_path))):
                    raw_img = Image.open(os.path.join(TEST_IMG_DIR, os.path.basename(clean_img_path)))

            if raw_img is None:
                err_body = json.dumps({"success": False, "error": "No image loaded to fill holes."}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                try: self.wfile.write(err_body)
                except Exception: pass
                return

            try:
                inpainter = BigLamaInpainter(model_type=model_type)
                healed_img, hole_count = inpainter.auto_fill_holes(raw_img)
                res_b64 = pil_to_base64_png(healed_img)
                res_body = json.dumps({
                    "success": True,
                    "image_base64": res_b64,
                    "holes_filled": hole_count
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(res_body)))
                self.end_headers()
                try: self.wfile.write(res_body)
                except Exception: pass
                return
            except Exception as e:
                err_body = json.dumps({"success": False, "error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                try: self.wfile.write(err_body)
                except Exception: pass
                return

        elif parsed.path == "/api/download-lora-presets":
            try:
                target_dir = os.path.join(ROOT_DIR, "lora_presets")
                os.makedirs(target_dir, exist_ok=True)
                installed_files = seed_safetensors_lora_presets(target_dir)
                installed_count = len(installed_files)

                all_presets = get_all_presets()
                res_body = json.dumps({
                    "success": True,
                    "installed_count": installed_count,
                    "message": f"Successfully downloaded and verified {installed_count} real anime .safetensors LoRA packs!",
                    "presets": serialize_presets_for_json(all_presets)
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(res_body)))
                self.end_headers()
                try: self.wfile.write(res_body)
                except Exception: pass
                return
            except Exception as e:
                err_body = json.dumps({"success": False, "error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                try: self.wfile.write(err_body)
                except Exception: pass
                return

        elif parsed.path == "/api/grok-key":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                data = json.loads(body.decode("utf-8")) if body else {}
            except Exception:
                data = {}
            action = str(data.get("action", "status"))
            if action == "set":
                api_key = str(data.get("api_key", "")).strip()
                if not api_key:
                    res_body = json.dumps({"success": False, "error": "api_key is empty"}).encode("utf-8")
                    self.send_response(400)
                else:
                    save_grok_key(api_key)
                    res_body = json.dumps({"success": True, "configured": True}).encode("utf-8")
                    self.send_response(200)
            elif action == "clear":
                clear_grok_key()
                res_body = json.dumps({"success": True, "configured": False}).encode("utf-8")
                self.send_response(200)
            else:  # status
                res_body = json.dumps({
                    "success": True,
                    "configured": grok_key_configured(),
                    "models": grok_mod.GROK_MODELS,
                    "default_model": grok_mod.GROK_DEFAULT_MODEL,
                }).encode("utf-8")
                self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(res_body)))
            self.end_headers()
            try: self.wfile.write(res_body)
            except Exception: pass
            return

        elif parsed.path == "/api/save-local":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            data = json.loads(body.decode("utf-8")) if body else {}
            rel_dir = data.get("output_dir", "output_results")
            target_dir = os.path.join(ROOT_DIR, rel_dir) if not os.path.isabs(rel_dir) else rel_dir
            os.makedirs(target_dir, exist_ok=True)

            items = data.get("items")
            if not items:
                img_b64 = data.get("image_base64")
                fname = data.get("filename", "sticker.png")
                if img_b64:
                    items = [{"image_base64": img_b64, "filename": fname}]
                else:
                    items = []

            saved_paths = []
            for itm in items:
                b64_str = itm.get("image_base64", "")
                if not b64_str:
                    continue
                header, encoded = b64_str.split(",", 1) if "," in b64_str else ("", b64_str)
                fname = itm.get("filename", "sticker.png")
                clean_name = os.path.basename(fname).strip()
                if not clean_name:
                    clean_name = f"sticker_{int(time.time() * 1000)}.png"
                if not clean_name.lower().endswith((".png", ".webp", ".jpg", ".jpeg")):
                    clean_name += ".png"

                out_path = os.path.join(target_dir, clean_name)
                base, ext = os.path.splitext(clean_name)
                counter = 1
                while os.path.exists(out_path):
                    out_path = os.path.join(target_dir, f"{base}_{counter}{ext}")
                    counter += 1

                try:
                    img_data = base64.b64decode(encoded)
                    with open(out_path, "wb") as f:
                        f.write(img_data)
                    saved_paths.append(out_path)
                except Exception as e:
                    print(f"[Server] Failed to write {out_path}: {e}")

            import gc
            gc.collect()

            res_body = json.dumps({
                "success": True,
                "saved_count": len(saved_paths),
                "saved_paths": saved_paths,
                "output_dir": target_dir
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(res_body)))
            self.end_headers()
            try: self.wfile.write(res_body)
            except Exception: pass
            return

        elif parsed.path == "/api/open-output-folder":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            data = json.loads(body.decode("utf-8")) if body else {}
            rel_dir = data.get("output_dir", "output_results")
            target_dir = os.path.join(ROOT_DIR, rel_dir) if not os.path.isabs(rel_dir) else rel_dir
            try:
                open_folder_in_explorer(target_dir)
                res_body = json.dumps({"success": True, "path": target_dir}).encode("utf-8")
                self.send_response(200)
            except Exception as e:
                res_body = json.dumps({"success": False, "error": str(e)}).encode("utf-8")
                self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(res_body)))
            self.end_headers()
            try: self.wfile.write(res_body)
            except Exception: pass
            return

        self.send_response(404)
        self.end_headers()


def run_server(port: int = PORT, host: str = "0.0.0.0", on_bound=None):
    os.makedirs(STATIC_DIR, exist_ok=True)
    try:
        seed_safetensors_lora_presets(os.path.join(ROOT_DIR, "lora_presets"))
    except Exception as e:
        print(f"[Studio Server] Note: LoRA presets check: {e}")
    ThreadingHTTPServer.allow_reuse_address = True

    server = None
    actual_port = port
    for p in range(port, port + 20):
        try:
            server = ThreadingHTTPServer((host, p), StudioHandler)
            actual_port = p
            break
        except OSError as e:
            # Error 10048 is WSAEADDRINUSE, 10013 is WSAEACCES on Windows, 98/13 on Linux
            err_code = getattr(e, 'winerror', None) or getattr(e, 'errno', None)
            if err_code in (10048, 10013, 98, 13) or "10048" in str(e) or "10013" in str(e):
                print(f"[Studio Server] Port {p} is currently in use or unavailable ({e}). Trying port {p + 1}...", flush=True)
                continue
            raise

    if server is None:
        raise RuntimeError(f"Could not bind to any port in range {port}-{port+20}")

    server.daemon_threads = True
    url = f"http://localhost:{actual_port}"
    print(f"==================================================", flush=True)
    print(f"Sticker Deskew Studio running at: {url}", flush=True)
    print(f"==================================================", flush=True)

    if on_bound:
        try:
            on_bound(actual_port)
        except Exception as e:
            print(f"[Studio Server] Warning in on_bound callback: {e}", flush=True)

    server.serve_forever()


if __name__ == "__main__":
    run_server()

