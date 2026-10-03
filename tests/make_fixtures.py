"""Synthetic fixture generator for the deskew pipeline test suite.

Creates deterministic test images by transforming the real stickers in
Test_stickes/ : rotation, shadows, watermarks, glare, complex backgrounds.
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

STICKER_DIR = os.path.join(ROOT, "Test_stickes")
FIXTURE_DIR = os.path.join(ROOT, "tests", "fixtures")
os.makedirs(FIXTURE_DIR, exist_ok=True)

CLEAN = os.path.join(STICKER_DIR, "diesentai.com_ada-kisscut_01_ASDSADASASD.png")
COMPLEX = os.path.join(STICKER_DIR, "nekodecal.com_roller-maid-stickers_12_3392.jpg")


def _load(path, max_dim=1400):
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    return img


def make_rotated(angle=6.0, bg="white"):
    """Character sticker rotated on a plain background (deskew test)."""
    img = _load(CLEAN)
    canvas = Image.new("RGB", (int(img.width * 1.25), int(img.height * 1.25)), bg)
    pos = ((canvas.width - img.width) // 2, (canvas.height - img.height) // 2)
    rot = img.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=bg)
    canvas.paste(rot, ((canvas.width - rot.width) // 2, (canvas.height - rot.height) // 2))
    out = os.path.join(FIXTURE_DIR, f"rotated_{int(angle)}.png")
    canvas.save(out)
    return out


def make_shadowed(angle=4.0):
    """Rotated sticker with a soft drop shadow (segmentor + deskew stress)."""
    img = _load(CLEAN)
    rot = img.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor="white")
    canvas = Image.new("RGB", (rot.width + 200, rot.height + 200), "white")
    # shadow: cutout alpha -> blurred black offset
    alpha = rot.convert("L").point(lambda v: 255 if v < 245 else 0)
    shadow = Image.new("RGBA", rot.size, (60, 60, 60, 0))
    shadow.putalpha(alpha.point(lambda v: 110 if v else 0))
    shadow = shadow.filter(ImageFilter.GaussianBlur(9))
    canvas.paste(Image.new("RGB", rot.size, "white"), (100, 100))
    canvas.paste((40, 40, 40), (118, 122), shadow.split()[3])
    canvas.paste(rot, (100, 100))
    out = os.path.join(FIXTURE_DIR, f"shadowed_{int(angle)}.png")
    canvas.save(out)
    return out


def make_watermarked(text="@pixelartist", alpha_val=52, tiles=True):
    """Sticker with light overlaid watermark text (watermark remover test)."""
    img = _load(CLEAN)
    overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", max(24, img.width // 22)
        )
    except OSError:
        font = ImageFont.load_default()
    if tiles:
        step_x, step_y = img.width // 3, img.height // 4
        for iy in range(4):
            for ix in range(3):
                draw.text(
                    (ix * step_x + 20, iy * step_y + 30),
                    text,
                    fill=(120, 120, 120, alpha_val),
                    font=font,
                )
    else:
        draw.text((img.width // 4, img.height // 2), text, fill=(120, 120, 120, alpha_val), font=font)
    out_img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    out = os.path.join(FIXTURE_DIR, f"watermarked_a{alpha_val}_{'tiles' if tiles else 'single'}.png")
    out_img.save(out)
    return out


def make_glare(intensity=0.55):
    """Sticker photographed with a diagonal specular glare band (shine test)."""
    img = _load(CLEAN)
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # diagonal band centered at 30% width, 70px wide, soft edges
    d = np.abs((xx + yy * 0.35) - w * 0.35)
    band = np.clip(1.0 - d / 90.0, 0, 1) ** 1.5
    glare_rgb = np.array(img, dtype=np.float32)
    glare_rgb += band[:, :, None] * 255 * intensity
    out_img = Image.fromarray(np.clip(glare_rgb, 0, 255).astype(np.uint8))
    out = os.path.join(FIXTURE_DIR, f"glare_{int(intensity*100)}.png")
    out_img.save(out)
    return out


def make_complex_bg():
    """Cut-out character placed on a busy patterned background (segmentation test)."""
    sticker = _load(CLEAN, 900)
    from rembg import remove, new_session
    s = new_session("isnet-anime")
    cut = remove(sticker, session=s)  # RGBA with alpha
    # busy background: diagonal stripes + circles
    bg = Image.new("RGB", (cut.width + 160, cut.height + 160), (240, 170, 200))
    d = ImageDraw.Draw(bg)
    for i in range(-bg.height, bg.width, 42):
        d.line([(i, 0), (i + bg.height, bg.height)], fill=(225, 120, 160), width=14)
    for cy in range(60, bg.height, 150):
        for cx in range(60, bg.width, 150):
            d.ellipse([cx - 26, cy - 26, cx + 26, cy + 26], outline=(255, 255, 255), width=5)
    bg.paste(cut, (80, 80), cut)
    out = os.path.join(FIXTURE_DIR, "complex_bg.png")
    bg.save(out)
    return out


def make_held_items():
    """Character holding distinct objects (tray) — held-item retention test.

    Uses the roller-maid sticker (trays with parfaits) as-is on plain white.
    """
    img = _load(COMPLEX)
    out = os.path.join(FIXTURE_DIR, "held_items.png")
    img.save(out)
    return out


def all_fixtures():
    return [
        make_rotated(6.0),
        make_rotated(-3.5),
        make_shadowed(4.0),
        make_watermarked(alpha_val=52, tiles=True),
        make_watermarked(alpha_val=90, tiles=False),
        make_glare(0.55),
        make_complex_bg(),
        make_held_items(),
    ]


if __name__ == "__main__":
    for f in all_fixtures():
        print("wrote", f)
