"""Sticker Pipeline Orchestrator
Coordinates the 4 decoupled systems into a unified, easy-to-use pipeline:
System 1: BackgroundSegmentor
System 2: CharacterExtractor
System 3: PerspectiveDeskewer
System 4: PostProcessor
"""

import time
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from PIL import Image
import numpy as np
import cv2

if __package__ is None or __package__ == "":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from deskew_pipeline.segmentor import BackgroundSegmentor, SegmentationResult
    from deskew_pipeline.character_extractor import CharacterExtractor, CharacterExtractionResult
    from deskew_pipeline.deskewer import PerspectiveDeskewer, DeskewResult
    from deskew_pipeline.postprocessor import PostProcessor, PostProcessResult
    from deskew_pipeline.border_generator import DiecutBorderGenerator
    from deskew_pipeline.enhancer import AIEnhancer
    from deskew_pipeline.pca_aligner import PCAOrientationAligner
    from deskew_pipeline.watermark_remover import WatermarkRemover, WatermarkRemovalResult
    from deskew_pipeline.shine_remover import ShineRemover, ShineRemovalResult
    from deskew_pipeline.color_enhancer import ColorEnhancer, ColorEnhanceResult
    from deskew_pipeline.mask_ops import apply_character_hint, apply_negative_mask
    from deskew_pipeline.alpha_refine import refine_alpha_edges
else:
    from .segmentor import BackgroundSegmentor, SegmentationResult
    from .character_extractor import CharacterExtractor, CharacterExtractionResult
    from .deskewer import PerspectiveDeskewer, DeskewResult
    from .postprocessor import PostProcessor, PostProcessResult
    from .border_generator import DiecutBorderGenerator
    from .enhancer import AIEnhancer
    from .pca_aligner import PCAOrientationAligner
    from .watermark_remover import WatermarkRemover, WatermarkRemovalResult
    from .shine_remover import ShineRemover, ShineRemovalResult
    from .color_enhancer import ColorEnhancer, ColorEnhanceResult
    from .mask_ops import apply_character_hint, apply_negative_mask
    from .alpha_refine import refine_alpha_edges



@dataclass
class PipelineResult:
    final_rgba: Image.Image
    raw_image: Image.Image
    segmented_sticker: Image.Image
    character_art: Image.Image
    deskewed_image: Image.Image
    metadata: Dict[str, Any]
    seg_result: Optional[SegmentationResult] = None
    watermark_cleaned: Optional[Image.Image] = None
    cel_restored: Optional[Image.Image] = None
    color_enhanced: Optional[Image.Image] = None

    def save(self, output_path: str):
        """Saves the final clean transparent PNG."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        self.final_rgba.save(output_path, "PNG")
        print(f"[StickerPipeline] Saved output to: {output_path}")

    def save_all_stages(self, output_dir: str, base_name: str = "result"):
        """Saves intermediate outputs of each system for inspection."""
        os.makedirs(output_dir, exist_ok=True)
        self.raw_image.save(os.path.join(output_dir, f"{base_name}_1_raw.png"))
        self.segmented_sticker.save(os.path.join(output_dir, f"{base_name}_2_segmented.png"))
        self.character_art.save(os.path.join(output_dir, f"{base_name}_3_character_art.png"))
        self.deskewed_image.save(os.path.join(output_dir, f"{base_name}_4_deskewed.png"))
        self.final_rgba.save(os.path.join(output_dir, f"{base_name}_5_final_clean.png"))
        print(f"[StickerPipeline] Saved all 5 stages to directory: {output_dir}")


class StickerPipeline:
    def __init__(
        self,
        segmentor_model: str = "isnet-anime",
        extract_mode: str = "character_only",
        deskew_mode: str = "auto",
        target_aspect_ratio: Optional[float] = None,
        padding_px: int = 15,
        defringe: bool = True,
        ensure_connected: bool = True,
        connectivity_mode: str = "bridge",
        bridge_thickness_px: int = 10,
        pca_align: bool = True,
        pca_align_mode: str = "bottom_upright",
        add_diecut_border: bool = False,
        border_thickness_px: int = 14,
        border_color: Tuple[int, int, int] = (255, 255, 255),
        border_mode: str = "solid",
        border_smoothing: int = 50,
        highlight_glow_radius: int = 20,
        highlight_box_padding: int = 16,
        enhance_resolution: bool = False,
        enhance_scale: int = 2,
        enhance_model: str = "ultrasharp_lite",
        preserve_colors: bool = True,
        manual_rotation_deg: float = 0.0,
        horizontal_skew_deg: float = 0.0,
        vertical_skew_deg: float = 0.0,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
        alpha_threshold: int = 15,
        clean_shadow_smudges: bool = True,
        smudge_sensitivity: int = 50,
        edge_inset_px: int = 0,
        clean_hair_gaps: bool = True,
        remove_watermarks: bool = False,
        watermark_sensitivity: int = 50,
        watermark_region: str = "corners_and_margins",
        remove_shine: bool = False,
        shine_strength: int = 60,
        color_tiers: int = 40,
        flat_cel_look: bool = False,
        color_pop_preset: str = "off",
        color_pop_vibrance: Optional[float] = None,
        color_pop_clarity: Optional[float] = None,
        character_bbox: Optional[Tuple[int, int, int, int]] = None,
        character_hint_mask: Optional[np.ndarray] = None,
        character_negative_mask: Optional[np.ndarray] = None,
        auto_fill_holes: bool = False,
        refine_edges: bool = True
    ):
        """
        Args:
            segmentor_model: 'isnet-anime', 'birefnet-general', 'u2net', or 'digital'
            extract_mode: 'character_only' (peels white die-cut margin), 'full_sticker'
            deskew_mode: 'auto' (detects quad vs die-cut), 'quad', 'moments', 'parametric', 'none'
            target_aspect_ratio: Optional aspect ratio constraint (e.g. 1.0 for square)
            padding_px: Padding around final cropped sticker
            defringe: If True, removes boundary color bleeding
            ensure_connected: Guarantees output is a single connected physical/digital sticker
            connectivity_mode: 'bridge' (bridges disconnected parts) or 'largest' (keeps main body)
            bridge_thickness_px: Thickness of connecting bridge between disjoint components
            pca_align: Runs PCA spatial analysis to ensure orientation is aligned with the bottom
            pca_align_mode: 'bottom_upright', 'bottom_baseline', 'auto', 'horizontal', 'snap_orthogonal', 'neutral'
            add_diecut_border: Adds a clean, rounded die-cut sticker border (stroke)
            border_thickness_px: Thickness of custom die-cut border
            border_color: RGB color of the die-cut border (default: white)
            enhance_resolution: Upscales output into higher-resolution using cutting-edge SOTA AI
            enhance_scale: Upscale factor (2 or 4)
            enhance_model: 'animesharp' (SOTA anime & illustration) or 'ultrasharp_lite' (fast high-detail)
            preserve_colors: 100% preserves original source artwork colors with zero drift
            manual_rotation_deg: User-specified manual rotation angle in degrees
            flip_horizontal: Mirror horizontally
            flip_vertical: Mirror vertically
            alpha_threshold: Minimum alpha cutoff to strip faint haze (0-255)
            clean_shadow_smudges: Removes drop-shadow smudges outside sticker border
            smudge_sensitivity: Aggressiveness of smudge/shadow detector (0-100)
            edge_inset_px: Optional pixel shave from outer edge
            clean_hair_gaps: Clean background and shadow trapped inside hair loops and limb gaps
            remove_watermarks: Pre-processing stage to remove corner stamps/URLs first
            watermark_sensitivity: Watermark detection aggressiveness (10-100)
            watermark_region: 'corners_and_margins' or 'full'
            remove_shine: Pre-processing stage to remove plastic/specular glare and restore anime cel colors
            shine_strength: Cel flattening and glare suppression strength (10-100)
            color_tiers: Number of discrete anime cel tonal tiers for ultra-detailed original source file fidelity (default 40, range 16-64)
            flat_cel_look: Enables discrete posterized anime cel tiers (off by default)
            color_pop_preset: Float32 OKLab color enhancer preset ('off', 'natural', 'anime_pop', 'photo_sticker')
            color_pop_vibrance: Vibrance multiplier for Color Pop
            color_pop_clarity: Local contrast clarity for Color Pop
            character_bbox: Optional (x, y, w, h) bounding box to focus segmentation strictly on character
            character_hint_mask: Optional 2D mask painted over the character; isolates only highlighted artwork
            auto_fill_holes: Automatically detects and fills enclosed holes/voids using AnimeLaMa
        """
        self.deskew_mode = deskew_mode
        self.target_aspect_ratio = target_aspect_ratio
        self.pca_align = pca_align
        self.pca_align_mode = pca_align_mode
        self.add_diecut_border = add_diecut_border
        self.border_smoothing = border_smoothing
        self.enhance_resolution = enhance_resolution
        self.enhance_scale = enhance_scale
        self.enhance_model = enhance_model
        self.preserve_colors = preserve_colors
        self.manual_rotation_deg = manual_rotation_deg
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.alpha_threshold = alpha_threshold
        self.clean_shadow_smudges = clean_shadow_smudges
        self.smudge_sensitivity = smudge_sensitivity
        self.edge_inset_px = edge_inset_px
        self.clean_hair_gaps = clean_hair_gaps
        self.character_bbox = character_bbox
        self.character_hint_mask = character_hint_mask
        self.character_negative_mask = character_negative_mask
        self.auto_fill_holes = auto_fill_holes
        self.refine_edges = refine_edges

        self.segmentor = BackgroundSegmentor(model_name=segmentor_model)
        self.character_extractor = CharacterExtractor(
            mode=extract_mode,
            ensure_connected=ensure_connected,
            connectivity_mode=connectivity_mode,
            bridge_thickness_px=bridge_thickness_px,
            clean_hair_gaps=clean_hair_gaps,
            clean_shadows=clean_shadow_smudges
        )
        self.deskewer = PerspectiveDeskewer()
        self.postprocessor = PostProcessor(
            padding_px=padding_px,
            defringe=defringe
        )
        self.border_generator = DiecutBorderGenerator(
            thickness_px=border_thickness_px,
            color=border_color,
            mode=border_mode,
            die_cut_smoothing=border_smoothing,
            glow_radius=highlight_glow_radius,
            box_padding=highlight_box_padding
        )
        self.enhancer = AIEnhancer(
            scale=enhance_scale,
            model_name=enhance_model,
            preserve_colors=preserve_colors
        )
        self.pca_aligner = PCAOrientationAligner(mode=pca_align_mode)
        self.watermark_remover = WatermarkRemover()
        self.shine_remover = ShineRemover()
        self.color_enhancer = ColorEnhancer()

        self.remove_watermarks = remove_watermarks
        self.watermark_sensitivity = watermark_sensitivity
        self.watermark_region = watermark_region
        self.remove_shine = remove_shine
        self.shine_strength = shine_strength
        self.color_tiers = color_tiers
        self.flat_cel_look = flat_cel_look
        self.color_pop_preset = color_pop_preset
        self.color_pop_vibrance = color_pop_vibrance
        self.color_pop_clarity = color_pop_clarity
        self.horizontal_skew_deg = float(horizontal_skew_deg)
        self.vertical_skew_deg = float(vertical_skew_deg)

    def process(
        self,
        image_input,
        parametric_pitch: float = 0.0,
        parametric_yaw: float = 0.0,
        parametric_roll: float = 0.0,
        manual_rotation: float = 0.0,
        horizontal_skew: Optional[float] = None,
        vertical_skew: Optional[float] = None,
        precomputed_segmentation: Optional[SegmentationResult] = None,
        character_bbox: Optional[Tuple[int, int, int, int]] = None,
        character_hint_mask: Optional[np.ndarray] = None,
        character_negative_mask: Optional[np.ndarray] = None,
        progress_callback = None
    ) -> PipelineResult:
        """Processes an image through all systems."""
        t_start = time.time()
        if progress_callback:
            progress_callback(10, "Initializing pipeline & reading image...")
        active_bbox = character_bbox if character_bbox is not None else self.character_bbox
        active_hint_mask = character_hint_mask if character_hint_mask is not None else self.character_hint_mask
        active_neg_mask = character_negative_mask if character_negative_mask is not None else self.character_negative_mask

        # Load image
        if isinstance(image_input, str):
            raw_img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            raw_img = image_input
        else:
            raise TypeError("Expected image path string or PIL Image object.")

        # Existing transparency: if the input is already a cutout (alpha channel present with
        # real transparency), composite over WHITE instead of black. RGBA->RGB via convert()
        # drops alpha and composites over BLACK, which rembg then treats as background and
        # hallucinates fills around the character (measured: opaque area 67.8% of canvas vs
        # 28.7% input, with heavy fill artifacts).
        pre_composited = False
        if raw_img.mode == "RGBA":
            alpha_arr = np.array(raw_img)[:, :, 3]
            if (alpha_arr == 0).any() and (alpha_arr == 255).any():
                # genuine cutout: composite over white, mark as pre-segmented
                bg = Image.new("RGBA", raw_img.size, (255, 255, 255, 255))
                raw_img = Image.alpha_composite(bg, raw_img.convert("RGBA")).convert("RGB")
                pre_composited = True
            else:
                raw_img = raw_img.convert("RGB")
        elif raw_img.mode != "RGB":
            raw_img = raw_img.convert("RGB")

        # Pre-Processing Stage 1: Watermark Remover (executes on raw canvas BEFORE segmentation)
        t_wm = 0.0
        t_shine = 0.0
        watermark_cleaned_img = None
        cel_restored_img = None
        wm_detected = False
        shine_detected = False
        color_enhanced_img = None
        color_pop_applied = False
        color_pop_meta = None
        t_color_pop = 0.0
        current_prep_img = raw_img

        if self.remove_watermarks:
            wm_res = self.watermark_remover.remove(
                current_prep_img,
                sensitivity=self.watermark_sensitivity,
                region=self.watermark_region
            )
            current_prep_img = wm_res.cleaned_image
            watermark_cleaned_img = wm_res.cleaned_image
            t_wm = wm_res.execution_time_s
            wm_detected = wm_res.watermark_detected

        # System 1: Background & Foreground Segmentor (receives clean pre-processed image)
        if progress_callback:
            progress_callback(25, "System 1/5: AI Background & Foreground Segmentation...")
        if precomputed_segmentation is not None and not self.remove_watermarks:
            seg_res = precomputed_segmentation
            t_seg = 0.0
            if active_hint_mask is not None or active_neg_mask is not None:
                mod_mask = seg_res.mask.copy()
                if active_hint_mask is not None:
                    mod_mask = apply_character_hint(mod_mask, active_hint_mask)
                if active_neg_mask is not None:
                    mod_mask = apply_negative_mask(mod_mask, active_neg_mask)
                mod_rgba = np.array(seg_res.rgba)
                mod_rgba[:, :, 3] = mod_mask
                seg_res = SegmentationResult(
                    rgba=Image.fromarray(mod_rgba, "RGBA"),
                    mask=mod_mask,
                    bbox=seg_res.bbox,
                    confidence=seg_res.confidence,
                    features=seg_res.features
                )
        else:
            t0 = time.time()
            seg_res = self.segmentor.segment(
                current_prep_img,
                alpha_threshold=self.alpha_threshold,
                clean_shadow_smudges=self.clean_shadow_smudges,
                smudge_sensitivity=self.smudge_sensitivity,
                edge_inset_px=self.edge_inset_px,
                character_bbox=active_bbox,
                character_hint_mask=active_hint_mask,
                character_negative_mask=active_neg_mask
            )
            t_seg = time.time() - t0

        # System 2: Character / Artwork Extractor (passes raw_img for pristine background sampling)
        if progress_callback:
            progress_callback(48, "System 2/5: Separating hair gaps & extracting character...")
        t0 = time.time()
        char_res = self.character_extractor.extract(
            seg_res.rgba,
            seg_res.mask,
            raw_image=raw_img,
            character_bbox=active_bbox,
            character_hint_mask=active_hint_mask,
            character_negative_mask=active_neg_mask
        )
        # Snap the soft matte to the artwork's colour edges (narrow band around the outline only)
        if self.refine_edges and char_res.character_mask is not None:
            refined_mask = refine_alpha_edges(np.array(raw_img), char_res.character_mask)
            char_rgba = np.array(char_res.rgba)
            char_rgba[:, :, 3] = refined_mask
            char_res.rgba = Image.fromarray(char_rgba, "RGBA")
            char_res.character_mask = refined_mask
        t_char = time.time() - t0

        # System 3: 3D Perspective Plane & Affine Deskewer
        if progress_callback:
            progress_callback(62, "System 3/5: Leveling & deskewing character artwork...")
        t0 = time.time()
        deskew_input_rgba = char_res.rgba
        deskew_mask = char_res.character_mask

        deskew_mode = self.deskew_mode
        if deskew_mode == "auto":
            # 1. Check if the object is a true quadrilateral card/square/rectangle FIRST
            # (Cards, square stickers, rectangular photos need 4-corner perspective rectification)
            is_quad_poly = False
            contours, _ = cv2.findContours(deskew_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                peri = cv2.arcLength(largest, True)
                # True quadrilateral cards/squares match 4 vertices under strict epsilon <= 0.025
                poly = cv2.approxPolyDP(largest, 0.025 * peri, True)
                hull = cv2.convexHull(largest)
                hull_area = max(1.0, cv2.contourArea(hull))
                solidity = cv2.contourArea(largest) / hull_area
                if len(poly) == 4 and solidity > 0.90:
                    is_quad_poly = True

            if is_quad_poly:
                deskew_mode = "quad"
            else:
                # 2. If not a quadrilateral, check for a dominant straight horizontal baseline cut
                # (e.g. peeker sticker). Gate on flat aspect: tall organic characters produce
                # phantom 'hem' baselines (measured 10-11 deg on upright art) — never deskew those.
                raw_rect = cv2.minAreaRect(max(contours, key=cv2.contourArea)) if contours else ((0, 0), (1, 1), 0)
                (_, _), (prw, prh), _ = raw_rect
                flat_aspect = max(prw, prh) / max(1.0, min(prw, prh))
                baseline_angle = (
                    self.deskewer.detect_flat_baseline_angle(deskew_mask)
                    if flat_aspect >= 2.5
                    else None
                )
                if baseline_angle is not None and abs(baseline_angle) >= 0.3:
                    deskew_mode = "baseline"
                else:
                    # 3. Organic / character artwork: detect physical skew from cut edges, minAreaRect, or facial features
                    char_skew = self.deskewer.detect_character_skew_angle(deskew_mask)
                    if char_skew is not None and abs(char_skew) >= 0.35:
                        deskew_mode = "character"
                    else:
                        deskew_mode = "none"

        if deskew_mode == "quad":
            deskew_res = self.deskewer.deskew_quadrilateral(
                deskew_input_rgba,
                deskew_mask,
                target_aspect_ratio=self.target_aspect_ratio
            )
        elif deskew_mode == "baseline":
            deskew_res = self.deskewer.deskew_flat_baseline(
                deskew_input_rgba,
                deskew_mask
            )
        elif deskew_mode == "character":
            deskew_res = self.deskewer.deskew_character_artwork(
                deskew_input_rgba,
                deskew_mask
            )
        elif deskew_mode == "parametric":
            deskew_res = self.deskewer.deskew_parametric_3d(
                deskew_input_rgba,
                pitch_deg=parametric_pitch,
                yaw_deg=parametric_yaw,
                roll_deg=parametric_roll
            )
        elif deskew_mode == "moments":
            deskew_res = self.deskewer.deskew_diecut_moments(
                deskew_input_rgba,
                deskew_mask
            )
        else:
            deskew_res = DeskewResult(rgba=deskew_input_rgba, homography=np.eye(3))
        t_deskew = time.time() - t0

        # Orientation Alignment (PCA)
        # Note: Do not double-rotate when moments, baseline, or character deskew is already used
        t0 = time.time()
        pca_meta = None
        deskewed_output = deskew_res.rgba
        if self.pca_align and self.pca_align_mode != "neutral" and deskew_mode not in ("moments", "baseline", "character"):
            deskewed_output, pca_meta = self.pca_aligner.align(deskewed_output)
        t_pca = time.time() - t0

        # Manual Rotation & Flips
        applied_rotation = manual_rotation if manual_rotation != 0.0 else self.manual_rotation_deg
        if applied_rotation != 0.0:
            deskewed_output = deskewed_output.rotate(-applied_rotation, expand=True, resample=Image.BICUBIC)

        # Skew / Shear (Horizontal & Vertical)
        applied_h_skew = horizontal_skew if horizontal_skew is not None else self.horizontal_skew_deg
        applied_v_skew = vertical_skew if vertical_skew is not None else self.vertical_skew_deg
        if applied_h_skew != 0.0 or applied_v_skew != 0.0:
            deskewed_output = self.deskewer.apply_skew(deskewed_output, applied_h_skew, applied_v_skew)

        if self.flip_horizontal:
            deskewed_output = deskewed_output.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_vertical:
            deskewed_output = deskewed_output.transpose(Image.FLIP_TOP_BOTTOM)

        # Character Art Color Restoration: Anime De-Shine & Cel Color Restorer
        # Restores authentic, ultra-detailed original 2D anime digital source cel colors
        # on the deskewed character artwork BEFORE final edge finishing & border packaging
        current_art = deskewed_output
        if self.remove_shine:
            shine_res = self.shine_remover.remove_shine(
                current_art,
                strength=self.shine_strength,
                color_tiers=self.color_tiers,
                flat_cel_look=self.flat_cel_look
            )
            current_art = shine_res.cleaned_image
            cel_restored_img = shine_res.cleaned_image
            t_shine = shine_res.execution_time_s
            shine_detected = shine_res.shine_detected

        # System 4: Edge Finishing & Asset Packaging
        # Defringes cel colors, applies smooth 4x SSAA sub-pixel antialiasing,
        # guarantees zero-background RGB, and tightly crops with user padding.
        t0 = time.time()
        post_res = self.postprocessor.process(current_art)
        current_rgba = post_res.rgba
        t_post = time.time() - t0

        # Optional: Auto-Fill Internal Holes & Accidental Cutout Voids via AnimeLaMa
        if self.auto_fill_holes:
            try:
                from .inpainter import BigLamaInpainter
                inpainter = BigLamaInpainter(model_type="anime")
                current_rgba, _ = inpainter.auto_fill_holes(current_rgba, raw_image=raw_img, filter_bg_holes=True)
            except Exception as e:
                print(f"[StickerPipeline] Warning: Auto-fill holes skipped: {e}")

        # Optional: AI Super-Resolution Enhancement
        # Applied to character artwork BEFORE color grading and border generation so neural
        # upscaler reconstructs clean high-res textures, edges, and lineart.
        if self.enhance_resolution:
            if progress_callback:
                progress_callback(75, "System 4/5: AI Super-Resolution neural upscaling...")
            current_rgba = self.enhancer.enhance(current_rgba, progress_callback=progress_callback)

        # Character Art Color Enhancement: Float32 OKLab Color Pop & Real AI LoRA
        # Executed AFTER AI super-resolution upscaling (and hole healing/defringing)
        # so all color profiles, hue-locked vibrance, deep inky shadow contrast,
        # and local clarity are applied directly to the high-resolution neural upscaled artwork.
        t0 = time.time()
        if self.color_pop_preset != "off" or self.color_pop_vibrance is not None or self.color_pop_clarity is not None:
            if progress_callback:
                progress_callback(85, "System 4b: Applying Float32 OKLab color profile & LoRA finishing...")
            enh_res = self.color_enhancer.enhance(
                current_rgba,
                preset=self.color_pop_preset,
                vibrance=self.color_pop_vibrance if self.color_pop_vibrance is not None else 1.0,
                clarity=self.color_pop_clarity if self.color_pop_clarity is not None else 0.0
            )
            current_rgba = enh_res.enhanced_image
            color_enhanced_img = enh_res.enhanced_image
            color_pop_applied = True
            color_pop_meta = enh_res.metadata
        t_color_pop = time.time() - t0

        # FINAL STAGE: Add Crisp, Ultra-Smooth Die-Cut Sticker Border
        # Always performed LAST on the finalized, upscaled, hole-healed artwork
        # so the border is 100% silky-smooth, vector-sharp, and uncompressed.
        if self.add_diecut_border:
            if progress_callback:
                progress_callback(90, "System 5/5: Generating ultra-smooth die-cut vinyl border...")
            scale_factor = self.enhance_scale if self.enhance_resolution else 1
            if scale_factor > 1:
                scaled_border_generator = DiecutBorderGenerator(
                    thickness_px=int(round(self.border_generator.thickness_px * scale_factor)),
                    color=self.border_generator.color,
                    smoothing_radius=self.border_generator.smoothing_radius * 1.2,
                    die_cut_smoothing=self.border_generator.die_cut_smoothing,
                    mode=self.border_generator.mode,
                    glow_radius=int(round(self.border_generator.glow_radius * scale_factor)),
                    box_padding=int(round(self.border_generator.box_padding * scale_factor)),
                    box_corner_radius=int(round(self.border_generator.box_corner_radius * scale_factor))
                )
                current_rgba = scaled_border_generator.add_border(current_rgba)
            else:
                current_rgba = self.border_generator.add_border(current_rgba)

            # Cleanly crop transparent padding around the final border
            pad_px = int(round(self.postprocessor.padding_px * scale_factor))
            cropped_np, _ = self.postprocessor._crop_with_padding(np.array(current_rgba), padding=pad_px)
            current_rgba = Image.fromarray(cropped_np, "RGBA")

        if progress_callback:
            progress_callback(100, "Done! High-resolution sticker asset finalized.")

        t_total = time.time() - t_start

        metadata = {
            "execution_time_total_s": round(t_total, 3),
            "stage_times_s": {
                "preprocess_watermark": round(t_wm, 4),
                "preprocess_shine_remover": round(t_shine, 4),
                "color_pop_enhancer": round(t_color_pop, 4),
                "system1_segmentation": round(t_seg, 3),
                "system2_character_extraction": round(t_char, 3),
                "system3_deskew": round(t_deskew, 3),
                "pca_alignment": round(t_pca, 3),
                "system4_postprocess": round(t_post, 3)
            },
            "watermark_removed": self.remove_watermarks,
            "watermark_detected": wm_detected,
            "shine_removed": self.remove_shine,
            "shine_detected": shine_detected,
            "color_pop_applied": color_pop_applied,
            "color_pop_preset": self.color_pop_preset,
            "color_pop_metadata": color_pop_meta,
            "clean_hair_gaps": self.clean_hair_gaps,
            "selected_deskew_mode": deskew_mode,
            "pca_align_enabled": self.pca_align,
            "pca_alignment_metadata": pca_meta,
            "diecut_border_added": self.add_diecut_border,
            "ai_enhanced": self.enhance_resolution,
            "enhance_model": self.enhance_model if self.enhance_resolution else None,
            "preserve_colors": self.preserve_colors if self.enhance_resolution else None,
            "character_bbox": active_bbox,
            "homography_matrix": deskew_res.homography.tolist(),
            "detected_corners": deskew_res.detected_corners.tolist() if deskew_res.detected_corners is not None else None,
            "detected_angles": deskew_res.detected_angles,
            "final_size": current_rgba.size,
            "original_size": post_res.original_size
        }

        return PipelineResult(
            final_rgba=current_rgba,
            raw_image=raw_img,
            segmented_sticker=seg_res.rgba,
            character_art=char_res.rgba,
            deskewed_image=deskewed_output,
            metadata=metadata,
            seg_result=seg_res,
            watermark_cleaned=watermark_cleaned_img,
            cel_restored=cel_restored_img,
            color_enhanced=color_enhanced_img
        )

