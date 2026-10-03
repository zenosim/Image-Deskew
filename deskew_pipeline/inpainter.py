"""Dual-Engine Inpainting: AnimeLaMa (Anime Specialist) + Big-LaMa ONNX (General).

AnimeLaMa (default): Fine-tuned on anime/manga datasets by df1412. Excels at reconstructing
anime eyes, eyelashes, hair strands, lineart, and cel shading. Uses TorchScript (.pt).

Big-LaMa ONNX (fallback): General-purpose inpainting trained on Places2 natural photos.
Good for photographic content but produces blurry smudges on anime features.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Union, Tuple

# Prevent PyTorch thread oversubscription which can cause pseudo-deadlocks
# when running alongside the HTTP server thread pool
try:
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
except Exception:
    pass


class BigLamaInpainter:
    """High-fidelity inpainting engine with anime-specialized and general modes.
    
    Features:
    - Dual engine: AnimeLaMa (TorchScript, anime specialist) or Big-LaMa ONNX (general).
    - Automatic download / caching from Hugging Face Hub.
    - Contextual bounding-box cropping with generous padding for 100% resolution retention.
    - Tiled high-resolution inpainting for large masked areas.
    - Anti-aliasing edge dilation to eliminate obstruction fringes.
    - Seamless Gaussian feathering into the original canvas.
    - Full transparency (RGBA) support.
    """
    
    # Anime-specialist model (default)
    ANIME_REPO = "df1412/anime-big-lama"
    ANIME_FILENAME = "anime-manga-big-lama.pt"
    
    # General ONNX model (fallback)
    ONNX_REPO = "Carve/LaMa-ONNX"
    ONNX_FILENAME = "lama_fp32.onnx"
    
    # Singleton caches
    _anime_model = None
    _anime_model_path = None
    _onnx_session = None
    _onnx_model_path = None

    def __init__(self, model_type: str = "anime", model_path: Optional[str] = None):
        """Initialize the inpainter.
        
        Args:
            model_type: 'anime' (default, AnimeLaMa specialist) or 'general' (Big-LaMa ONNX).
            model_path: Optional direct path to model file. If None, auto-downloads.
        """
        self.model_type = model_type
        if model_path is not None:
            if model_type == "anime":
                BigLamaInpainter._anime_model_path = model_path
            else:
                BigLamaInpainter._onnx_model_path = model_path
        self._ensure_engine()

    def _ensure_engine(self):
        """Load the appropriate engine (lazy singleton)."""
        if self.model_type == "anime":
            self._ensure_anime_engine()
        else:
            self._ensure_onnx_engine()

    @classmethod
    def _ensure_anime_engine(cls):
        """Load AnimeLaMa TorchScript model (singleton)."""
        if cls._anime_model is not None:
            return
        
        import torch
        from huggingface_hub import hf_hub_download
        
        if cls._anime_model_path is None or not os.path.exists(cls._anime_model_path):
            cls._anime_model_path = hf_hub_download(
                repo_id=cls.ANIME_REPO,
                filename=cls.ANIME_FILENAME
            )
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cls._anime_device = device
        cls._anime_model = torch.jit.load(cls._anime_model_path, map_location=device)
        cls._anime_model.eval()

    @classmethod
    def _ensure_onnx_engine(cls):
        """Load Big-LaMa ONNX model (singleton)."""
        if cls._onnx_session is not None:
            return

        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        if cls._onnx_model_path is None or not os.path.exists(cls._onnx_model_path):
            cls._onnx_model_path = hf_hub_download(
                repo_id=cls.ONNX_REPO,
                filename=cls.ONNX_FILENAME
            )

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = min(os.cpu_count() or 4, 8)

        available = ort.get_available_providers()
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")
        if "DmlExecutionProvider" in available:
            providers.insert(0, "DmlExecutionProvider")

        cls._onnx_session = ort.InferenceSession(
            cls._onnx_model_path,
            sess_options=sess_options,
            providers=providers
        )

    def inpaint(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Union[Image.Image, np.ndarray],
        dilate_px: int = 8,
        context_padding_ratio: float = 0.6,
        min_context_px: int = 96
    ) -> Image.Image:
        """Inpaint regions specified by the mask.
        
        Args:
            image: Input image (PIL Image or RGB/RGBA numpy array).
            mask: Binary mask where white (255 or > 0) is the area to fill,
                  and black (0) is preserved.
            dilate_px: Pixels to dilate the mask to eliminate fringe bleed.
            context_padding_ratio: Context padding as a ratio of the mask bounding box.
            min_context_px: Minimum pixel padding around the mask bounding box.
            
        Returns:
            Inpainted PIL Image (preserving original mode RGB or RGBA).
        """
        self._ensure_engine()

        # Convert image to numpy
        orig_is_pil = isinstance(image, Image.Image)
        orig_mode = image.mode if orig_is_pil else ("RGBA" if image.shape[2] == 4 else "RGB")

        if orig_is_pil:
            img_arr = np.array(image)
        else:
            img_arr = image.copy()

        has_alpha = (img_arr.ndim == 3 and img_arr.shape[2] == 4)
        if has_alpha:
            orig_alpha = img_arr[:, :, 3].copy()
            rgb_arr = img_arr[:, :, :3].copy()
        else:
            orig_alpha = None
            rgb_arr = img_arr.copy() if img_arr.ndim == 3 else cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)

        # Convert mask to single-channel uint8 binary
        if isinstance(mask, Image.Image):
            mask_arr = np.array(mask)
        else:
            mask_arr = mask.copy()

        if mask_arr.ndim == 3:
            if mask_arr.shape[2] == 4:
                mask_bin = np.where((mask_arr[:, :, 3] > 30) & (mask_arr[:, :, :3].max(axis=2) > 30), 255, 0).astype(np.uint8)
            else:
                mask_bin = np.where(mask_arr.max(axis=2) > 30, 255, 0).astype(np.uint8)
        else:
            mask_bin = np.where(mask_arr > 30, 255, 0).astype(np.uint8)

        h, w = rgb_arr.shape[:2]
        if mask_bin.shape[:2] != (h, w):
            mask_bin = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)

        # Pre-process RGB boundary colors on transparent voids to prevent dark fringe bleed
        if has_alpha and np.any(orig_alpha < 30):
            transparent_mask = (orig_alpha < 30).astype(np.uint8)
            if np.any(transparent_mask & (mask_bin > 0)):
                try:
                    rgb_arr = cv2.inpaint(rgb_arr, transparent_mask, 5, cv2.INPAINT_TELEA)
                except Exception:
                    pass

        # Quick return if mask is completely empty
        if not np.any(mask_bin):
            return image if orig_is_pil else Image.fromarray(image)

        # 1. Anti-aliasing mask dilation to capture edge fringe colors
        if dilate_px > 0:
            kernel_size = dilate_px * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            mask_dilated = cv2.dilate(mask_bin, kernel)
        else:
            mask_dilated = mask_bin.copy()

        h, w = rgb_arr.shape[:2]

        # 2. Contextual Bounding Box Calculation
        y_indices, x_indices = np.where(mask_dilated > 0)
        min_x, max_x = int(x_indices.min()), int(x_indices.max())
        min_y, max_y = int(y_indices.min()), int(y_indices.max())

        bw = max_x - min_x + 1
        bh = max_y - min_y + 1

        # Check if mask covers virtually the entire image
        if bw > 0.85 * w and bh > 0.85 * h:
            # Inpaint full image scaled to 512x512
            inpainted_rgb = self._run_model_512(rgb_arr, mask_dilated)
            # Seamless blend
            blend_weight = cv2.GaussianBlur(mask_dilated.astype(np.float32) / 255.0, (9, 9), 3.0)[:, :, None]
            final_rgb = (inpainted_rgb.astype(np.float32) * blend_weight +
                         rgb_arr.astype(np.float32) * (1.0 - blend_weight))
            final_rgb = np.clip(final_rgb, 0, 255).astype(np.uint8)
        else:
            # Check if region is large enough to benefit from tiled approach
            # Threshold 700px prevents over-tiling on typical brush strokes.
            # For masks 400-700px, contextual crop with extra padding works fine.
            max_side = max(bw, bh)
            if max_side > 700:
                # Tiled high-res inpainting for very large masked areas
                final_rgb = self._tiled_inpaint(rgb_arr, mask_dilated, dilate_px)
            else:
                # Standard contextual crop approach for small/medium masks
                final_rgb = self._contextual_crop_inpaint(
                    rgb_arr, mask_dilated, min_x, max_x, min_y, max_y,
                    bw, bh, context_padding_ratio, min_context_px, dilate_px
                )

        # Reconstruct output image and restore alpha on filled voids
        if has_alpha:
            new_alpha = orig_alpha.copy()
            # If inpainting covered transparent voids/holes, restore alpha to 255 (opaque)
            # with smooth antialiased blend at the mask boundary
            void_pixels = (orig_alpha < 30) & (mask_dilated > 0)
            if np.any(void_pixels):
                blur_mask = cv2.GaussianBlur(mask_dilated.astype(np.float32) / 255.0, (5, 5), 1.0)
                blended_alpha = np.clip(orig_alpha.astype(np.float32) + blur_mask * (255.0 - orig_alpha.astype(np.float32)), 0, 255).astype(np.uint8)
                new_alpha[mask_dilated > 0] = blended_alpha[mask_dilated > 0]
                new_alpha[mask_dilated > 40] = 255

            final_rgba = np.dstack((final_rgb, new_alpha))
            out_img = Image.fromarray(final_rgba, mode="RGBA")
        else:
            out_img = Image.fromarray(final_rgb, mode="RGB")

        return out_img

    @staticmethod
    def detect_internal_holes(
        image: Union[Image.Image, np.ndarray],
        min_area: int = 8,
        max_area: int = 60000,
        raw_image: Optional[Union[Image.Image, np.ndarray]] = None,
        filter_bg_holes: bool = True,
        **kwargs
    ) -> Optional[np.ndarray]:
        """
        Detects enclosed internal background holes or voids inside the character/sticker alpha mask
        (e.g., accidental cutout holes in hair, clothing, or limbs caused by background removal).
        If raw_image is provided and filter_bg_holes is True, cavities whose original pixels were
        background-colored (such as intentional hair gaps or negative space) are protected and not filled.
        Returns a binary uint8 mask of the detected internal holes (255 = hole, 0 = keep).
        """
        if "min_hole_area" in kwargs:
            min_area = kwargs["min_hole_area"]
        if "max_hole_area" in kwargs:
            max_area = kwargs["max_hole_area"]
        if isinstance(image, Image.Image):
            np_img = np.array(image)
        else:
            np_img = image

        if np_img.ndim != 3 or np_img.shape[2] != 4:
            return None

        alpha = np_img[:, :, 3]
        if not np.any(alpha > 30):
            return None

        h, w = alpha.shape
        bin_alpha = (alpha > 40).astype(np.uint8) * 255

        # Background color check if raw_image is provided
        is_raw_bg = None
        if filter_bg_holes and raw_image is not None:
            if isinstance(raw_image, Image.Image):
                raw_np = np.array(raw_image.convert("RGB"))
            else:
                raw_np = raw_image[:, :, :3] if raw_image.ndim == 3 else raw_image
            if raw_np.shape[:2] == (h, w):
                border_px = np.vstack([
                    raw_np[:12, :].reshape(-1, 3),
                    raw_np[-12:, :].reshape(-1, 3),
                    raw_np[:, :12].reshape(-1, 3),
                    raw_np[:, -12:].reshape(-1, 3)
                ])
                bg_median = np.median(border_px, axis=0)
                diff = np.linalg.norm(raw_np.astype(float) - bg_median.astype(float), axis=2)
                gray = cv2.cvtColor(raw_np, cv2.COLOR_RGB2GRAY)
                hsv = cv2.cvtColor(raw_np, cv2.COLOR_RGB2HSV)
                sat = hsv[:, :, 1]
                h_deg = hsv[:, :, 0] * 2

                # Background-like pixels: canvas background color, drop shadow, or light watermark text
                is_bg_color = (diff < 55.0)
                is_shadow = (sat < 35) & (gray > 60)
                is_skin = ((h_deg <= 30) | (h_deg >= 340)) & (sat >= 15) & (sat <= 85) & (gray >= 115)
                is_raw_bg = (is_bg_color | is_shadow) & (~is_skin)

        # 2-level contour hierarchy: RETR_CCOMP
        # hierarchy: [Next, Previous, First_Child, Parent]
        # Internal holes have Parent != -1
        contours, hierarchy = cv2.findContours(bin_alpha, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if contours is None or hierarchy is None or len(contours) == 0:
            return None

        hierarchy = hierarchy[0]
        hole_mask = np.zeros((h, w), dtype=np.uint8)
        found_holes = 0

        for i, cnt in enumerate(contours):
            parent_idx = hierarchy[i][3]
            if parent_idx != -1:
                area = cv2.contourArea(cnt)
                if min_area <= area <= max_area:
                    # Check if this hole was originally background/shadow in raw image (hair gap)
                    if is_raw_bg is not None:
                        temp_cnt_mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.drawContours(temp_cnt_mask, [cnt], -1, 255, -1)
                        pts_in_hole = temp_cnt_mask > 0
                        if np.any(pts_in_hole):
                            bg_share = np.mean(is_raw_bg[pts_in_hole])
                            if bg_share > 0.30:
                                # Legitimate hair gap / background cavity / drop shadow, do NOT fill!
                                continue

                    cv2.drawContours(hole_mask, [cnt], -1, 255, -1)
                    found_holes += 1

        if found_holes == 0 or not np.any(hole_mask):
            return None

        return hole_mask

    def auto_fill_holes(
        self,
        image: Union[Image.Image, np.ndarray],
        min_area: int = 8,
        max_area: int = 60000,
        dilate_px: int = 4,
        raw_image: Optional[Union[Image.Image, np.ndarray]] = None,
        filter_bg_holes: bool = True
    ) -> Tuple[Image.Image, int]:
        """
        Automatically detects all enclosed cutout voids/holes inside the character and uses
        AnimeLaMa to synthesize the missing texture and restore opacity.
        Returns (healed_image, count_of_holes).
        """
        hole_mask = self.detect_internal_holes(
            image, min_area=min_area, max_area=max_area, raw_image=raw_image, filter_bg_holes=filter_bg_holes
        )
        if hole_mask is None or not np.any(hole_mask):
            orig_img = image if isinstance(image, Image.Image) else Image.fromarray(image)
            return orig_img, 0

        num_labels, _ = cv2.connectedComponents(hole_mask)
        hole_count = max(0, num_labels - 1)

        healed = self.inpaint(image, hole_mask, dilate_px=dilate_px)
        return healed, hole_count

    def _contextual_crop_inpaint(
        self, rgb_arr, mask_dilated, min_x, max_x, min_y, max_y,
        bw, bh, context_padding_ratio, min_context_px, dilate_px
    ):
        """Standard contextual crop inpainting for small/medium masked regions."""
        h, w = rgb_arr.shape[:2]
        pad = max(int(max(bw, bh) * context_padding_ratio), min_context_px)
        cx = (min_x + max_x) // 2
        cy = (min_y + max_y) // 2
        side = max(bw, bh) + 2 * pad

        # Compute crop bounds clamped to image
        x0 = max(0, cx - side // 2)
        y0 = max(0, cy - side // 2)
        x1 = min(w, x0 + side)
        y1 = min(h, y0 + side)

        # Re-adjust if hitting borders to keep roughly square context
        if (x1 - x0) < side and x0 > 0:
            x0 = max(0, x1 - side)
        if (y1 - y0) < side and y0 > 0:
            y0 = max(0, y1 - side)

        crop_rgb = rgb_arr[y0:y1, x0:x1]
        crop_mask = mask_dilated[y0:y1, x0:x1]

        crop_h, crop_w = crop_rgb.shape[:2]

        # Inpaint the contextual crop
        inpainted_crop = self._run_model_512(crop_rgb, crop_mask)
        if inpainted_crop.shape[:2] != (crop_h, crop_w):
            inpainted_crop = cv2.resize(inpainted_crop, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)

        # Feather blending only within the crop area
        blur_size = max(5, int(dilate_px) * 2 + 1)
        if blur_size % 2 == 0:
            blur_size += 1
        crop_blend_weight = cv2.GaussianBlur(
            crop_mask.astype(np.float32) / 255.0,
            (blur_size, blur_size),
            max(2.0, dilate_px * 0.4)
        )[:, :, None]

        # Cut off infinitesimal Gaussian tails (< 0.005) so surrounding pixels are untouched
        pure_orig = (crop_blend_weight < 0.005)
        crop_blend_weight[pure_orig] = 0.0

        blended_crop = (inpainted_crop.astype(np.float32) * crop_blend_weight +
                        crop_rgb.astype(np.float32) * (1.0 - crop_blend_weight))
        blended_crop = np.clip(np.round(blended_crop), 0, 255).astype(np.uint8)

        # Strictly restore exact original byte values on untouched pixels
        blended_crop[pure_orig.squeeze()] = crop_rgb[pure_orig.squeeze()]

        final_rgb = rgb_arr.copy()
        final_rgb[y0:y1, x0:x1] = blended_crop
        return final_rgb

    # Maximum tiles before falling back to single-pass (prevents runaway processing)
    MAX_TILES = 6

    def _tiled_inpaint(self, rgb_arr, mask_dilated, dilate_px):
        """Tiled high-resolution inpainting for large masked regions.
        Splits the masked area into overlapping 512x512 tiles for better detail.
        Falls back to single contextual-crop pass if tile count exceeds MAX_TILES."""
        h, w = rgb_arr.shape[:2]
        
        # Find bounding box of mask
        y_indices, x_indices = np.where(mask_dilated > 0)
        if len(y_indices) == 0:
            return rgb_arr.copy()
        min_y, max_y = int(y_indices.min()), int(y_indices.max())
        min_x, max_x = int(x_indices.min()), int(x_indices.max())
        
        tile_size = 512
        overlap = 96  # reduced from 128 to cut tile count
        step = tile_size - overlap
        
        # Add context padding around the mask region
        pad = 64
        region_y0 = max(0, min_y - pad)
        region_y1 = min(h, max_y + pad + 1)
        region_x0 = max(0, min_x - pad)
        region_x1 = min(w, max_x + pad + 1)
        
        # Pre-count tiles; if too many, fall back to single-pass contextual crop
        tile_coords = []
        for ty in range(region_y0, region_y1, step):
            for tx in range(region_x0, region_x1, step):
                ty1 = min(h, ty + tile_size)
                tx1 = min(w, tx + tile_size)
                ty0_c = max(0, ty1 - tile_size)
                tx0_c = max(0, tx1 - tile_size)
                if np.any(mask_dilated[ty0_c:ty1, tx0_c:tx1]):
                    tile_coords.append((ty0_c, tx0_c, ty1, tx1))
        
        if len(tile_coords) > self.MAX_TILES:
            # Too many tiles would cause a hang — use single contextual crop instead
            bw = max_x - min_x + 1
            bh = max_y - min_y + 1
            return self._contextual_crop_inpaint(
                rgb_arr, mask_dilated, min_x, max_x, min_y, max_y,
                bw, bh, 0.4, 64, dilate_px
            )
        
        # Region-local accumulators (saves memory vs full-image arrays)
        rh = region_y1 - region_y0
        rw = region_x1 - region_x0
        accum = np.zeros((rh, rw, 3), dtype=np.float64)
        weight = np.zeros((rh, rw), dtype=np.float64)
        
        for ty0, tx0, ty1, tx1 in tile_coords:
            tile_mask = mask_dilated[ty0:ty1, tx0:tx1]
            tile_rgb = rgb_arr[ty0:ty1, tx0:tx1]
            tile_result = self._run_model_512(tile_rgb, tile_mask)
            
            if tile_result.shape[:2] != tile_rgb.shape[:2]:
                tile_result = cv2.resize(tile_result, (tile_rgb.shape[1], tile_rgb.shape[0]),
                                         interpolation=cv2.INTER_LANCZOS4)
            
            # Cosine window for smooth blending
            th, tw = tile_result.shape[:2]
            wy = np.hanning(th + 2)[1:-1]
            wx = np.hanning(tw + 2)[1:-1]
            tile_weight = np.outer(wy, wx)
            
            # Only blend where mask is active
            tile_blend_mask = (tile_mask > 0).astype(np.float64)
            effective_weight = tile_weight * tile_blend_mask
            
            # Map into region-local coordinates
            ry0 = ty0 - region_y0
            rx0 = tx0 - region_x0
            ry1 = ty1 - region_y0
            rx1 = tx1 - region_x0
            accum[ry0:ry1, rx0:rx1] += tile_result.astype(np.float64) * effective_weight[:, :, None]
            weight[ry0:ry1, rx0:rx1] += effective_weight
        
        # Merge tiles into region
        final_rgb = rgb_arr.copy()
        has_data = weight > 0
        region_rgb = final_rgb[region_y0:region_y1, region_x0:region_x1]
        for c in range(3):
            region_rgb[:, :, c][has_data] = np.clip(
                accum[:, :, c][has_data] / weight[has_data], 0, 255
            ).astype(np.uint8)
        
        # Feather blend the tiled result back into original
        blur_size = max(5, dilate_px * 2 + 1)
        if blur_size % 2 == 0:
            blur_size += 1
        blend_weight = cv2.GaussianBlur(
            mask_dilated.astype(np.float32) / 255.0,
            (blur_size, blur_size),
            max(2.0, dilate_px * 0.4)
        )[:, :, None]
        
        pure_orig = blend_weight < 0.005
        blend_weight[pure_orig] = 0.0
        
        merged = (final_rgb.astype(np.float32) * blend_weight +
                  rgb_arr.astype(np.float32) * (1.0 - blend_weight))
        merged = np.clip(np.round(merged), 0, 255).astype(np.uint8)
        merged[pure_orig.squeeze(axis=2)] = rgb_arr[pure_orig.squeeze(axis=2)]
        
        return merged

    def _run_model_512(self, rgb_image: np.ndarray, mask_binary: np.ndarray) -> np.ndarray:
        """Run the appropriate model (AnimeLaMa or ONNX) on 512x512 normalized input.
        
        Args:
            rgb_image: HxWx3 uint8 RGB image.
            mask_binary: HxW uint8 binary mask (0 or 255).
            
        Returns:
            HxWx3 uint8 RGB image of reconstructed content.
        """
        if self.model_type == "anime":
            return self._run_anime_512(rgb_image, mask_binary)
        else:
            return self._run_onnx_512(rgb_image, mask_binary)

    # AnimeLaMa is fully-convolutional and accepts any input size divisible by 8 (verified:
    # 128/256/1024 run cleanly, 500/513/100x140 raise a TorchScript shape error). Running it at
    # (close to) the crop's own resolution instead of always downsampling to 512x512 avoids the
    # soft/blurry fills that a forced 512->crop_size resize produces on anything larger than 512px.
    ANIME_MAX_DIM = 768  # ~8s/tile on CPU at 768x768; larger crops still get downscaled to this cap

    @staticmethod
    def _round_up_8(x: int) -> int:
        return ((int(x) + 7) // 8) * 8

    def _run_anime_512(self, rgb_image: np.ndarray, mask_binary: np.ndarray) -> np.ndarray:
        """Run AnimeLaMa TorchScript model at native (or near-native) resolution.

        Crops that fit under ANIME_MAX_DIM once padded to a multiple of 8 are run with zero
        resampling loss (edge-replicated padding only, cropped back off afterwards). Only crops
        larger than the cap are downscaled, and only by as much as necessary.
        """
        import torch

        orig_h, orig_w = rgb_image.shape[:2]
        pad_h, pad_w = self._round_up_8(orig_h), self._round_up_8(orig_w)

        if max(pad_h, pad_w) <= self.ANIME_MAX_DIM:
            # Native resolution: pad up to a multiple of 8 with replicated edges, no resampling.
            run_h, run_w = pad_h, pad_w
            img_run = cv2.copyMakeBorder(rgb_image, 0, run_h - orig_h, 0, run_w - orig_w, cv2.BORDER_REPLICATE)
            mask_run = cv2.copyMakeBorder(mask_binary, 0, run_h - orig_h, 0, run_w - orig_w, cv2.BORDER_REPLICATE)
            was_padded = True
        else:
            # Crop exceeds the latency cap: downscale as little as possible to fit under it.
            scale = self.ANIME_MAX_DIM / float(max(orig_h, orig_w))
            run_h = max(8, self._round_up_8(int(round(orig_h * scale))))
            run_w = max(8, self._round_up_8(int(round(orig_w * scale))))
            img_run = cv2.resize(rgb_image, (run_w, run_h), interpolation=cv2.INTER_AREA)
            mask_run = cv2.resize(mask_binary, (run_w, run_h), interpolation=cv2.INTER_NEAREST)
            was_padded = False

        # AnimeLaMa expects:
        # image: float32, [0.0, 1.0], shape (1, 3, H, W)
        # mask:  float32, {0.0, 1.0}, shape (1, 1, H, W)
        img_tensor = torch.from_numpy(
            (img_run.astype(np.float32) / 255.0).transpose(2, 0, 1)
        ).unsqueeze(0)

        mask_tensor = torch.from_numpy(
            (mask_run.astype(np.float32) > 127).astype(np.float32)
        ).unsqueeze(0).unsqueeze(0)

        try:
            device = getattr(self, "_anime_device", "cpu")
            img_tensor = img_tensor.to(device)
            mask_tensor = mask_tensor.to(device)

            with torch.no_grad():
                output = self._anime_model(img_tensor, mask_tensor)

            # Output: float32, [0.0, 1.0], shape (1, 3, H, W)
            out_np = output[0].detach().cpu().numpy().transpose(1, 2, 0)
            out_rgb_run = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"[BigLamaInpainter] AnimeLaMa inference error: {e}. Falling back to Big-LaMa ONNX engine...")
            self._ensure_onnx_engine()
            return self._run_onnx_512(rgb_image, mask_binary)

        if was_padded:
            # Drop the replicated-edge padding; no resampling was ever applied.
            out_rgb = out_rgb_run[:orig_h, :orig_w]
        else:
            # Only the truly oversized path needs to scale back up.
            out_rgb = cv2.resize(out_rgb_run, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

        return out_rgb

    def _run_onnx_512(self, rgb_image: np.ndarray, mask_binary: np.ndarray) -> np.ndarray:
        """Run Big-LaMa ONNX model on a 512x512 normalized tensor."""
        orig_h, orig_w = rgb_image.shape[:2]

        # Resize to 512x512
        if (orig_w, orig_h) != (512, 512):
            img_512 = cv2.resize(rgb_image, (512, 512), interpolation=cv2.INTER_AREA)
            mask_512 = cv2.resize(mask_binary, (512, 512), interpolation=cv2.INTER_NEAREST)
        else:
            img_512 = rgb_image
            mask_512 = mask_binary

        # image: float32, range [0.0, 1.0], shape (1, 3, 512, 512)
        img_tensor = (img_512.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        # mask: float32, range {0.0, 1.0}, shape (1, 1, 512, 512)
        mask_tensor = (mask_512.astype(np.float32) > 127).astype(np.float32)[None, None, ...]

        # Model execution
        inputs = {
            self._onnx_session.get_inputs()[0].name: img_tensor,
            self._onnx_session.get_inputs()[1].name: mask_tensor
        }
        outputs = self._onnx_session.run(None, inputs)

        # Big-LaMa ONNX output is float32 in [0.0, 255.0], shape (1, 3, 512, 512)
        out_tensor = outputs[0][0]
        out_rgb_512 = np.clip(out_tensor.transpose(1, 2, 0), 0, 255).astype(np.uint8)

        # Resize back to original dimensions if needed
        if (orig_w, orig_h) != (512, 512):
            out_rgb = cv2.resize(out_rgb_512, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            out_rgb = out_rgb_512

        return out_rgb
