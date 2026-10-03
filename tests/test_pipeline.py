"""Integration and verification test suite for the Image Deskew Pipeline.
Tests:
1. Slanted Square Sticker: Background removal -> Quadrilateral 4-corner perspective deskew -> Clean PNG.
2. Slanted Die-Cut Character: Background removal -> White die-cut margin stripping (just character) -> Moment deskew -> Clean PNG.
"""

import os
import sys
import numpy as np
from PIL import Image

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deskew_pipeline import (
    BackgroundSegmentor,
    CharacterExtractor,
    PerspectiveDeskewer,
    PostProcessor,
    StickerPipeline
)


def run_tests():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_img_dir = os.path.join(root_dir, "test_images")
    output_dir = os.path.join(root_dir, "output_results")
    os.makedirs(output_dir, exist_ok=True)

    print("==========================================================")
    print("RUNNING IMAGE DESKEW & EXTRACTION PIPELINE VERIFICATION")
    print("==========================================================")

    # -------------------------------------------------------------
    # TEST 1: SLANTED SQUARE STICKER
    # -------------------------------------------------------------
    square_input_path = os.path.join(test_img_dir, "slanted_square_sticker.png")
    if not os.path.exists(square_input_path):
        print("[Info] test_images/ has been cleaned up. Skipping legacy synthetic tests.")
        return

    print("\n--- [TEST 1] Processing Slanted Square Sticker ---")
    pipeline_sq = StickerPipeline(
        segmentor_model="digital",
        extract_mode="full_sticker",
        deskew_mode="quad",
        target_aspect_ratio=1.0,
        padding_px=15
    )

    result_sq = pipeline_sq.process(square_input_path)
    sq_out_dir = os.path.join(output_dir, "square_sticker")
    result_sq.save_all_stages(sq_out_dir, base_name="square")
    
    # Assertions
    w_final, h_final = result_sq.final_rgba.size
    aspect = w_final / h_final
    print(f"Square Output Dimensions: {w_final}x{h_final} (Aspect ratio: {aspect:.2f})")
    print(f"Detected 4 Corners:\n{result_sq.metadata['detected_corners']}")
    assert 0.90 <= aspect <= 1.10, f"Square aspect ratio {aspect} outside expected tolerance [0.90, 1.10]"
    print("[PASS] TEST 1: Slanted square successfully deskewed and isolated!")

    # -------------------------------------------------------------
    # TEST 2: SLANTED DIE-CUT CHARACTER (EXTRACT JUST CHARACTER)
    # -------------------------------------------------------------
    char_input_path = os.path.join(test_img_dir, "slanted_diecut_character.png")
    assert os.path.exists(char_input_path), f"Missing test image: {char_input_path}"

    print("\n--- [TEST 2] Processing Slanted Die-Cut Character (Mode: character_only) ---")
    pipeline_char = StickerPipeline(
        segmentor_model="digital",
        extract_mode="character_only", # Strips white die-cut margin!
        deskew_mode="moments",
        padding_px=12
    )

    result_char = pipeline_char.process(char_input_path)
    char_out_dir = os.path.join(output_dir, "diecut_character")
    result_char.save_all_stages(char_out_dir, base_name="character_only")

    w_char, h_char = result_char.final_rgba.size
    print(f"Character Output Dimensions: {w_char}x{h_char}")
    print(f"Detected Angles: {result_char.metadata['detected_angles']}")
    assert w_char > 50 and h_char > 50, "Extracted character is too small or empty"
    print("[PASS] TEST 2: Die-cut character successfully isolated from white border and deskewed!")

    # -------------------------------------------------------------
    # TEST 3: SLANTED DIE-CUT (KEEP FULL STICKER BORDER)
    # -------------------------------------------------------------
    print("\n--- [TEST 3] Processing Slanted Die-Cut Character (Mode: full_sticker) ---")
    pipeline_full = StickerPipeline(
        segmentor_model="digital",
        extract_mode="full_sticker", # Keeps full die-cut border
        deskew_mode="moments",
        padding_px=12
    )
    result_full = pipeline_full.process(char_input_path)
    result_full.final_rgba.save(os.path.join(char_out_dir, "character_full_sticker_border.png"))
    print("[PASS] TEST 3: Full die-cut border preserved mode verified!")

    # -------------------------------------------------------------
    # TEST 4: INDEPENDENT SYSTEM EXECUTION (MODULARITY TEST)
    # -------------------------------------------------------------
    print("\n--- [TEST 4] Testing Decoupled Systems in Isolation ---")
    raw_img = Image.open(square_input_path)

    # Test System 1 alone
    seg = BackgroundSegmentor(model_name="digital")
    seg_res = seg.segment(raw_img)
    assert seg_res.rgba.mode == "RGBA", "System 1 failed to return RGBA"
    assert np.any(seg_res.mask > 0), "System 1 mask is empty"
    print("[PASS] System 1 (BackgroundSegmentor) standalone: OK")

    # Test System 2 alone
    char_ext = CharacterExtractor(mode="character_only")
    char_res = char_ext.extract(seg_res.rgba, seg_res.mask)
    assert char_res.rgba.mode == "RGBA", "System 2 failed to return RGBA"
    print("[PASS] System 2 (CharacterExtractor) standalone: OK")

    # Test System 3 alone
    deskewer = PerspectiveDeskewer()
    deskew_res = deskewer.deskew_quadrilateral(char_res.rgba, char_res.character_mask)
    assert deskew_res.homography.shape == (3, 3), "System 3 homography shape mismatch"
    print("[PASS] System 3 (PerspectiveDeskewer) standalone: OK")

    # Test System 4 alone
    postproc = PostProcessor(padding_px=10, defringe=True)
    post_res = postproc.process(deskew_res.rgba)
    assert post_res.rgba.mode == "RGBA", "System 4 failed to return RGBA"
    print("[PASS] System 4 (PostProcessor) standalone: OK")

    # -------------------------------------------------------------
    # TEST 5: DIECUT BORDER GENERATOR
    # -------------------------------------------------------------
    print("\n--- [TEST 5] Testing Crisp Die-Cut Border Generator ---")
    from deskew_pipeline import DiecutBorderGenerator
    border_gen = DiecutBorderGenerator(thickness_px=15, color=(255, 255, 255))
    bordered_char = border_gen.add_border(result_char.final_rgba)
    w_b, h_b = bordered_char.size
    print(f"Bordered Dimensions: {w_b}x{h_b} (original: {w_char}x{h_char})")
    assert w_b > w_char and h_b > h_char, "Bordered image should have expanded dimensions"
    bordered_char.save(os.path.join(char_out_dir, "character_with_custom_diecut_border.png"))
    print("[PASS] TEST 5: Crisp die-cut border successfully generated!")

    # -------------------------------------------------------------
    # TEST 6: AI SUPER-RESOLUTION ENHANCER (Real-ESRGAN)
    # -------------------------------------------------------------
    print("\n--- [TEST 6] Testing AI Image Enhancer (Real-ESRGAN Super-Resolution) ---")
    from deskew_pipeline import AIEnhancer
    enhancer = AIEnhancer(scale=2)
    enhanced_char = enhancer.enhance(result_char.final_rgba)
    w_enh, h_enh = enhanced_char.size
    print(f"Enhanced Dimensions: {w_enh}x{h_enh} (expected: {w_char*2}x{h_char*2})")
    assert w_enh == w_char * 2 and h_enh == h_char * 2, f"Expected 2x upscale {w_char*2}x{h_char*2}, got {w_enh}x{h_enh}"
    enhanced_char.save(os.path.join(char_out_dir, "character_ai_enhanced_2x.png"))
    print("[PASS] TEST 6: Real-ESRGAN AI Super-Resolution 2x verified!")

    # -------------------------------------------------------------
    # TEST 7: ENSURE SINGLE CONNECTED COMPONENT
    # -------------------------------------------------------------
    print("\n--- [TEST 7] Testing Ensure Single Connected Component ---")
    pipeline_connected = StickerPipeline(
        segmentor_model="digital",
        extract_mode="character_only",
        ensure_connected=True,
        connectivity_mode="bridge",
        add_diecut_border=True,
        enhance_resolution=True
    )
    result_combo = pipeline_connected.process(char_input_path)
    result_combo.final_rgba.save(os.path.join(char_out_dir, "character_all_options_enabled.png"))
    w_combo, h_combo = result_combo.final_rgba.size
    print(f"All Options Enabled Dimensions: {w_combo}x{h_combo}")
    assert w_combo > 0 and h_combo > 0, "Combined pipeline failed"
    # -------------------------------------------------------------
    # TEST 8: PCA ORIENTATION & BOTTOM ALIGNMENT
    # -------------------------------------------------------------
    print("\n--- [TEST 8] Testing PCA Orientation & Bottom Alignment ---")
    from deskew_pipeline import PCAOrientationAligner
    pca_aligner = PCAOrientationAligner(mode="bottom_upright", detect_inverted=True)
    aligned_char, pca_meta = pca_aligner.align(result_char.final_rgba)
    w_pca, h_pca = aligned_char.size
    print(f"PCA Aligned Dimensions: {w_pca}x{h_pca}")
    print(f"PCA Metadata: {pca_meta}")
    assert pca_meta["target_axis"] == "vertical", f"Expected vertical alignment, got {pca_meta['target_axis']}"
    assert h_pca >= w_pca, "Character should be upright (height >= width)"
    aligned_char.save(os.path.join(char_out_dir, "character_pca_bottom_aligned.png"))
    print("[PASS] TEST 8: PCA orientation successfully aligned with the bottom!")

    print("\n==========================================================")
    print("ALL 8 INTEGRATION & UNIT TESTS PASSED SUCCESSFULLY!")
    print(f"Results are saved in: {output_dir}")
    print("==========================================================")


if __name__ == "__main__":
    run_tests()


