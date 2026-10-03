"""Command-line interface for the Image Deskew & Character Extraction Pipeline.
Usage:
    python -m deskew_pipeline.cli --input ./my_sticker.png --output ./output_results/clean_square.png --mode full_sticker
    python -m deskew_pipeline.cli --input ./input_folder --output ./output_results/ --mode character_only
"""

import argparse
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from deskew_pipeline.pipeline import StickerPipeline
else:
    from .pipeline import StickerPipeline



def main():
    parser = argparse.ArgumentParser(description="Image Deskew & Character Background Removal Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Path to input image file or directory")
    parser.add_argument("--output", "-o", required=True, help="Path to output PNG file or directory")
    parser.add_argument("--model", "-m", default="u2net", choices=["u2net", "birefnet-general", "digital"],
                        help="Segmentation model to use")
    parser.add_argument("--mode", default="character_only", choices=["character_only", "full_sticker"],
                        help="Extraction mode: 'character_only' (strips white die-cut margin) or 'full_sticker'")
    parser.add_argument("--deskew", default="auto", choices=["auto", "quad", "baseline", "moments", "parametric", "none"],
                        help="Deskew strategy")
    parser.add_argument("--ensure-connected", action="store_true", default=True,
                        help="Ensure extracted asset is a single contiguous connected piece (no floating fragments)")
    parser.add_argument("--connectivity-mode", default="bridge", choices=["bridge", "largest"],
                        help="'bridge' connects floating elements to main body; 'largest' discards floating parts")
    parser.add_argument("--pca-align", action="store_true", default=True,
                        help="Perform PCA analysis after deskewing to ensure orientation is aligned with the bottom")
    parser.add_argument("--pca-mode", default="bottom_upright",
                        choices=["bottom_upright", "bottom_baseline", "auto", "horizontal", "snap_orthogonal"],
                        help="PCA alignment strategy: 'bottom_upright', 'bottom_baseline', 'auto', 'horizontal', 'snap_orthogonal'")
    parser.add_argument("--add-diecut-border", action="store_true", default=False,
                        help="Add a crisp rounded die-cut sticker border (stroke) around the asset")
    parser.add_argument("--border-width", type=int, default=14, help="Width of custom die-cut border in pixels")
    parser.add_argument("--ai-enhance", action="store_true", default=False,
                        help="Upscale output into higher-resolution with AI super-resolution")
    parser.add_argument("--enhance-scale", type=int, default=2, choices=[2, 4], help="AI upscale scale factor")
    parser.add_argument("--padding", type=int, default=15, help="Padding around cropped output")
    parser.add_argument("--pitch", type=float, default=0.0, help="Parametric 3D pitch angle (if deskew=parametric)")
    parser.add_argument("--yaw", type=float, default=0.0, help="Parametric 3D yaw angle (if deskew=parametric)")
    parser.add_argument("--roll", type=float, default=0.0, help="Parametric 3D roll angle (if deskew=parametric)")
    parser.add_argument("--save-stages", action="store_true", help="Save intermediate pipeline stages")
    parser.add_argument("--remove-shine", action="store_true", default=False,
                        help="Remove glare and apply ultra-detailed anime cel color restoration")
    parser.add_argument("--shine-strength", type=int, default=60, help="Cel flattening and glare suppression strength (10-100)")
    parser.add_argument("--color-tiers", type=int, default=40, help="Number of anime cel color bands for master source fidelity (16-64)")

    args = parser.parse_args()

    pipeline = StickerPipeline(
        segmentor_model=args.model,
        extract_mode=args.mode,
        deskew_mode=args.deskew,
        padding_px=args.padding,
        ensure_connected=args.ensure_connected,
        connectivity_mode=args.connectivity_mode,
        pca_align=args.pca_align,
        pca_align_mode=args.pca_mode,
        add_diecut_border=args.add_diecut_border,
        border_thickness_px=args.border_width,
        enhance_resolution=args.ai_enhance,
        enhance_scale=args.enhance_scale,
        remove_shine=args.remove_shine,
        shine_strength=args.shine_strength,
        color_tiers=args.color_tiers
    )

    if os.path.isfile(args.input):
        print(f"Processing single image: {args.input}")
        result = pipeline.process(
            args.input,
            parametric_pitch=args.pitch,
            parametric_yaw=args.yaw,
            parametric_roll=args.roll
        )
        if args.save_stages:
            stage_dir = os.path.splitext(args.output)[0] + "_stages"
            result.save_all_stages(stage_dir)
        result.save(args.output)
        print(f"Done! Final output size: {result.metadata['final_size']}")
    elif os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        valid_exts = (".png", ".jpg", ".jpeg", ".webp")
        files = [f for f in os.listdir(args.input) if f.lower().endswith(valid_exts)]
        print(f"Found {len(files)} images in directory: {args.input}")
        import gc
        for i, fname in enumerate(files, 1):
            in_path = os.path.join(args.input, fname)
            out_name = os.path.splitext(fname)[0] + "_deskewed.png"
            out_path = os.path.join(args.output, out_name)
            print(f"[{i}/{len(files)}] Processing: {fname}")
            try:
                result = pipeline.process(in_path)
                result.save(out_path)
            finally:
                if 'result' in locals():
                    del result
                gc.collect()
        print("Batch processing complete!")
    else:
        print(f"Error: Input path '{args.input}' does not exist.")


if __name__ == "__main__":
    main()
