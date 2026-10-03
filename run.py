"""StickerDeskew Studio & Pipeline Launcher
- If run without arguments (or double-clicking run.bat): starts the Web Studio and opens your browser.
- If run with an image path argument: runs the vision pipeline on that image and saves the output to output_results/.
"""
import os
import sys
import time
import webbrowser
import threading

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORKSPACE_DIR)

from studio.server import run_server, PORT
from deskew_pipeline import StickerPipeline


def open_browser_for_port(port: int):
    """Waits briefly for server to be fully ready then opens browser."""
    time.sleep(0.6)
    url = f"http://localhost:{port}"
    print(f"[Launcher] Opening web browser at: {url}", flush=True)
    webbrowser.open(url)


def run_cli_mode(target_path: str, model: str = "u2net"):
    print("=" * 60, flush=True)
    print("       StickerDeskew Pipeline - CLI Processing Mode", flush=True)
    print("=" * 60, flush=True)
    if not os.path.exists(target_path):
        print(f"[Error] Path not found: {target_path}", flush=True)
        sys.exit(1)

    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
    images_to_process = []

    if os.path.isdir(target_path):
        for root, _, files in os.walk(target_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    images_to_process.append(os.path.join(root, file))
        print(f"[Folder Mode] Found {len(images_to_process)} image(s) in: {target_path}", flush=True)
    else:
        images_to_process.append(target_path)

    if not images_to_process:
        print(f"[Error] No compatible image files found in: {target_path}", flush=True)
        sys.exit(1)

    print(f"Initializing StickerPipeline (Model: {model})...", flush=True)
    pipeline = StickerPipeline(
        segmentor_model=model,
        extract_mode="character_only",
        deskew_mode="none",
        clean_shadow_smudges=True,
        smudge_sensitivity=50,
        alpha_threshold=15
    )

    out_dir = os.path.join(WORKSPACE_DIR, "output_results")
    os.makedirs(out_dir, exist_ok=True)

    success_count = 0
    import gc
    for idx, img_path in enumerate(images_to_process, 1):
        print(f"[{idx}/{len(images_to_process)}] Processing: {os.path.basename(img_path)}", flush=True)
        try:
            result = pipeline.process(img_path)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(out_dir, f"{base_name}_clean.png")
            result.save(out_path)
            print(f"    ✓ Saved: {out_path}", flush=True)
            success_count += 1
        except Exception as e:
            print(f"    ✕ Failed to process {img_path}: {e}", flush=True)
        finally:
            if 'result' in locals():
                del result
            gc.collect()

    print("-" * 60, flush=True)
    print(f"[Finished] Successfully processed {success_count}/{len(images_to_process)} image(s) to output_results/", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        model_choice = "u2net"
        if len(sys.argv) > 2:
            model_choice = sys.argv[2]
        run_cli_mode(sys.argv[1], model=model_choice)
    else:
        print("=" * 60, flush=True)
        print("       StickerDeskew Studio - Starting Application", flush=True)
        print("=" * 60, flush=True)
        print("Press Ctrl+C in this window anytime to stop the server.", flush=True)
        print("-" * 60, flush=True)

        def on_server_bound(actual_port: int):
            threading.Thread(target=open_browser_for_port, args=(actual_port,), daemon=True).start()

        try:
            run_server(port=PORT, on_bound=on_server_bound)
        except KeyboardInterrupt:
            print("\n[Launcher] Studio server stopped cleanly.", flush=True)

