"""Unit tests for WatermarkRemover and ShineRemover modules."""

import os
import sys
import unittest
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deskew_pipeline import WatermarkRemover, ShineRemover, StickerPipeline


class TestPreprocessors(unittest.TestCase):
    def test_watermark_remover(self):
        remover = WatermarkRemover()

        # Create a blank image with a simulated corner watermark text
        w, h = 300, 300
        img = Image.new("RGB", (w, h), color=(240, 230, 220))
        draw = ImageDraw.Draw(img)
        draw.text((w - 120, h - 30), "© ARTIST_2026", fill=(40, 40, 40))

        res = remover.remove(img, sensitivity=60, region="corners_and_margins")
        self.assertTrue(res.watermark_detected, "Watermark was not detected.")
        self.assertLess(res.execution_time_s, 0.1, "Watermark removal took too long (> 100ms).")
        self.assertEqual(res.cleaned_image.size, (w, h))

    def test_shine_remover(self):
        remover = ShineRemover()

        # Create a test image with high-specular glare (high value, low saturation)
        w, h = 120, 200
        img = Image.new("RGB", (w, h), color=(245, 180, 170))  # Base skin tone
        draw = ImageDraw.Draw(img)
        # Add white plastic glare patch
        draw.rectangle([40, 80, 80, 120], fill=(255, 255, 255))
        # Add dark outline
        draw.line([(30, 10), (30, 190)], fill=(30, 20, 20), width=2)

        res = remover.remove_shine(img, strength=70)
        self.assertTrue(res.shine_detected, "Shine was not detected.")
        self.assertLess(res.execution_time_s, 0.1, "Shine removal took too long (> 100ms).")
        self.assertEqual(res.cleaned_image.size, (w, h))

    def test_pipeline_integration(self):
        pipeline = StickerPipeline(
            segmentor_model="digital",
            remove_watermarks=True,
            remove_shine=True,
            shine_strength=65
        )
        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 150, 150], fill=(240, 160, 150))

        res = pipeline.process(img)
        self.assertIsNotNone(res.final_rgba)
        self.assertIsNotNone(res.watermark_cleaned)
        self.assertIsNotNone(res.cel_restored)
        self.assertIn("preprocess_watermark", res.metadata["stage_times_s"])
        self.assertIn("preprocess_shine_remover", res.metadata["stage_times_s"])


if __name__ == "__main__":
    unittest.main()
