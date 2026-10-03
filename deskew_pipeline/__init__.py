"""Image Deskew & Background Removal Pipeline Suite
A modular architecture of 4 decoupled systems for stickers & digital character assets.
"""

from .segmentor import BackgroundSegmentor, SegmentationResult
from .character_extractor import CharacterExtractor, CharacterExtractionResult
from .deskewer import PerspectiveDeskewer, DeskewResult
from .postprocessor import PostProcessor, PostProcessResult
from .border_generator import DiecutBorderGenerator
from .enhancer import AIEnhancer
from .pca_aligner import PCAOrientationAligner
from .watermark_remover import WatermarkRemover, WatermarkRemovalResult
from .shine_remover import ShineRemover, ShineRemovalResult
from .inpainter import BigLamaInpainter
from .color_enhancer import ColorEnhancer, ColorEnhanceResult
from .pipeline import StickerPipeline, PipelineResult

__all__ = [
    "BackgroundSegmentor",
    "SegmentationResult",
    "CharacterExtractor",
    "CharacterExtractionResult",
    "PerspectiveDeskewer",
    "DeskewResult",
    "PostProcessor",
    "PostProcessResult",
    "DiecutBorderGenerator",
    "AIEnhancer",
    "PCAOrientationAligner",
    "WatermarkRemover",
    "WatermarkRemovalResult",
    "ShineRemover",
    "ShineRemovalResult",
    "BigLamaInpainter",
    "ColorEnhancer",
    "ColorEnhanceResult",
    "StickerPipeline",
    "PipelineResult"
]

