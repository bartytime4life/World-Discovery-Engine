"""
Image/CV utilities (stubs)

Plug-replaceable hooks for:
- edge detection
- Hough shapes
- texture features
- VLM captions

Kept as placeholders to avoid heavy deps by default; real implementations can import cv2, skimage, etc.
"""
from __future__ import annotations
from typing import Dict, List, Tuple


def dummy_edges(gray_img) -> Dict:
    """Placeholder returning a static dictionary."""
    return {"edges": 0, "edge_density": 0.0}


def dummy_hough(gray_img) -> Dict:
    """Placeholder; returns no lines/circles."""
    return {"lines": 0, "circles": 0}


def dummy_texture(gray_img) -> Dict:
    """Placeholder for LBP/GLCM features."""
    return {"glcm_contrast": 0.0, "lbp_hist": []}


def dummy_vlm_caption(rgb_img) -> str:
    """Placeholder caption."""
    return "scene with vegetation and water bodies (placeholder)"
