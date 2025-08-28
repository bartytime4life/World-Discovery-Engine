"""
VLM Caption Wrapper (stub)

This shows where you'd integrate a vision-language model (CLIP, PaliGemma, etc.).
"""
from __future__ import annotations
from typing import Any


class VLMCaptions:
    def __init__(self, model_name: str = "clip"):
        self.model_name = model_name

    def describe(self, img: Any) -> str:
        # Integrate actual model calls here
        return "placeholder caption from VLM"
