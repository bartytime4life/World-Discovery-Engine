# FILE: world_engine/models/vlm_caption.py
# =============================================================================
# 🧠 Vision–Language Model (VLM) Caption & Zero-Shot Scoring Wrapper
#
# Goals
# -----
# - Provide a single, dependency-optional interface for:
#     1) generating captions for an image tile
#     2) zero-shot ranking of textual prompts vs image (e.g., "circular ditch")
#     3) producing image/text embeddings (when supported by backend)
# - Keep imports soft/optional (no hard dependency explosion).
# - Work on CPU by default; leverage CUDA if available.
# - Be deterministic (fixed seeds) where possible.
#
# Backends supported (auto-detected in order):
#   • "openclip"     — via `open_clip` (preferred for CLIP-style zero-shot)
#   • "clip"         — via `clip` (OpenAI CLIP reference)
#   • "hf-blip2"     — via `transformers` (BLIP-2 captioning)
#   • "hf-blip"      — via `transformers` (BLIP captioning)
#
# If a chosen backend is unavailable, the wrapper falls through to the next.
# If none are available, it returns safe, deterministic placeholders.
#
# Typical usage
# -------------
#   vlm = VLMCaptions.auto(model_hint="openclip")   # or VLMCaptions("clip")
#   caption = vlm.describe(pil_or_numpy_image)
#   score = vlm.rank_prompts(pil_or_numpy_image, ["circular ditch", "grid", "mound"])
#   emb_i = vlm.embed_image(pil_or_numpy_image)
#   emb_t = vlm.embed_text(["circular ditch", "grid"])
#
# Notes
# -----
# - Images can be PIL.Image, NumPy HxWxC uint8 arrays, or torch tensors.
# - Prompts are normalized with a light hygiene pass; you can also supply your own.
# - Default archaeology-oriented prompts are provided for convenience.
# - The wrapper never raises on missing heavy deps; it degrades gracefully.
# =============================================================================

from __future__ import annotations

import io
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# Torch is optional; we only import when available.
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    TORCH_AVAILABLE = False

# torchvision is optional; we only import transforms if present.
try:
    import torchvision.transforms as T  # type: ignore
    TV_AVAILABLE = True
except Exception:  # pragma: no cover
    T = None  # type: ignore
    TV_AVAILABLE = False

# Optional backends
try:
    import open_clip  # type: ignore
    OPENCLIP_AVAILABLE = True
except Exception:  # pragma: no cover
    open_clip = None  # type: ignore
    OPENCLIP_AVAILABLE = False

try:
    import clip  # type: ignore
    CLIP_AVAILABLE = True
except Exception:  # pragma: no cover
    clip = None  # type: ignore
    CLIP_AVAILABLE = False

try:
    from transformers import (  # type: ignore
        AutoProcessor,
        Blip2ForConditionalGeneration,
        BlipForConditionalGeneration,
    )
    HF_AVAILABLE = True
except Exception:  # pragma: no cover
    AutoProcessor = None  # type: ignore
    Blip2ForConditionalGeneration = None  # type: ignore
    BlipForConditionalGeneration = None  # type: ignore
    HF_AVAILABLE = False


# ------------------------- Default Archaeology Prompts ------------------------- #

DEFAULT_ARCHAEO_PROMPTS: Tuple[str, ...] = (
    "circular ditch",
    "ring-shaped earthwork",
    "rectangular clearing",
    "grid of straight lines",
    "mound or tell",
    "raised terrace",
    "linear embankment",
    "radial pattern",
    "ancient road alignment",
    "geometric earthwork",
    "moat-like feature",
)


# ------------------------------- Utilities ----------------------------------- #

def _seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _to_pil(img: Any):
    """Best-effort convert input to a PIL.Image.Image (without importing PIL globally)."""
    try:
        from PIL import Image  # lazy import
    except Exception:  # pragma: no cover
        raise RuntimeError("Pillow is required for image I/O. Please `pip install pillow`.")

    if hasattr(img, "mode") and hasattr(img, "size"):  # looks like PIL already
        return img
    if TORCH_AVAILABLE and isinstance(img, torch.Tensor):
        t = img.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)
        arr = t.numpy()
        arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    # Bytes?
    if isinstance(img, (bytes, bytearray)):
        return Image.open(io.BytesIO(img)).convert("RGB")
    raise TypeError("Unsupported image type; provide PIL.Image, np.ndarray, torch.Tensor, or bytes.")


def _device(prefer_cuda: bool = True) -> str:
    if TORCH_AVAILABLE and prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _normalize_prompts(prompts: Sequence[str]) -> List[str]:
    norm: List[str] = []
    for p in prompts:
        s = " ".join((p or "").strip().split())
        if s:
            norm.append(s)
    return norm


def _safety_redact(text: str, max_len: int = 256) -> str:
    """Trim and lightly sanitize a model string output."""
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    # Collapse spaces
    text = " ".join(text.split())
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def _np_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ------------------------------- Backend API --------------------------------- #

class _BaseVLM:
    """Abstract-ish backend API; concrete backends override a subset of methods."""

    name: str = "base"

    def available(self) -> bool:
        return False

    def describe(self, image: Any) -> str:
        return "VLM backend not available"

    def rank_prompts(self, image: Any, prompts: Sequence[str]) -> Dict[str, float]:
        # Return uniform distribution as a harmless fallback
        prompts = _normalize_prompts(prompts)
        if not prompts:
            return {}
        score = 1.0 / float(len(prompts))
        return {p: score for p in prompts}

    def embed_image(self, image: Any) -> Optional[np.ndarray]:
        return None

    def embed_text(self, prompts: Sequence[str]) -> Optional[np.ndarray]:
        return None


# ------------------------------ OpenCLIP Backend ----------------------------- #

class _OpenCLIPBackend(_BaseVLM):
    name = "openclip"

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: Optional[str] = None):
        self.ok = False
        self.device = device or _device()
        if not OPENCLIP_AVAILABLE or not TORCH_AVAILABLE:
            return
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(  # type: ignore
                model_name, pretrained=pretrained, device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)  # type: ignore
            self.ok = True
        except Exception:
            self.ok = False

    def available(self) -> bool:
        return bool(self.ok)

    def _prep(self, image: Any) -> "torch.Tensor":
        im = _to_pil(image).convert("RGB")
        if TV_AVAILABLE:
            return self.preprocess(im).unsqueeze(0).to(self.device)  # type: ignore[attr-defined]
        # Minimal fallback if transforms missing (shouldn't happen for open_clip)
        arr = np.array(im).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return t

    @torch.no_grad()  # type: ignore
    def rank_prompts(self, image: Any, prompts: Sequence[str]) -> Dict[str, float]:
        if not self.available():
            return super().rank_prompts(image, prompts)
        prompts = _normalize_prompts(prompts)
        if not prompts:
            return {}
        img = self._prep(image)
        tokens = self.tokenizer(prompts).to(self.device)  # type: ignore

        # Encode
        img_feat = self.model.encode_image(img)
        txt_feat = self.model.encode_text(tokens)
        # Normalize to unit
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        # Cosine sim
        sims = (img_feat @ txt_feat.T).squeeze(0).float().cpu().numpy()
        probs = _np_softmax(sims)
        return {p: float(prob) for p, prob in zip(prompts, probs)}

    @torch.no_grad()  # type: ignore
    def embed_image(self, image: Any) -> Optional[np.ndarray]:
        if not self.available():
            return None
        img = self._prep(image)
        feat = self.model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).float().cpu().numpy()

    @torch.no_grad()  # type: ignore
    def embed_text(self, prompts: Sequence[str]) -> Optional[np.ndarray]:
        if not self.available():
            return None
        prompts = _normalize_prompts(prompts)
        if not prompts:
            return None
        tokens = self.tokenizer(prompts).to(self.device)  # type: ignore
        feat = self.model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.float().cpu().numpy()

    # OpenCLIP is not a captioning model—provide a heuristic pseudo-caption via top-prompt
    def describe(self, image: Any) -> str:
        # Use a default archaeology vocabulary to pick the most likely phrase,
        # prefixed with a generic leading phrase for readability.
        scores = self.rank_prompts(image, DEFAULT_ARCHAEO_PROMPTS)
        if not scores:
            return "image with natural terrain"
        best = max(scores.items(), key=lambda kv: kv[1])[0]
        return _safety_redact(f"likely {best} in a vegetated landscape")


# -------------------------------- CLIP Backend ------------------------------- #

class _CLIPBackend(_BaseVLM):
    name = "clip"

    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        self.ok = False
        self.device = device or _device()
        if not CLIP_AVAILABLE or not TORCH_AVAILABLE:
            return
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)  # type: ignore
            self.ok = True
        except Exception:
            self.ok = False

    def available(self) -> bool:
        return bool(self.ok)

    def _prep(self, image: Any) -> "torch.Tensor":
        im = _to_pil(image).convert("RGB")
        return self.preprocess(im).unsqueeze(0).to(self.device)  # type: ignore[attr-defined]

    @torch.no_grad()  # type: ignore
    def rank_prompts(self, image: Any, prompts: Sequence[str]) -> Dict[str, float]:
        if not self.available():
            return super().rank_prompts(image, prompts)
        prompts = _normalize_prompts(prompts)
        if not prompts:
            return {}
        img = self._prep(image)
        tokens = clip.tokenize(prompts).to(self.device)  # type: ignore

        img_feat = self.model.encode_image(img)
        txt_feat = self.model.encode_text(tokens)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        sims = (img_feat @ txt_feat.T).squeeze(0).float().cpu().numpy()
        probs = _np_softmax(sims)
        return {p: float(prob) for p, prob in zip(prompts, probs)}

    @torch.no_grad()  # type: ignore
    def embed_image(self, image: Any) -> Optional[np.ndarray]:
        if not self.available():
            return None
        img = self._prep(image)
        feat = self.model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).float().cpu().numpy()

    @torch.no_grad()  # type: ignore
    def embed_text(self, prompts: Sequence[str]) -> Optional[np.ndarray]:
        if not self.available():
            return None
        prompts = _normalize_prompts(prompts)
        if not prompts:
            return None
        tokens = clip.tokenize(prompts).to(self.device)  # type: ignore
        feat = self.model.encode_text(tokens)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.float().cpu().numpy()

    def describe(self, image: Any) -> str:
        scores = self.rank_prompts(image, DEFAULT_ARCHAEO_PROMPTS)
        if not scores:
            return "image with natural terrain"
        best = max(scores.items(), key=lambda kv: kv[1])[0]
        return _safety_redact(f"likely {best} in a vegetated landscape")


# ---------------------------- HF BLIP / BLIP-2 Backend ----------------------- #

class _HFBlipBackend(_BaseVLM):
    name = "hf-blip"

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", device: Optional[str] = None):
        self.ok = False
        self.device = device or _device()
        self.processor = None
        self.model = None
        if not HF_AVAILABLE or not TORCH_AVAILABLE:
            return
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)  # type: ignore
            self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)  # type: ignore
            self.ok = True
        except Exception:
            self.ok = False

    def available(self) -> bool:
        return bool(self.ok)

    @torch.no_grad()  # type: ignore
    def describe(self, image: Any) -> str:
        if not self.available():
            return "an outdoor scene"
        pil = _to_pil(image).convert("RGB")
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)  # type: ignore
        out = self.model.generate(**inputs, max_new_tokens=30)  # type: ignore
        caption = self.processor.decode(out[0], skip_special_tokens=True)  # type: ignore
        return _safety_redact(caption)


class _HFBlip2Backend(_BaseVLM):
    name = "hf-blip2"

    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", device: Optional[str] = None):
        self.ok = False
        self.device = device or _device()
        self.processor = None
        self.model = None
        if not HF_AVAILABLE or not TORCH_AVAILABLE:
            return
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)  # type: ignore
            self.model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)  # type: ignore
            self.model = self.model.to(self.device)
            self.ok = True
        except Exception:
            self.ok = False

    def available(self) -> bool:
        return bool(self.ok)

    @torch.no_grad()  # type: ignore
    def describe(self, image: Any) -> str:
        if not self.available():
            return "an outdoor scene"
        pil = _to_pil(image).convert("RGB")
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)  # type: ignore
        out = self.model.generate(**inputs, max_new_tokens=40)  # type: ignore
        caption = self.processor.decode(out[0], skip_special_tokens=True)  # type: ignore
        return _safety_redact(caption)


# --------------------------- High-Level Public Wrapper ------------------------ #

@dataclass
class VLMConfig:
    backend: str = "auto"             # "auto" | "openclip" | "clip" | "hf-blip2" | "hf-blip"
    prefer_cuda: bool = True          # use GPU if available
    seed: int = 42                    # determinism
    # Backend model choices (only used if that backend is selected)
    openclip_model: str = "ViT-B-32"
    openclip_pretrained: str = "laion2b_s34b_b79k"
    clip_model: str = "ViT-B/32"
    blip_model: str = "Salesforce/blip-image-captioning-base"
    blip2_model: str = "Salesforce/blip2-opt-2.7b"


class VLMCaptions:
    """
    Unified VLM wrapper with graceful degradation.

    Methods
    -------
    describe(image) -> str
    rank_prompts(image, prompts) -> Dict[str, float]
    embed_image(image) -> Optional[np.ndarray]
    embed_text(prompts) -> Optional[np.ndarray]
    """

    def __init__(self, model_name: str = "auto", **kwargs):
        """
        Parameters
        ----------
        model_name : str
            "auto" | "openclip" | "clip" | "hf-blip2" | "hf-blip"
        kwargs     : passed into VLMConfig to customize model selection.
        """
        _seed_everything()
        cfg_dict = dict(backend=model_name, **kwargs)
        self.cfg = VLMConfig(**cfg_dict)
        self.device = _device(self.cfg.prefer_cuda)
        self.backend: _BaseVLM = self._init_backend()

    # --- factory helpers --- #
    @classmethod
    def auto(cls, model_hint: str = "openclip", **kwargs) -> "VLMCaptions":
        """
        Convenience: select the best available backend, preferring `model_hint`.
        """
        # Try hinted backend first, then fall back to auto
        order = [model_hint, "openclip", "clip", "hf-blip2", "hf-blip"]
        for b in order:
            inst = cls(b, **kwargs)
            if inst.is_available():
                return inst
        return cls("auto", **kwargs)

    def _init_backend(self) -> _BaseVLM:
        """Instantiate the requested/available backend."""
        name = (self.cfg.backend or "auto").lower()
        candidates: List[_BaseVLM] = []

        if name in ("openclip", "auto"):
            candidates.append(_OpenCLIPBackend(self.cfg.openclip_model, self.cfg.openclip_pretrained, self.device))
        if name in ("clip", "auto"):
            candidates.append(_CLIPBackend(self.cfg.clip_model, self.device))
        if name in ("hf-blip2", "auto"):
            candidates.append(_HFBlip2Backend(self.cfg.blip2_model, self.device))
        if name in ("hf-blip", "auto"):
            candidates.append(_HFBlipBackend(self.cfg.blip_model, self.device))

        # Pick first available
        for b in candidates:
            if getattr(b, "available", lambda: False)():
                return b
        # Fallback empty base
        return _BaseVLM()

    # --- capability queries --- #
    def backend_name(self) -> str:
        return getattr(self.backend, "name", "base")

    def is_available(self) -> bool:
        return self.backend.available()

    # --- core API --- #
    def describe(self, img: Any) -> str:
        """
        Generate a short caption/description for the image tile.
        For CLIP-style backends this picks the best matching archaeology prompt.
        """
        try:
            text = self.backend.describe(img)
        except Exception:
            text = "an outdoor scene"
        return _safety_redact(text)

    def rank_prompts(self, img: Any, prompts: Sequence[str], normalize: bool = True) -> Dict[str, float]:
        """
        Zero-shot rank textual prompts against the image.

        Returns
        -------
        Dict[prompt, score] where scores sum to 1.0 when `normalize=True`.
        """
        prompts = _normalize_prompts(prompts)
        if not prompts:
            return {}
        try:
            scores = self.backend.rank_prompts(img, prompts)
            if normalize and scores:
                arr = np.array([scores[p] for p in prompts], dtype=np.float64)
                if arr.sum() > 0:
                    arr = arr / arr.sum()
                scores = {p: float(v) for p, v in zip(prompts, arr.tolist())}
            return scores
        except Exception:
            # graceful fallback: uniform
            s = 1.0 / len(prompts)
            return {p: s for p in prompts}

    def embed_image(self, img: Any) -> Optional[np.ndarray]:
        try:
            return self.backend.embed_image(img)
        except Exception:
            return None

    def embed_text(self, prompts: Sequence[str]) -> Optional[np.ndarray]:
        prompts = _normalize_prompts(prompts)
        if not prompts:
            return None
        try:
            return self.backend.embed_text(prompts)
        except Exception:
            return None

    # --- quality helpers --- #
    def detect_keywords(
        self,
        img: Any,
        prompts: Sequence[str] = DEFAULT_ARCHAEO_PROMPTS,
        threshold: float = 0.25,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        Convenience: return the top-K prompts passing a probability threshold.
        """
        ranks = self.rank_prompts(img, prompts)
        if not ranks:
            return []
        items = sorted(ranks.items(), key=lambda kv: kv[1], reverse=True)
        items = [kv for kv in items if kv[1] >= threshold]
        return items[:top_k]
