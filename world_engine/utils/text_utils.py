# world_engine/utils/text_utils.py
# ======================================================================================
# World Discovery Engine (WDE)
# Text Utilities — OCR, cleaning, tokenization, entity & coordinate extraction,
#                  gazetteer matching, context windows, and redaction helpers
# --------------------------------------------------------------------------------------
# Purpose
#   Self-contained, Kaggle-friendly text utilities with optional dependencies and robust
#   fallbacks. Supports:
#     • OCR for images (and PDFs if pdf2image available)
#     • Text normalization & cleaning (whitespace, punctuation, accents)
#     • Lightweight tokenization & sentence splitting (regex-based, Unicode-aware)
#     • Keyword/keyphrase extraction (lightweight RAKE-like scoring)
#     • NER-lite: dates, coordinates (DMS/decimal/loose-DMS), hydro terms, toponyms
#     • Gazetteer name matching (exact + fuzzy via difflib), with span/context windows
#     • Context-window extraction around hits (e.g., for dossier snippets)
#     • Coordinate rounding/redaction for ethical release masks
#
# Design notes
#   • Optional deps (pytesseract, pillow, pdf2image, spacy). Gracefully degrade if missing.
#   • No network calls. Deterministic and unit-testable.
#   • Minimal global state. All functions are pure or locally seeded where needed.
#   • Avoid heavy NLP stacks unless user opts in (spaCy optional).
#
# License
#   MIT (c) 2025 World Discovery Engine contributors
# ======================================================================================

from __future__ import annotations

import io
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Optional Dependencies (graceful fallbacks)
# --------------------------------------------------------------------------------------
try:
    from PIL import Image  # type: ignore
    _PIL = True
except Exception:
    _PIL = False

try:
    import pytesseract  # type: ignore
    _TESSERACT = True
except Exception:
    _TESSERACT = False

try:
    # pdf2image requires poppler on the system to rasterize PDF -> PIL.Image list
    from pdf2image import convert_from_path  # type: ignore
    _PDF2IMAGE = True
except Exception:
    _PDF2IMAGE = False

try:
    import spacy  # type: ignore
    _SPACY = True
except Exception:
    _SPACY = False


__all__ = [
    # OCR
    "ocr_image",
    "ocr_pdf",
    # Cleaning / normalization
    "clean_text",
    "normalize_whitespace",
    "strip_accents",
    "normalize_quotes",
    # Tokenization / sentences
    "simple_tokens",
    "split_sentences",
    "split_paragraphs",
    # Keywords
    "simple_keywords",
    # Entities
    "extract_entities",
    "parse_coordinates",
    "toponym_candidates",
    "TextEntityResult",
    # Gazetteer
    "match_gazetteer",
    # Context & redaction
    "extract_context_windows",
    "round_coordinates",
    "redact_coordinates",
]


# --------------------------------------------------------------------------------------
# Constants & Small Utilities
# --------------------------------------------------------------------------------------

# Minimal English/Portuguese stopword set (extend as needed; tuned for Kaggle environment)
_STOPWORDS = {
    # English
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "with", "for", "from", "by", "as",
    "is", "are", "was", "were", "be", "been", "being", "it", "its", "that", "this", "these",
    "those", "at", "which", "into", "over", "under", "about", "through", "during", "before",
    "after", "above", "below", "up", "down", "no", "not", "only", "but", "such", "than",
    # Portuguese (common in Amazonia sources)
    "o", "a", "os", "as", "de", "do", "da", "dos", "das", "e", "ou", "em", "no", "na", "nos",
    "nas", "com", "por", "para", "como", "que", "se", "ao", "à", "às", "aos", "um", "uma",
    "uns", "umas", "não",
}

# Hydrologic & geomorphological terms (lightweight tagger)
_HYDRO_TERMS = {
    "river", "rio", "igarapé", "igarape", "lagoon", "laguna", "lake", "lago", "stream", "canal",
    "channel", "floodplain", "várzea", "varzea", "wetland", "marsh", "swamp", "creek", "estuário",
    "estuario", "mouth", "tributary", "furos", "igarapés",
}

# Basic date patterns (YYYY, dd/mm/yyyy, dd Mon yyyy)
_DATE_PAT = re.compile(
    r"\b(?:(?:19|20)\d{2}|(?:\d{1,2}[/.\-]\d{1,2}[/.\-](?:19|20)?\d{2})|(?:\d{1,2}\s+[A-Za-z]{3,9}\s+(?:19|20)\d{2}))\b"
)

# Decimal coordinate pattern (e.g., -3.456, -60.123)
_DEC_COORD_PAT = re.compile(
    r"(?P<lat>[+-]?\d{1,2}(?:\.\d+)?)[,\s]+(?P<lon>[+-]?\d{1,3}(?:\.\d+)?)"
)

# DMS coordinate (strict; e.g., 03°27'22"S, 060°30'45"W)
_DMS_PAT = re.compile(
    r"(?P<lat_deg>\d{1,2})[°:\s]+(?P<lat_min>\d{1,2})['’:\s]+(?P<lat_sec>\d{1,2}(?:\.\d+)?)"
    r"[\"”]?\s*(?P<lat_dir>[NS])[,;\s]+"
    r"(?P<lon_deg>\d{1,3})[°:\s]+(?P<lon_min>\d{1,2})['’:\s]+(?P<lon_sec>\d{1,2}(?:\.\d+)?)"
    r"[\"”]?\s*(?P<lon_dir>[EW])",
    re.IGNORECASE,
)

# Loose DMS (space-separated, punctuation-agnostic): 03 27 22 S 060 30 45 W
_LOOSE_DMS_PAT = re.compile(
    r"(?P<lat_deg>\d{1,2})\s+(?P<lat_min>\d{1,2})\s+(?P<lat_sec>\d{1,2}(?:\.\d+)?)\s*(?P<lat_dir>[NS])[,;\s]+"
    r"(?P<lon_deg>\d{1,3})\s+(?P<lon_min>\d{1,2})\s+(?P<lon_sec>\d{1,2}(?:\.\d+)?)\s*(?P<lon_dir>[EW])",
    re.IGNORECASE,
)

# Title-case detector for toponyms (greedy but practical; captures "Rio Negro", "São Gabriel")
_TOPONYM_PAT = re.compile(
    r"\b([A-ZÁÂÃÀÇÉÊÍÓÔÕÚÜ][a-záâãàçéêíóôõúü]+(?:[ -][A-ZÁÂÃÀÇÉÊÍÓÔÕÚÜ][a-záâãàçéêíóôõúü]+)*)\b"
)

# Sentence boundaries (very lightweight; avoids splitting on common abbreviations)
_SENT_PAT = re.compile(
    r"(?<!\b[A-Z][a-z])(?<=[.!?])\s+(?=[A-ZÁÂÃÀÇÉÊÍÓÔÕÚÜ])"
)

# Smart quotes and dashes normalization map
_PUNC_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201A": "'", "\u201B": "'",
    "\u201C": '"', "\u201D": '"', "\u201E": '"', "\u201F": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-",  # en/em dash, minus
}


# --------------------------------------------------------------------------------------
# Data Classes
# --------------------------------------------------------------------------------------

@dataclass
class TextEntityResult:
    """
    Structured entity extraction result.
    """
    dates: List[str]
    coordinates: List[Tuple[float, float]]
    hydro_terms: List[str]
    toponyms: List[str]
    keywords: List[str]


# --------------------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------------------

def ocr_image(
    image_input: str | bytes | "Image.Image",
    lang: str = "eng",
    psm: Optional[int] = None,
) -> str:
    """
    OCR an image using pytesseract if available, otherwise return empty string.

    Parameters
    ----------
    image_input : path | bytes | PIL.Image.Image
        The input image path, raw bytes, or PIL image.
    lang : str
        Language code(s) for tesseract (e.g., "eng", "por", "eng+por").
    psm : Optional[int]
        Page segmentation mode override (e.g., 6 for single uniform block).

    Returns
    -------
    str
        Extracted text (normalized to NFC). Empty string if OCR is unavailable or fails.
    """
    if not (_PIL and _TESSERACT):
        return ""

    try:
        if isinstance(image_input, str):
            im = Image.open(image_input)
        elif isinstance(image_input, bytes):
            im = Image.open(io.BytesIO(image_input))
        else:
            im = image_input  # assume PIL.Image
        config = ""
        if psm is not None:
            config = f"--psm {psm}"
        txt = pytesseract.image_to_string(im, lang=lang, config=config)
        return unicodedata.normalize("NFC", normalize_quotes(txt or ""))
    except Exception:
        return ""


def ocr_pdf(
    pdf_path: str,
    lang: str = "eng",
    dpi: int = 300,
    max_pages: Optional[int] = None,
    psm: Optional[int] = None,
) -> str:
    """
    OCR a PDF by rasterizing pages (requires pdf2image + poppler) then applying OCR.

    Parameters
    ----------
    pdf_path : str
        Path to PDF.
    lang : str
        OCR languages (e.g., "eng+por").
    dpi : int
        Rasterization resolution.
    max_pages : Optional[int]
        If set, only the first N pages are processed.
    psm : Optional[int]
        Tesseract PSM override.

    Returns
    -------
    str
        Concatenated OCR text for processed pages. Empty string if unsupported or failure.
    """
    if not (_PDF2IMAGE and _PIL and _TESSERACT):
        return ""

    try:
        pages = convert_from_path(pdf_path, dpi=dpi)
        if max_pages is not None:
            pages = pages[: max_pages]
        chunks = []
        for im in pages:
            config = ""
            if psm is not None:
                config = f"--psm {psm}"
            txt = pytesseract.image_to_string(im, lang=lang, config=config)
            chunks.append(normalize_quotes(txt))
        return unicodedata.normalize("NFC", "\n".join(filter(None, chunks)))
    except Exception:
        return ""


# --------------------------------------------------------------------------------------
# Cleaning, Normalization, Tokenization
# --------------------------------------------------------------------------------------

def normalize_quotes(text: str) -> str:
    """
    Normalize smart quotes/dashes to ASCII equivalents.
    """
    if not text:
        return ""
    out = []
    for ch in text:
        out.append(_PUNC_MAP.get(ch, ch))
    return "".join(out)


def strip_accents(text: str) -> str:
    """
    Remove accents via unicode decomposition.

    >>> strip_accents("várzea")
    'varzea'
    """
    if not text:
        return text
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def normalize_whitespace(text: str) -> str:
    """
    Collapse multiple whitespace to single spaces and trim ends.

    >>> normalize_whitespace("  hello   world \\n ")
    'hello world'
    """
    if not text:
        return text
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str, lower: bool = False, remove_accents: bool = False) -> str:
    """
    Basic text normalization pipeline (quotes, NFC, accents, whitespace, optional lower).
    """
    if not text:
        return ""
    t = normalize_quotes(text)
    t = unicodedata.normalize("NFC", t)
    if remove_accents:
        t = strip_accents(t)
    t = normalize_whitespace(t)
    if lower:
        t = t.lower()
    return t


def simple_tokens(text: str) -> List[str]:
    """
    Regex-based alpha tokenizer (Unicode-aware; retains diacritics).
    """
    if not text:
        return []
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text)


def split_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter (no external NLP).
    """
    if not text:
        return []
    # First normalize whitespace and quotes to reduce odd splits
    norm = normalize_whitespace(normalize_quotes(text))
    parts = _SENT_PAT.split(norm)
    # Join small fragments that were likely false boundaries
    out: List[str] = []
    for s in parts:
        s = s.strip()
        if not s:
            continue
        out.append(s)
    return out


def split_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs by blank lines.
    """
    if not text:
        return []
    # Normalize newlines; split on 2+ newlines
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n{2,}", t)
    return [normalize_whitespace(b) for b in blocks if b.strip()]


# --------------------------------------------------------------------------------------
# Keyword / Keyphrase Extraction (lightweight)
# --------------------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    # Simple alpha tokenizer, keeps diacritics for readability
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", text)


def simple_keywords(
    text: str,
    top_k: int = 20,
    min_len: int = 3,
    stopwords: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Lightweight RAKE-like keyword extraction without external deps.

    Heuristics:
      • Split to tokens, filter stopwords, score by (frequency * position_weight)
      • Merge adjacent non-stopword tokens as candidate phrases
      • Return unique candidates by score
    """
    sw = set(stopwords) if stopwords is not None else _STOPWORDS
    if not text:
        return []

    raw = unicodedata.normalize("NFC", text)
    tokens = _tokenize(strip_accents(raw).lower())
    if not tokens:
        return []

    # frequency + early-position weighting (earlier terms get slight boost)
    freq: Dict[str, float] = {}
    n = len(tokens)
    for i, tok in enumerate(tokens):
        if len(tok) < min_len or tok in sw:
            continue
        pos_w = 1.0 + 0.5 * (1.0 - (i / max(1, n - 1)))  # 1.5 at start → 1.0 at end
        freq[tok] = freq.get(tok, 0.0) + pos_w

    # phrase formation from original (preserves capitalization)
    phrases: Dict[str, float] = {}

    def _add_phrase(ph: str, score: float):
        ph_norm = normalize_whitespace(ph)
        if len(strip_accents(ph_norm)) >= min_len:
            phrases[ph_norm] = max(phrases.get(ph_norm, 0.0), score)

    words = re.findall(r"\b[\wÀ-ÖØ-öø-ÿ'’-]+\b", raw)
    curr, score_acc = [], 0.0
    for w in words:
        w_clean = strip_accents(w).lower()
        if len(w_clean) >= min_len and w_clean not in sw and re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", w):
            curr.append(w)
            score_acc += freq.get(w_clean, 0.0)
        else:
            if curr:
                _add_phrase(" ".join(curr), score_acc if score_acc > 0 else len(curr))
                curr, score_acc = [], 0.0
    if curr:
        _add_phrase(" ".join(curr), score_acc if score_acc > 0 else len(curr))

    if not phrases:
        ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
        return [k for k, _ in ranked[:top_k]]

    ranked_phrases = sorted(phrases.items(), key=lambda kv: kv[1], reverse=True)
    return [p for p, _ in ranked_phrases[:top_k]]


# --------------------------------------------------------------------------------------
# Coordinates
# --------------------------------------------------------------------------------------

def _dms_to_decimal(deg: int, minute: int, sec: float, hemi: str) -> float:
    sgn = -1.0 if hemi.upper() in ("S", "W") else 1.0
    return sgn * (abs(deg) + minute / 60.0 + sec / 3600.0)


def parse_coordinates(text: str) -> List[Tuple[float, float]]:
    """
    Parse decimal, strict DMS, and loose DMS coordinates from text.

    Returns
    -------
    List[Tuple[float, float]]
        List of (lat, lon) in decimal degrees, filtered for valid ranges and deduplicated.
    """
    if not text:
        return []
    coords: List[Tuple[float, float]] = []

    # Decimal pairs
    for m in _DEC_COORD_PAT.finditer(text):
        try:
            lat = float(m.group("lat"))
            lon = float(m.group("lon"))
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                coords.append((lat, lon))
        except Exception:
            continue

    # Strict DMS patterns
    for m in _DMS_PAT.finditer(text):
        try:
            lat = _dms_to_decimal(
                int(m.group("lat_deg")), int(m.group("lat_min")), float(m.group("lat_sec")), m.group("lat_dir")
            )
            lon = _dms_to_decimal(
                int(m.group("lon_deg")), int(m.group("lon_min")), float(m.group("lon_sec")), m.group("lon_dir")
            )
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                coords.append((lat, lon))
        except Exception:
            continue

    # Loose DMS patterns
    for m in _LOOSE_DMS_PAT.finditer(text):
        try:
            lat = _dms_to_decimal(
                int(m.group("lat_deg")), int(m.group("lat_min")), float(m.group("lat_sec")), m.group("lat_dir")
            )
            lon = _dms_to_decimal(
                int(m.group("lon_deg")), int(m.group("lon_min")), float(m.group("lon_sec")), m.group("lon_dir")
            )
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                coords.append((lat, lon))
        except Exception:
            continue

    # Deduplicate with small tolerance
    dedup: List[Tuple[float, float]] = []
    seen: set[Tuple[int, int]] = set()
    for lat, lon in coords:
        key = (int(round(lat * 10000)), int(round(lon * 10000)))
        if key not in seen:
            seen.add(key)
            dedup.append((lat, lon))
    return dedup


# --------------------------------------------------------------------------------------
# Entities
# --------------------------------------------------------------------------------------

def _extract_dates(text: str) -> List[str]:
    return sorted(set(_DATE_PAT.findall(text)))


def _extract_hydro_terms(text: str) -> List[str]:
    if not text:
        return []
    found = set()
    low = strip_accents(text).lower()
    for term in _HYDRO_TERMS:
        if term in low:
            found.add(term)
    return sorted(found)


def toponym_candidates(text: str, max_k: int = 50) -> List[str]:
    """
    Heuristic toponym candidates:
      • Title-cased multi-word sequences
      • Filter stopwords
      • Deduplicate & truncate
    """
    if not text:
        return []
    cands: List[str] = []
    for m in _TOPONYM_PAT.finditer(text):
        tok = m.group(1).strip()
        if strip_accents(tok.lower()) not in _STOPWORDS:
            cands.append(tok)
    # Dedup preserving order
    seen: set[str] = set()
    out: List[str] = []
    for t in cands:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= max_k:
            break
    return out


def extract_entities(
    text: str,
    use_spacy: bool = False,
    spacy_model: str = "xx_ent_wiki_sm",
    keyword_top_k: int = 20,
) -> TextEntityResult:
    """
    Extract lightweight entities: dates, coordinates, hydro terms, toponyms, keywords.
    """
    t_norm = unicodedata.normalize("NFC", text or "")
    dates = _extract_dates(t_norm)
    coords = parse_coordinates(t_norm)
    hydro = _extract_hydro_terms(t_norm)
    topo = toponym_candidates(t_norm)
    keys = simple_keywords(t_norm, top_k=keyword_top_k)

    # Optional spaCy NER to enrich toponyms
    if use_spacy and _SPACY:
        try:
            nlp = spacy.load(spacy_model)
            doc = nlp(t_norm)
            extra = [ent.text for ent in doc.ents if ent.label_.upper() in {"GPE", "LOC", "FAC"}]
            # merge
            s = set(topo)
            for e in extra:
                if e not in s:
                    topo.append(e)
                    s.add(e)
        except Exception:
            pass

    return TextEntityResult(dates=dates, coordinates=coords, hydro_terms=hydro, toponyms=topo, keywords=keys)


# --------------------------------------------------------------------------------------
# Gazetteer Matching (exact + fuzzy) with spans
# --------------------------------------------------------------------------------------

def match_gazetteer(
    text: str,
    gazetteer_names: Sequence[str],
    fuzzy: bool = True,
    min_ratio: float = 0.85,
) -> List[Tuple[str, str, float]]:
    """
    Match candidate toponyms in text against a gazetteer list.

    Returns a list of (candidate_string, matched_gazetteer, score).
    """
    cands = toponym_candidates(text)
    if not gazetteer_names or not cands:
        return []

    # Pre-normalize gazetteer
    def _norm(s: str) -> str:
        return normalize_whitespace(strip_accents(s).lower())

    norm_gaz = [_norm(g) for g in gazetteer_names]

    results: List[Tuple[str, str, float]] = []

    if not fuzzy:
        gaz_map = {g: orig for g, orig in zip(norm_gaz, gazetteer_names)}
        for c in cands:
            n = _norm(c)
            if n in gaz_map:
                results.append((c, gaz_map[n], 1.0))
        return results

    # Fuzzy via difflib
    try:
        import difflib

        for c in cands:
            n = _norm(c)
            best = difflib.get_close_matches(n, norm_gaz, n=1, cutoff=min_ratio)
            if best:
                idx = norm_gaz.index(best[0])
                score = difflib.SequenceMatcher(a=n, b=best[0]).ratio()
                results.append((c, gazetteer_names[idx], float(score)))
    except Exception:
        # fallback to exact only
        return match_gazetteer(text, gazetteer_names, fuzzy=False)

    # Deduplicate by matched gazetteer, keep highest score
    best_map: Dict[str, Tuple[str, str, float]] = {}
    for triple in results:
        _, g, s = triple
        if (g not in best_map) or (s > best_map[g][2]):
            best_map[g] = triple
    return list(best_map.values())


# --------------------------------------------------------------------------------------
# Context windows & coordinate redaction
# --------------------------------------------------------------------------------------

def extract_context_windows(
    text: str,
    spans: Sequence[Tuple[int, int]],
    radius: int = 80,
) -> List[str]:
    """
    Extract context windows around character spans (e.g., gazetteer or coordinate matches).

    Parameters
    ----------
    text : str
        Source text.
    spans : Sequence[Tuple[int, int]]
        Character (start, end) positions (Python slicing convention).
    radius : int
        Number of characters to include on each side.

    Returns
    -------
    List[str]
        Extracted windows (whitespace-normalized).
    """
    if not text or not spans:
        return []
    windows: List[str] = []
    n = len(text)
    for s, e in spans:
        s2 = max(0, s - radius)
        e2 = min(n, e + radius)
        windows.append(normalize_whitespace(text[s2:e2]))
    return windows


def round_coordinates(
    coords: Sequence[Tuple[float, float]],
    lat_decimals: int = 2,
    lon_decimals: int = 2,
) -> List[Tuple[float, float]]:
    """
    Round coordinates to coarse precision (e.g., ~1 km) for ethical releases.

    Note: 0.01° ~ 1.11 km at equator for lat; lon distance varies with latitude.
    """
    out: List[Tuple[float, float]] = []
    for lat, lon in coords:
        out.append((round(lat, lat_decimals), round(lon, lon_decimals)))
    return out


def redact_coordinates(text: str, replacement: str = "[COORD]") -> str:
    """
    Redact (mask) coordinates in text. Useful for public outputs where precise locations
    should not be disclosed without consent.
    """
    if not text:
        return ""
    t = _DEC_COORD_PAT.sub(replacement, text)
    t = _DMS_PAT.sub(replacement, t)
    t = _LOOSE_DMS_PAT.sub(replacement, t)
    return t


# --------------------------------------------------------------------------------------
# Module Self-Test (optional)
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple self-checks (safe to run in Kaggle/CI)
    sample = """
    Report from 1912 notes: “Black soil near the left bank of Rio Negro at approx. 03°07'12"S, 060°01'30"W”.
    Another reference: -3.12, -60.02 around the same lagoon. Expedition: 05/07/1912.
    Also noted: 03 07 12 S; 060 30 45 W near the floodplain. See discussion in Manaus.
    """

    print("CLEAN:", clean_text(sample)[:120], "...")
    print("TOKENS:", simple_tokens(sample)[:10])
    print("SENTS:", split_sentences(sample))

    coords = parse_coordinates(sample)
    print("COORDS:", coords)
    print("COORDS(rounded):", round_coordinates(coords, 2, 2))
    print("REDACT:", redact_coordinates(sample)[:160], "...")

    ent = extract_entities(sample, use_spacy=False)
    print("ENTITIES:", ent)

    gaz = ["Rio Negro", "Rio Solimões", "Manaus", "Itacoatiara"]
    matches = match_gazetteer(sample, gaz)
    print("GAZ MATCH:", matches)

    # Example context windows around toponym spans (demo by searching)
    spans = []
    for m in re.finditer(r"Rio Negro|Manaus", sample):
        spans.append((m.start(), m.end()))
    ctx = extract_context_windows(sample, spans, radius=40)
    print("CONTEXT WINDOWS:", ctx)
