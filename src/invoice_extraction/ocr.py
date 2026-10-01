"""OCR with Tesseract.

The preprocessing here must be identical in evaluation and production
(train-serve skew), so every caller goes through `ocr_image`. The engine
version is stored next to each cached result so a Tesseract upgrade
invalidates the cache instead of silently mixing outputs.
"""

import json
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps

from .config import OCR_DIR

TESSERACT_CONFIG = "--oem 1 --psm 4"  # LSTM engine; single column of variable-size text


def engine_version() -> str:
    return str(pytesseract.get_tesseract_version())


def preprocess(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    return ImageOps.grayscale(image)


def ocr_image(path: Path) -> str:
    with Image.open(path) as image:
        return pytesseract.image_to_string(preprocess(image), config=TESSERACT_CONFIG).strip()


def ocr_cached(doc_id: str, path: Path) -> str:
    """OCR once per document and engine version; reuse the text after that."""
    cache = OCR_DIR / f"{doc_id}.json"
    version = engine_version()
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached["engine"] == version and cached["config"] == TESSERACT_CONFIG:
            return cached["text"]

    text = ocr_image(path)
    OCR_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"engine": version, "config": TESSERACT_CONFIG, "text": text}))
    return text
