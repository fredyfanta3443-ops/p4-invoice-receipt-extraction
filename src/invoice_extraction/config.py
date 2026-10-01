"""Paths and tunable settings, all in one place."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "sroie"
IMG_DIR = RAW_DIR / "img"
KEY_DIR = RAW_DIR / "key"
OCR_DIR = DATA_DIR / "ocr"
SPLITS_PATH = DATA_DIR / "splits.json"
VENDORS_PATH = DATA_DIR / "vendors.json"
RUNS_DIR = DATA_DIR / "runs"

# LLM: "anthropic" (Claude) or "groq"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
MODEL = os.getenv("EXTRACT_MODEL", "claude-opus-5-5")
EFFORT = os.getenv("EXTRACT_EFFORT", "low")
GROQ_MODEL = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")
FEW_SHOT_K = 3

# Routing: auto-post only if every validator passes AND the lowest
# field confidence is at or above this threshold.
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.9"))

# Validators
VENDOR_MATCH_MIN_SCORE = 90  # rapidfuzz score, 0-100
MAX_PLAUSIBLE_TOTAL = 100_000.0

# Mock ERP
ERP_URL = os.getenv("ERP_URL", "http://127.0.0.1:8000")
