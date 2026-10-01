"""Load the SROIE dataset and prepare it for the pipeline.

- Loads the correct answers (company, date, address, total) for every receipt
  from data/raw/sroie/key/*.json.
- Finds each receipt's photo in data/raw/sroie/img/.
- Groups receipts by vendor, so spelling variants of one shop count as one vendor.
- Splits receipts into train / val / test by vendor, not by receipt, so a shop
  seen in training never appears in testing (otherwise the model just memorizes
  that shop's layout). Saved to data/splits.json.
- Builds the list of known vendors (a stand-in for the client's ERP supplier
  list) that the validator checks company names against. Saved to data/vendors.json.
"""

import json
import random
import re
from collections import defaultdict

from .config import IMG_DIR, KEY_DIR, SPLITS_PATH, VENDORS_PATH
from .schema import Receipt

SPLIT_FRACTIONS = {"test": 0.15, "val": 0.15}  # the rest is train
SEED = 42

_LEGAL_SUFFIXES = re.compile(r"\b(SDN|BHD|S/B|SB|ENTERPRISE|TRADING)\b")


def vendor_key(company: str) -> str:
    """Normalize a company name so variants of one vendor group together."""
    name = company.upper()
    name = _LEGAL_SUFFIXES.sub(" ", name)
    name = re.sub(r"[^A-Z0-9 ]", "", name)
    return " ".join(name.split())


def load_labels() -> dict[str, Receipt]:
    labels = {}
    for path in sorted(KEY_DIR.glob("*.json")):
        raw = json.loads(path.read_text())
        labels[path.stem] = Receipt(**{f: raw.get(f) for f in Receipt.model_fields})
    return labels


def image_path(doc_id: str):
    return IMG_DIR / f"{doc_id}.jpg"


def make_splits() -> dict[str, list[str]]:
    """Assign whole vendors to test/val/train until each hits its target size."""
    labels = load_labels()
    by_vendor: dict[str, list[str]] = defaultdict(list)
    for doc_id, receipt in labels.items():
        by_vendor[vendor_key(receipt.company or "")].append(doc_id)

    vendors = sorted(by_vendor)
    random.Random(SEED).shuffle(vendors)

    total = len(labels)
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    targets = {name: int(frac * total) for name, frac in SPLIT_FRACTIONS.items()}
    for vendor in vendors:
        docs = by_vendor[vendor]
        dest = next(
            (name for name in ("test", "val") if len(splits[name]) + len(docs) <= targets[name]),
            "train",
        )
        splits[dest].extend(docs)

    for docs in splits.values():
        docs.sort()
    SPLITS_PATH.write_text(json.dumps(splits, indent=2))
    return splits


def load_splits() -> dict[str, list[str]]:
    if not SPLITS_PATH.exists():
        return make_splits()
    return json.loads(SPLITS_PATH.read_text())


def build_vendor_master() -> list[str]:
    """Stand-in for the client's ERP vendor master list.

    In production this comes from the ERP (every supplier the company pays).
    Here we simulate it with every labeled company name. It is a reference
    list for the validator, not training data for the extractor.
    """
    names = sorted({r.company for r in load_labels().values() if r.company})
    VENDORS_PATH.write_text(json.dumps(names, indent=2))
    return names


def load_vendor_master() -> list[str]:
    if not VENDORS_PATH.exists():
        return build_vendor_master()
    return json.loads(VENDORS_PATH.read_text())
