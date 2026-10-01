"""Post auto-approved documents to the ERP, retrying safely."""

import hashlib
import time
from pathlib import Path

import httpx

from ..config import ERP_URL
from ..schema import Receipt
from ..validate import parse_receipt_date, parse_total

MAX_ATTEMPTS = 4


def idempotency_key(image_path: Path) -> str:
    """Derived from the document bytes, so re-uploading or re-running the same
    receipt always maps to the same ERP entry."""
    return hashlib.sha256(image_path.read_bytes()).hexdigest()


def post_invoice(doc_id: str, receipt: Receipt, key: str, base_url: str = ERP_URL) -> str:
    """Returns the ERP entry id. Retries on network errors and 5xx with the same key."""
    payload = {
        "doc_id": doc_id,
        "vendor": receipt.company,
        "invoice_date": parse_receipt_date(receipt.date).isoformat(),
        "address": receipt.address,
        "total": parse_total(receipt.total),
    }
    headers = {"Idempotency-Key": key}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        last_attempt = attempt == MAX_ATTEMPTS
        try:
            resp = httpx.post(f"{base_url}/invoices", json=payload, headers=headers, timeout=10)
        except httpx.TransportError:
            if last_attempt:
                raise
        else:
            if resp.status_code < 500 or last_attempt:
                resp.raise_for_status()
                return resp.json()["entry_id"]
        time.sleep(2 ** (attempt - 1))
    raise RuntimeError("unreachable")
