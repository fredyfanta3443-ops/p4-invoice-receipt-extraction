"""Deterministic checks on an extraction. Any failure sends the document to review."""

import re
from datetime import date, datetime

from dateutil import parser as date_parser
from rapidfuzz import fuzz, process

from .config import MAX_PLAUSIBLE_TOTAL, VENDOR_MATCH_MIN_SCORE
from .schema import Receipt, ValidationResult

EARLIEST_PLAUSIBLE_DATE = date(2000, 1, 1)


def parse_receipt_date(value: str) -> date:
    """SROIE receipts are Malaysian, so day comes before month (25/12/2018)."""
    return date_parser.parse(value, dayfirst=True, fuzzy=True).date()


def parse_total(value: str) -> float:
    cleaned = re.sub(r"[^\d.,-]", "", value)
    # "9,00" is a decimal comma (common OCR misread of "9.00"); "1,250.00" is thousands.
    if re.fullmatch(r"-?\d+,\d{2}", cleaned):
        cleaned = cleaned.replace(",", ".")
    return float(cleaned.replace(",", ""))


def check_date(receipt: Receipt) -> ValidationResult:
    if not receipt.date:
        return ValidationResult(name="date", passed=False, detail="missing")
    try:
        parsed = parse_receipt_date(receipt.date)
    except (ValueError, OverflowError):
        return ValidationResult(name="date", passed=False, detail=f"unparseable: {receipt.date!r}")
    if not EARLIEST_PLAUSIBLE_DATE <= parsed <= datetime.now().date():
        return ValidationResult(name="date", passed=False, detail=f"out of range: {parsed}")
    return ValidationResult(name="date", passed=True, detail=parsed.isoformat())


def check_total(receipt: Receipt) -> ValidationResult:
    if not receipt.total:
        return ValidationResult(name="total", passed=False, detail="missing")
    try:
        amount = parse_total(receipt.total)
    except ValueError:
        return ValidationResult(name="total", passed=False, detail=f"not numeric: {receipt.total!r}")
    if not 0 < amount <= MAX_PLAUSIBLE_TOTAL:
        return ValidationResult(name="total", passed=False, detail=f"implausible amount: {amount}")
    return ValidationResult(name="total", passed=True, detail=f"{amount:.2f}")


def check_vendor(receipt: Receipt, vendors: list[str]) -> ValidationResult:
    if not receipt.company:
        return ValidationResult(name="vendor", passed=False, detail="missing")
    match = process.extractOne(receipt.company.upper(), vendors, scorer=fuzz.token_sort_ratio,
                               processor=str.upper)
    if match is None or match[1] < VENDOR_MATCH_MIN_SCORE:
        best = f"{match[0]!r} ({match[1]:.0f})" if match else "none"
        return ValidationResult(name="vendor", passed=False, detail=f"unknown vendor, best match {best}")
    return ValidationResult(name="vendor", passed=True, detail=f"{match[0]} ({match[1]:.0f})")


def run_validators(receipt: Receipt, vendors: list[str]) -> list[ValidationResult]:
    return [check_date(receipt), check_total(receipt), check_vendor(receipt, vendors)]
