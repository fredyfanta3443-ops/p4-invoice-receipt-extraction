"""Confidence routing: auto-post only when validators pass and confidence is high."""

from .config import CONFIDENCE_THRESHOLD
from .schema import Extraction, Route, RoutingDecision, ValidationResult


def decide(
    extraction: Extraction,
    validations: list[ValidationResult],
    threshold: float = CONFIDENCE_THRESHOLD,
) -> RoutingDecision:
    # A document is only as trustworthy as its least certain field.
    min_conf = min(extraction.confidence.model_dump().values())

    reasons = [f"validator {v.name} failed: {v.detail}" for v in validations if not v.passed]
    if min_conf < threshold:
        reasons.append(f"min confidence {min_conf:.2f} < threshold {threshold:.2f}")

    route = Route.REVIEW if reasons else Route.AUTO_POST
    return RoutingDecision(route=route, min_confidence=min_conf, reasons=reasons)


def review_on_error(error: str) -> RoutingDecision:
    return RoutingDecision(route=Route.REVIEW, min_confidence=0.0, reasons=[error])
