"""One document end to end: OCR -> extract -> validate -> route -> post or queue."""

import time
from pathlib import Path

from .erp.client import idempotency_key, post_invoice
from .extract import ExtractionError, Extractor
from .ocr import ocr_cached
from .route import decide, review_on_error
from .schema import DocumentResult, Route
from .validate import run_validators


def process_document(
    doc_id: str,
    image: Path,
    extractor: Extractor,
    vendors: list[str],
    post: bool = True,
) -> DocumentResult:
    start = time.perf_counter()
    text = ocr_cached(doc_id, image)

    try:
        extraction, usage = extractor.extract(text)
    except ExtractionError as exc:
        return DocumentResult(
            doc_id=doc_id,
            extraction=None,
            validations=[],
            decision=review_on_error(str(exc)),
            error=str(exc),
            latency_s=time.perf_counter() - start,
        )

    validations = run_validators(extraction.receipt, vendors)
    decision = decide(extraction, validations)
    result = DocumentResult(
        doc_id=doc_id,
        extraction=extraction,
        validations=validations,
        decision=decision,
        usage=usage,
    )

    if post and decision.route is Route.AUTO_POST:
        try:
            result.erp_entry_id = post_invoice(doc_id, extraction.receipt, idempotency_key(image))
        except Exception as exc:  # ERP down or rejected: a human must handle it
            result.decision = review_on_error(f"ERP post failed: {exc}")
            result.error = str(exc)

    result.latency_s = time.perf_counter() - start
    return result
