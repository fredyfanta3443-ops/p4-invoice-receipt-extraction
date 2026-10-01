"""Data contracts shared across the pipeline."""

from enum import StrEnum

from pydantic import BaseModel, Field

FIELDS = ("company", "date", "address", "total")


class Receipt(BaseModel):
    """The four SROIE key fields, as written on the receipt."""

    company: str | None = Field(description="Seller / vendor name exactly as printed")
    date: str | None = Field(description="Transaction date exactly as printed")
    address: str | None = Field(description="Seller address, lines joined with a single space")
    total: str | None = Field(description="Final amount paid, digits and decimal point only, e.g. 9.00")


class FieldConfidence(BaseModel):
    company: float = Field(ge=0, le=1)
    date: float = Field(ge=0, le=1)
    address: float = Field(ge=0, le=1)
    total: float = Field(ge=0, le=1)


class Extraction(BaseModel):
    """What the LLM returns: the fields plus how sure it is about each."""

    receipt: Receipt
    confidence: FieldConfidence


class ValidationResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class Route(StrEnum):
    AUTO_POST = "auto_post"
    REVIEW = "review"


class RoutingDecision(BaseModel):
    route: Route
    min_confidence: float
    reasons: list[str]


class DocumentResult(BaseModel):
    """Everything the pipeline knows about one processed document."""

    doc_id: str
    extraction: Extraction | None
    validations: list[ValidationResult]
    decision: RoutingDecision
    erp_entry_id: str | None = None
    error: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    latency_s: float = 0.0
