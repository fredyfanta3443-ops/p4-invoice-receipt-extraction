"""Mock ERP: accepts invoice entries, deduplicated by Idempotency-Key.

Run with: uv run uvicorn invoice_extraction.erp.server:app

- First POST with a key creates the entry (201).
- Replaying the same key with the same body returns the original entry (200),
  so a client retry after a timeout never creates a duplicate.
- Reusing a key with a different body is a client bug (409).
"""

import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime

from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel

from ..config import DATA_DIR

DB_PATH = DATA_DIR / "erp.sqlite"


class InvoiceIn(BaseModel):
    doc_id: str
    vendor: str
    invoice_date: str  # ISO date
    address: str | None
    total: float


class InvoiceEntry(InvoiceIn):
    entry_id: str
    created_at: str


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS invoices (
            idempotency_key TEXT PRIMARY KEY,
            body_hash TEXT NOT NULL,
            entry TEXT NOT NULL
        )"""
    )
    return conn


def _body_hash(invoice: InvoiceIn) -> str:
    return hashlib.sha256(invoice.model_dump_json().encode()).hexdigest()


app = FastAPI(title="Mock ERP")


@app.post("/invoices", response_model=InvoiceEntry, status_code=201)
def create_invoice(
    invoice: InvoiceIn,
    response: Response,
    idempotency_key: str = Header(..., alias="Idempotency-Key"),
) -> InvoiceEntry:
    body_hash = _body_hash(invoice)
    with _connect() as conn:
        row = conn.execute(
            "SELECT body_hash, entry FROM invoices WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        if row:
            if row["body_hash"] != body_hash:
                raise HTTPException(409, "Idempotency-Key reused with a different body")
            response.status_code = 200
            return InvoiceEntry.model_validate_json(row["entry"])

        entry = InvoiceEntry(
            **invoice.model_dump(),
            entry_id=str(uuid.uuid4()),
            created_at=datetime.now(UTC).isoformat(),
        )
        # The primary key makes this safe under concurrent retries: a racing
        # duplicate insert fails instead of creating a second entry.
        try:
            conn.execute(
                "INSERT INTO invoices VALUES (?, ?, ?)",
                (idempotency_key, body_hash, entry.model_dump_json()),
            )
        except sqlite3.IntegrityError:
            response.status_code = 200
            row = conn.execute(
                "SELECT entry FROM invoices WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            return InvoiceEntry.model_validate_json(row["entry"])
        return entry


@app.get("/invoices", response_model=list[InvoiceEntry])
def list_invoices() -> list[InvoiceEntry]:
    with _connect() as conn:
        rows = conn.execute("SELECT entry FROM invoices").fetchall()
    return [InvoiceEntry(**json.loads(r["entry"])) for r in rows]
