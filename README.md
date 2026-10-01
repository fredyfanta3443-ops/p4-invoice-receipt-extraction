# Invoice and receipt extraction with confidence routing

Reads scanned receipts, extracts company / date / address / total, and either
auto-posts them to an ERP or sends them to human review. See
[PROJECT_BRIEF.md](PROJECT_BRIEF.md) for the problem and approach.

## Pipeline

```
image -> OCR (Tesseract) -> LLM extraction (Claude, Pydantic schema, per-field confidence)
      -> validators (date parses, total plausible, vendor fuzzy-matches master list)
      -> route: all validators pass AND min confidence >= threshold ? auto-post : review
      -> mock ERP (FastAPI), Idempotency-Key = sha256(image bytes)
```

| Module | Role |
|---|---|
| `data.py` | Load SROIE labels, split **by vendor** (no vendor in two splits), vendor master list |
| `ocr.py` | Tesseract with fixed preprocessing; cache keyed on engine version |
| `extract.py` | Claude extraction; few-shot examples from the train split only |
| `validate.py` | Date, total and vendor checks |
| `route.py` | Auto-post vs review decision |
| `erp/server.py`, `erp/client.py` | Mock ERP with idempotent POST; client with safe retries |
| `pipeline.py`, `cli.py` | End-to-end run per document |

## Setup

```bash
brew install tesseract
uv sync
```

Create a `.env` file (gitignored) for the LLM provider:

```bash
# Groq
LLM_PROVIDER=groq
GROQ_API_KEY=...
GROQ_CHAT_MODEL=openai/gpt-oss-20b

# or Claude
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=...
```

Data: SROIE images and key files in `data/raw/sroie/{img,key}` (from
github.com/zzzDavid/ICDAR-2019-SROIE, `data/img` and `data/key`).

## Run

```bash
uv run invoice-extraction prepare                  # splits + vendor list
uv run invoice-extraction ocr --split test         # cache OCR text
uv run invoice-extraction erp                      # terminal 1: mock ERP on :8000
uv run invoice-extraction run --split test -n 20   # terminal 2: full pipeline
```

Settings (env vars): `LLM_PROVIDER` (`anthropic` or `groq`), `EXTRACT_MODEL` (default `claude-opus-5-5`), `EXTRACT_EFFORT`
(default `low`), `CONFIDENCE_THRESHOLD` (default `0.9`), `ERP_URL`.
