# P04: Invoice and Receipt Extraction with Confidence Routing

## 1. The problem we're actually solving
- A finance team types invoice fields into the ERP by hand, which is slow and causes errors.
- The backlog delays payments, and the company pays late fees because of it.
- The request is "automate invoice entry", but the real goal is to **auto-process as many documents as possible while keeping the error rate on auto-posted documents under an agreed limit**.
- Nobody checks auto-posted documents, so their errors stay hidden unless we measure them on purpose.
- The client needs a clear answer to: *"What's the error rate on invoices nobody reviews?"*

## 2. The solution we'll provide (non-technical)
- A system that reads receipts and invoices and fills in the key fields: vendor, date, address and total.
- It enters the documents it's confident about into the ERP automatically and sends uncertain ones to a person to review.
- A single setting lets the business choose how much gets automated versus how much risk it accepts.
- The system is safe to retry, so an invoice is never entered twice.
- Reporting shows the audited error rate, time saved per document, and cost savings per month.

## 3. How we'll approach it (technical)
- **Extraction:** Tesseract OCR, then an LLM extracts the fields into a Pydantic schema. Few-shot examples come from the training split only. The data is SROIE, with CORD as a stretch goal for line items.
- **Baseline and validators:** templates and regex rules for the top vendors, with the LLM handling the long tail. Checks confirm the date parses, the total is numeric and plausible, and the company fuzzy-matches a vendor list.
- **Routing and posting:** a document is auto-posted only when the validators pass and confidence is above a threshold; everything else goes to a review queue. Posting goes to a mock FastAPI ERP endpoint with idempotency keys.
- **Evaluation:** field-level exact and normalized match, plus the rate of documents with every field correct. The key chart is straight-through rate vs. auto-post error rate as the threshold moves, along with cost and latency per document.
- **Production checks:**
  - Split the data by vendor, not by document, so the system can't memorize a vendor's layout.
  - Use the same OCR engine, version and image preprocessing in evaluation and production.
  - Monitor the validator failure rate per vendor to catch new vendors and layouts.
  - Treat human corrections from the review queue as new labels.
  - Audit a random 2% of auto-posted documents and report that error rate with a confidence interval.
  - Run as a micro-batch every 15 minutes or on each upload.
