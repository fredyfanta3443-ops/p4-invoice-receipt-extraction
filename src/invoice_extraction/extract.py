"""Turn OCR text into structured fields with an LLM (Claude or Groq).

Few-shot examples come from the training split only, and they sit in the
system prompt so the whole prefix is cached across documents.
"""

import anthropic
import groq
from pydantic import ValidationError

from .config import EFFORT, FEW_SHOT_K, GROQ_MODEL, LLM_PROVIDER, MODEL
from .data import image_path, load_labels, load_splits, vendor_key
from .ocr import ocr_cached
from .schema import Extraction

INSTRUCTIONS = """\
You extract fields from OCR text of scanned retail receipts. The OCR is noisy: \
characters can be misread (0/O, 1/l, 5/S), digits can be dropped, and lines can be out of order.

Return these fields:
- company: the seller's name as printed at the top of the receipt.
- date: the transaction date as printed (keep the receipt's own format).
- address: the seller's address, lines joined with single spaces.
- total: the final amount the customer paid, digits and a decimal point only (e.g. 9.00). \
Prefer the rounded/net total over subtotals, tax lines or cash tendered.

Correct obvious OCR errors when the context makes the right value clear. \
If a field is missing or unreadable, return null for it.

For each field also give a confidence between 0 and 1: the probability that your value \
is exactly right. Be calibrated. Use a low score when the OCR is garbled, when several \
candidates compete (e.g. two different totals), or when you had to guess a digit. \
A confidence of 0.9 should be right about 90% of the time."""


class ExtractionError(Exception):
    """The LLM gave no usable extraction; the document goes to human review."""


def _few_shot_ids(k: int) -> list[str]:
    """First receipt from each of the first k distinct training vendors."""
    labels = load_labels()
    picked, seen = [], set()
    for doc_id in load_splits()["train"]:
        vendor = vendor_key(labels[doc_id].company or "")
        if vendor not in seen:
            seen.add(vendor)
            picked.append(doc_id)
        if len(picked) == k:
            break
    return picked


def build_system_prompt(k: int = FEW_SHOT_K) -> list[dict]:
    labels = load_labels()
    examples = []
    for i, doc_id in enumerate(_few_shot_ids(k), 1):
        ocr_text = ocr_cached(doc_id, image_path(doc_id))
        examples.append(
            f"<example index=\"{i}\">\n<ocr>\n{ocr_text}\n</ocr>\n"
            f"<fields>\n{labels[doc_id].model_dump_json(indent=2)}\n</fields>\n</example>"
        )
    return [
        {"type": "text", "text": INSTRUCTIONS},
        {
            "type": "text",
            "text": "<examples>\n" + "\n".join(examples) + "\n</examples>",
            "cache_control": {"type": "ephemeral"},
        },
    ]


class ClaudeExtractor:
    def __init__(self, client: anthropic.Anthropic | None = None):
        self.client = client or anthropic.Anthropic()
        self.system = build_system_prompt()

    def extract(self, ocr_text: str) -> tuple[Extraction, dict]:
        """Returns the extraction and token usage for cost tracking."""
        response = self.client.beta.messages.parse(
            model=MODEL,
            max_tokens=4096,
            system=self.system,
            messages=[{"role": "user", "content": f"<ocr>\n{ocr_text}\n</ocr>"}],
            output_format=Extraction,
            output_config={"effort": EFFORT},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise ExtractionError(f"model refused (request {response._request_id})")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            raise ExtractionError(f"no parsable output, stop_reason={response.stop_reason}")

        usage = response.usage
        return response.parsed_output, {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_input_tokens": usage.cache_read_input_tokens or 0,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens or 0,
        }


class GroqExtractor:
    def __init__(self, client: groq.Groq | None = None):
        self.client = client or groq.Groq()
        self.system = "\n\n".join(block["text"] for block in build_system_prompt())

    def extract(self, ocr_text: str) -> tuple[Extraction, dict]:
        """Returns the extraction and token usage for cost tracking."""
        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": f"<ocr>\n{ocr_text}\n</ocr>"},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "extraction", "schema": Extraction.model_json_schema()},
            },
            reasoning_effort=EFFORT,
        )
        choice = response.choices[0]
        if choice.finish_reason != "stop" or not choice.message.content:
            raise ExtractionError(f"no usable output, finish_reason={choice.finish_reason}")
        try:
            extraction = Extraction.model_validate_json(choice.message.content)
        except ValidationError as exc:
            raise ExtractionError(f"output did not match schema: {exc}") from exc

        return extraction, {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }


Extractor = ClaudeExtractor | GroqExtractor


def make_extractor() -> Extractor:
    if LLM_PROVIDER == "groq":
        return GroqExtractor()
    return ClaudeExtractor()
