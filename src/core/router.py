"""
Layer 1 — Router / Classifier

Analyses the first ~3 000 characters of any document (contract or invoice)
and returns a DocumentContext with jurisdiction, language, currency and
document type.  The call is intentionally cheap: short prompt, low token
budget, no deep parsing.

Fallback contract
-----------------
* If router_confidence < ROUTER_CONFIDENCE_THRESHOLD the method raises
  LowConfidenceError so the Orchestrator can request human intervention
  rather than proceeding with an unreliable classification.
* If the LLM returns an unexpected value for jurisdiction/currency the
  field is set to "UNKNOWN" and confidence is penalised accordingly.
"""

from __future__ import annotations

import os
from typing import Literal

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, model_validator

load_dotenv()

# Only the first N characters are sent to the router — keeps the call fast.
_ROUTER_PREVIEW_CHARS = 3_000

# Below this threshold the Orchestrator must not proceed automatically.
ROUTER_CONFIDENCE_THRESHOLD = 0.75

_VALID_JURISDICTIONS = {"IT", "CH"}
_VALID_CURRENCIES = {"EUR", "CHF"}
_VALID_DOC_TYPES = {"CONTRACT", "INVOICE"}


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class DocumentContext(BaseModel):
    jurisdiction: Literal["IT", "CH", "UNKNOWN"]
    language: str                                      # e.g. "it", "fr", "de", "en"
    currency: Literal["EUR", "CHF", "UNKNOWN"]
    document_type: Literal["CONTRACT", "INVOICE", "UNKNOWN"]
    router_confidence: float                           # 0.0 – 1.0

    @model_validator(mode="after")
    def _penalise_unknowns(self) -> "DocumentContext":
        """Each UNKNOWN field reduces the overall confidence."""
        unknowns = sum([
            self.jurisdiction == "UNKNOWN",
            self.currency == "UNKNOWN",
            self.document_type == "UNKNOWN",
        ])
        if unknowns:
            penalty = unknowns * 0.15
            self.router_confidence = max(0.0, round(self.router_confidence - penalty, 4))
        return self


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class LowConfidenceError(RuntimeError):
    """Raised when the router cannot classify the document with enough certainty."""

    def __init__(self, context: DocumentContext) -> None:
        self.context = context
        super().__init__(
            f"Router confidence too low to proceed automatically "
            f"({context.router_confidence:.0%}). "
            f"Detected: jurisdiction={context.jurisdiction}, "
            f"currency={context.currency}, "
            f"document_type={context.document_type}. "
            "Human review required."
        )


# ---------------------------------------------------------------------------
# Router tool schema
# ---------------------------------------------------------------------------

_ROUTER_TOOL = {
    "name": "classify_document",
    "description": (
        "Classify the document's jurisdiction, language, currency and type "
        "from the opening section. Return your best guess even if uncertain, "
        "but reflect uncertainty in the confidence score."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "jurisdiction": {
                "type": "string",
                "enum": ["IT", "CH", "UNKNOWN"],
                "description": (
                    "Legal jurisdiction of the document. "
                    "'IT' = Italy (uses EUR, IVA), "
                    "'CH' = Switzerland (uses CHF, MWST/TVA). "
                    "Use 'UNKNOWN' only if there is no evidence at all."
                ),
            },
            "language": {
                "type": "string",
                "description": (
                    "Primary language of the document as a 2-letter ISO code: "
                    "'it' (Italian), 'fr' (French), 'de' (German), 'en' (English)."
                ),
            },
            "currency": {
                "type": "string",
                "enum": ["EUR", "CHF", "UNKNOWN"],
                "description": (
                    "Currency symbol or code found in the document. "
                    "'EUR' or '€' → 'EUR'. 'CHF' or 'Fr.' → 'CHF'."
                ),
            },
            "document_type": {
                "type": "string",
                "enum": ["CONTRACT", "INVOICE", "UNKNOWN"],
                "description": (
                    "'CONTRACT' if this is an agreement or set of terms. "
                    "'INVOICE' if this is a billing document (fattura, Rechnung, facture)."
                ),
            },
            "confidence": {
                "type": "number",
                "description": (
                    "Your confidence in ALL of the above classifications combined, "
                    "from 0.0 (no idea) to 1.0 (absolutely certain). "
                    "Score below 0.75 means critical information is missing."
                ),
            },
        },
        "required": ["jurisdiction", "language", "currency", "document_type", "confidence"],
    },
}

_ROUTER_SYSTEM_PROMPT = (
    "You are a document classifier for a legal and financial audit system. "
    "Your only job is to identify the jurisdiction, language, currency and document type "
    "from the opening section of a document. "
    "Be concise and precise. Do NOT extract financial figures or clause details."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_document(text: str, *, raise_on_low_confidence: bool = True) -> DocumentContext:
    """
    Classify *text* and return a DocumentContext.

    Parameters
    ----------
    text:
        Full document text (only the first _ROUTER_PREVIEW_CHARS are sent).
    raise_on_low_confidence:
        If True (default) raises LowConfidenceError when router_confidence
        falls below ROUTER_CONFIDENCE_THRESHOLD.  Set to False in tests or
        when the caller wants to handle ambiguous documents itself.
    """
    preview = text[:_ROUTER_PREVIEW_CHARS]

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",   # lightweight model — fast and cheap
        max_tokens=256,
        system=_ROUTER_SYSTEM_PROMPT,
        tools=[_ROUTER_TOOL],
        tool_choice={"type": "tool", "name": "classify_document"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Classify this document based on its opening section only:\n\n"
                    + preview
                ),
            }
        ],
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    raw = tool_block.input

    # Normalise values — coerce anything unexpected to UNKNOWN
    jurisdiction = raw["jurisdiction"] if raw["jurisdiction"] in _VALID_JURISDICTIONS else "UNKNOWN"
    currency = raw["currency"] if raw["currency"] in _VALID_CURRENCIES else "UNKNOWN"
    document_type = raw["document_type"] if raw["document_type"] in _VALID_DOC_TYPES else "UNKNOWN"

    context = DocumentContext(
        jurisdiction=jurisdiction,
        language=raw.get("language", "unknown").lower()[:5],
        currency=currency,
        document_type=document_type,
        router_confidence=float(raw.get("confidence", 0.0)),
    )

    if raise_on_low_confidence and context.router_confidence < ROUTER_CONFIDENCE_THRESHOLD:
        raise LowConfidenceError(context)

    return context
