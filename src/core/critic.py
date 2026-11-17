"""
Layer 3 — The Critic / Verifier Agent

Receives the original document text and the raw JSON extraction produced by a
Layer-2 regional extractor.  For every field it locates the supporting
evidence in the source text, assigns a confidence score, and — only when
absolutely certain — corrects a wrong value.  When uncertain it lowers the
confidence and leaves the original value untouched so the Orchestrator can
surface a "Needs Human Review" flag instead of silently propagating bad data.

Conflict-resolution rules (encoded in the prompt and enforced in code):
  1. Evidence found, value matches  → is_confirmed=True,  confidence 0.92–1.0
  2. Evidence found, value differs  → is_confirmed=False, confidence 0.50–0.80,
                                       corrected_value set to the correct figure
  3. No evidence at all             → is_confirmed=False, confidence 0.10–0.45,
                                       corrected_value=None  (original kept)
  4. Ambiguous / multiple mentions  → is_confirmed=False, confidence 0.45–0.65,
                                       corrected_value=None
"""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, model_validator

from src.models import CONFIDENCE_THRESHOLD

load_dotenv()

# Full document is sent to the Critic — it needs to search the entire text.
# We cap at 90 k chars to stay well inside the context window even after the
# prompt overhead from the extraction JSON.
_MAX_DOC_CHARS = 90_000

_MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

class FieldVerification(BaseModel):
    """Critic's verdict on a single extracted field."""

    field_name: str
    extracted_value: Any          # what Layer 2 produced
    corrected_value: Any = None   # only set when Critic is certain of a fix
    is_confirmed: bool            # True = evidence found and value matches
    confidence: float             # 0.0 → 1.0
    source_quote: str = ""        # verbatim excerpt from the original document
    correction_note: str = ""     # explanation when corrected_value is set
    needs_review: bool = False    # auto-set below

    @model_validator(mode="after")
    def _auto_flag(self) -> "FieldVerification":
        if self.confidence < CONFIDENCE_THRESHOLD:
            self.needs_review = True
        return self

    @property
    def final_value(self) -> Any:
        """Returns corrected_value when the Critic is certain; else extracted_value."""
        return self.corrected_value if (self.corrected_value is not None and self.is_confirmed is False and self.correction_note) else self.extracted_value


class CriticReport(BaseModel):
    """Complete output of the Critic Agent for one document extraction."""

    field_verifications: list[FieldVerification]
    silent_errors_found: list[str]   # contradictions the Critic spotted
    overall_confidence: float        # mean confidence across all fields
    critic_notes: str = ""           # free-form observations

    @model_validator(mode="after")
    def _compute_overall(self) -> "CriticReport":
        if self.field_verifications:
            scores = [fv.confidence for fv in self.field_verifications]
            self.overall_confidence = round(sum(scores) / len(scores), 4)
        return self

    @property
    def fields_needing_review(self) -> list[str]:
        return [fv.field_name for fv in self.field_verifications if fv.needs_review]

    @property
    def confirmed_fields(self) -> list[str]:
        return [fv.field_name for fv in self.field_verifications if fv.is_confirmed]


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

_CRITIC_TOOL = {
    "name": "verify_extraction",
    "description": (
        "For each field in the preliminary extraction, locate supporting "
        "evidence in the original document and return a structured verdict."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "field_verifications": {
                "type": "array",
                "description": "One entry per field in the preliminary extraction.",
                "items": {
                    "type": "object",
                    "properties": {
                        "field_name": {
                            "type": "string",
                            "description": "Exact name of the field being verified.",
                        },
                        "is_confirmed": {
                            "type": "boolean",
                            "description": (
                                "True only when the source document contains explicit "
                                "evidence that matches the extracted value exactly."
                            ),
                        },
                        "confidence": {
                            "type": "number",
                            "description": (
                                "0.92–1.0  : evidence found, value matches. "
                                "0.50–0.80 : evidence found but value differs (set corrected_value). "
                                "0.45–0.65 : ambiguous or multiple conflicting mentions. "
                                "0.10–0.45 : no evidence found at all."
                            ),
                        },
                        "source_quote": {
                            "type": "string",
                            "description": (
                                "The shortest verbatim excerpt from the document that "
                                "supports or refutes the extracted value. "
                                "Empty string if no relevant text was found."
                            ),
                        },
                        "corrected_value": {
                            "description": (
                                "The correct value found in the document. "
                                "Set ONLY when you are 100%% certain the extracted value "
                                "is wrong AND you have found the correct value in the text. "
                                "Leave null in all other cases — do NOT guess."
                            ),
                        },
                        "correction_note": {
                            "type": "string",
                            "description": (
                                "Required when corrected_value is set. "
                                "Explain concisely why the original extraction was wrong."
                            ),
                        },
                    },
                    "required": ["field_name", "is_confirmed", "confidence", "source_quote"],
                },
            },
            "silent_errors_found": {
                "type": "array",
                "description": (
                    "List any contradictions, implausible values, or logical "
                    "inconsistencies you noticed that are NOT already captured "
                    "in a specific field verification (e.g. currency mismatch "
                    "between header and body, dates that make no sense, etc.)."
                ),
                "items": {"type": "string"},
            },
            "critic_notes": {
                "type": "string",
                "description": (
                    "Optional free-form observations about document quality, "
                    "layout ambiguities, or anything that might affect audit reliability."
                ),
            },
        },
        "required": ["field_verifications", "silent_errors_found"],
    },
}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an implacable financial auditor with forensic attention to detail.
You have been given:
  A) The original source document (contract or invoice).
  B) A preliminary extraction produced by an automated system.

Your mission is to verify extraction B against document A — field by field.

RULES YOU MUST FOLLOW:
1. For each field, search the document for the single clearest piece of evidence.
2. Quote the exact phrase or number you found (source_quote). Be verbatim, not paraphrased.
3. If the document evidence matches the extracted value → is_confirmed=True, confidence ≥ 0.92.
4. If you find clear evidence of a DIFFERENT value → is_confirmed=False, set corrected_value
   to the correct figure, write a concise correction_note. Confidence 0.50–0.80.
5. If you find NO evidence at all → is_confirmed=False, corrected_value=null, confidence ≤ 0.45.
6. NEVER invent or infer a corrected_value. If you are not 100%% certain, leave it null.
7. Flag "silent errors": contradictions inside the document itself, currency mismatches,
   date anomalies, or values that are technically extracted correctly but are logically
   suspicious (e.g. a penalty rate that is 0 in a document that mentions SLA clauses).

You are the last line of defence before this data reaches an auditor.
Be rigorous. Be honest. Do not pass uncertain data as certain.\
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_extraction(
    document_text: str,
    extracted_fields: dict[str, Any],
    document_type: str = "document",
) -> CriticReport:
    """
    Verify *extracted_fields* against *document_text* and return a CriticReport.

    Parameters
    ----------
    document_text:
        Full raw text of the source document (contract or invoice).
    extracted_fields:
        Dict mapping field_name → extracted_value produced by a Layer-2 extractor.
        Example: {"base_monthly_fee": 5000.0, "vat_rate": 22.0, ...}
    document_type:
        Human-readable label used in the prompt ("contract" or "invoice").
    """
    doc_text = document_text[:_MAX_DOC_CHARS]

    extraction_block = json.dumps(extracted_fields, indent=2, ensure_ascii=False)

    user_message = (
        f"## Original {document_type.upper()} text\n\n"
        f"{doc_text}\n\n"
        f"---\n\n"
        f"## Preliminary extraction to verify\n\n"
        f"```json\n{extraction_block}\n```\n\n"
        f"Verify every field above against the document text. "
        f"Return one FieldVerification entry per field."
    )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model=_MODEL,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        tools=[_CRITIC_TOOL],
        tool_choice={"type": "tool", "name": "verify_extraction"},
        messages=[{"role": "user", "content": user_message}],
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    raw = tool_block.input

    verifications = [
        FieldVerification(
            field_name=fv["field_name"],
            extracted_value=extracted_fields.get(fv["field_name"]),
            corrected_value=fv.get("corrected_value"),
            is_confirmed=fv["is_confirmed"],
            confidence=float(fv["confidence"]),
            source_quote=fv.get("source_quote") or "",
            correction_note=fv.get("correction_note") or "",
        )
        for fv in raw["field_verifications"]
    ]

    # Guard: if the LLM skipped a field, add a low-confidence placeholder
    verified_names = {fv.field_name for fv in verifications}
    for field_name, value in extracted_fields.items():
        if field_name not in verified_names:
            verifications.append(
                FieldVerification(
                    field_name=field_name,
                    extracted_value=value,
                    is_confirmed=False,
                    confidence=0.20,
                    source_quote="",
                    correction_note="Field was not verified by the Critic Agent.",
                )
            )

    return CriticReport(
        field_verifications=verifications,
        silent_errors_found=raw.get("silent_errors_found", []),
        overall_confidence=0.0,   # recomputed by model_validator
        critic_notes=raw.get("critic_notes", ""),
    )
