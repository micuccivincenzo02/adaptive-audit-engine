"""
Base Extractor — abstract interface for all Layer-2 regional extractors.

Every regional extractor must:
  1. Subclass BaseExtractor.
  2. Implement extract_contract() and extract_invoice().
  3. Return a RawExtraction whose .fields dict maps exactly to the field
     names defined in ContractRules / InvoiceData (without FieldResult
     wrappers — those are applied later by the Orchestrator after the
     Critic has enriched each value with a confidence score).

RawExtraction is intentionally a plain dataclass, not a Pydantic model,
because it is a transient intermediate object that never leaves the
pipeline boundary.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import anthropic
from dotenv import load_dotenv

from src.core.router import DocumentContext

load_dotenv()

_MAX_CHARS = 100_000
_MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Shared output container
# ---------------------------------------------------------------------------

@dataclass
class RawExtraction:
    """
    Transient output of a Layer-2 extractor.

    fields             : {field_name: raw_value} — fed directly to the Critic.
    extractor_id       : identifies which extractor produced this (for logging).
    extraction_warnings: non-fatal issues noticed during extraction
                         (e.g. "VAT clause not found, defaulted to 22.0").
    """
    fields: dict[str, Any]
    extractor_id: str
    extraction_warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared tool schemas
# Kept here so both regional extractors use identical field names and
# descriptions, diverging only in region-specific context strings.
# ---------------------------------------------------------------------------

def _build_contract_tool(region_context: str) -> dict:
    """Returns the tool definition for contract extraction, injecting region context."""
    return {
        "name": "extract_contract_rules",
        "description": (
            f"Extract the key financial and operational rules from a facility-management "
            f"contract. {region_context}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "country": {
                    "type": "string",
                    "description": "Jurisdiction code: 'IT' for Italy, 'CH' for Switzerland.",
                },
                "currency": {
                    "type": "string",
                    "description": "Currency used in the contract: 'EUR' (Italy) or 'CHF' (Switzerland).",
                },
                "base_monthly_fee": {
                    "type": "number",
                    "description": "Recurring monthly base fee (numeric only, in contract currency).",
                },
                "extra_hourly_rate": {
                    "type": "number",
                    "description": "Hourly rate for extra or authorised services outside the base scope.",
                },
                "urgent_callout_fee": {
                    "type": "number",
                    "description": "Fixed fee per urgent / emergency intervention.",
                },
                "sla_penalty_per_day": {
                    "type": "number",
                    "description": "Penalty amount per calendar day of missed service.",
                },
                "indexation_allowed_month": {
                    "type": "string",
                    "description": (
                        "The calendar month from which price indexation is contractually "
                        "permitted (full English name, e.g. 'January'). "
                        "Return empty string if the contract has no indexation clause."
                    ),
                },
                "vat_rate": {
                    "type": "number",
                    "description": (
                        "VAT / tax rate as a percentage (e.g. 22.0 for Italy IVA, "
                        "8.1 for Switzerland MWST/TVA/IVA). "
                        "Extract from the tax clause; do not assume a default."
                    ),
                },
            },
            "required": [
                "country", "currency", "base_monthly_fee", "extra_hourly_rate",
                "urgent_callout_fee", "sla_penalty_per_day",
                "indexation_allowed_month", "vat_rate",
            ],
        },
    }


def _build_invoice_tool(region_context: str) -> dict:
    """Returns the tool definition for invoice extraction, injecting region context."""
    return {
        "name": "extract_invoice_data",
        "description": (
            f"Extract structured billing data from a facility-management invoice. "
            f"{region_context}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "billed_base_fee": {
                    "type": "number",
                    "description": "Standard monthly base fee as billed on this invoice.",
                },
                "billed_extra_hours": {
                    "type": "number",
                    "description": "Number of extra hours billed. 0 if none.",
                },
                "billed_extra_hourly_rate": {
                    "type": "number",
                    "description": "Hourly rate charged for extra hours. 0 if no extra hours.",
                },
                "billed_urgent_callout_fee": {
                    "type": "number",
                    "description": "Amount billed as urgent / emergency call-out fee. 0 if absent.",
                },
                "unauthorized_services_total": {
                    "type": "number",
                    "description": (
                        "Total billed for services not covered by the base contract "
                        "(excluding urgent call-out fees). 0 if none."
                    ),
                },
                "indexation_applied": {
                    "type": "boolean",
                    "description": "True if a price indexation or inflation adjustment appears on this invoice.",
                },
                "invoice_month": {
                    "type": "string",
                    "description": (
                        "Calendar month this invoice covers, as a full English name "
                        "(e.g. 'March'). Empty string if not determinable."
                    ),
                },
                "sla_penalty_deducted": {
                    "type": "number",
                    "description": "Amount deducted as SLA penalty for missed service days. 0 if none.",
                },
                "detected_delay_days": {
                    "type": "integer",
                    "description": (
                        "Number of service-delay days mentioned on the invoice "
                        "(late arrivals, missed SLA days, non-service days). 0 if none."
                    ),
                },
                "billed_vat_rate": {
                    "type": "number",
                    "description": (
                        "VAT rate applied on this invoice as a percentage. "
                        "Look for IVA, MWST, TVA, MwSt labels. 0 if no tax shown."
                    ),
                },
            },
            "required": [
                "billed_base_fee", "billed_extra_hours", "billed_extra_hourly_rate",
                "billed_urgent_callout_fee", "unauthorized_services_total",
                "indexation_applied", "invoice_month", "sla_penalty_deducted",
                "detected_delay_days", "billed_vat_rate",
            ],
        },
    }


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseExtractor(ABC):
    """
    Interface every regional extractor must implement.

    Subclasses configure:
      - EXTRACTOR_ID   : unique string identifier (used in logs and CriticReport)
      - _system_prompt : region-aware system instruction passed to the LLM
      - _contract_tool / _invoice_tool : built via _build_contract_tool()
        and _build_invoice_tool() with region-specific context strings
    """

    EXTRACTOR_ID: str = "base"

    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def extract_contract(
        self, text: str, context: DocumentContext
    ) -> RawExtraction:
        """Extract contract rules from *text* and return a RawExtraction."""

    @abstractmethod
    def extract_invoice(
        self, text: str, context: DocumentContext
    ) -> RawExtraction:
        """Extract invoice data from *text* and return a RawExtraction."""

    # ------------------------------------------------------------------
    # Shared helper used by both subclasses
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        text: str,
        tool: dict,
        system_prompt: str,
        user_prefix: str,
    ) -> dict[str, Any]:
        """
        Single LLM call that enforces tool use and returns the raw input dict.
        Caps *text* at _MAX_CHARS before sending.
        """
        truncated = text[:_MAX_CHARS]
        response = self._client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            system=system_prompt,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": f"{user_prefix}\n\n{truncated}"}],
        )
        tool_block = next(b for b in response.content if b.type == "tool_use")
        return tool_block.input
