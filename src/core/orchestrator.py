"""
Orchestrator — the single entry point for the entire audit pipeline.

Coordinates all three layers in sequence:
  Layer 1 (Router)    → classify jurisdiction, language, currency
  Layer 2 (Extractor) → region-specific field extraction
  Layer 3 (Critic)    → field-by-field verification + confidence scores
  Auditor             → deterministic anomaly detection on verified data

The only public function is run_audit_pipeline().  Everything else is
internal assembly logic.

Error surface
-------------
All exceptions are caught and converted into a FullAuditResult with
status="error".  The UI never receives a raw Python exception.
"""

from __future__ import annotations

import traceback
from typing import Any, Literal

from pydantic import BaseModel, model_validator

from src.auditor import audit_invoice
from src.core.critic import CriticReport, FieldVerification, verify_extraction
from src.core.router import DocumentContext, LowConfidenceError, classify_document
from src.models import (
    AuditReport,
    ContractRules,
    ExtractionMetadata,
    FieldResult,
    InvoiceData,
)
from src.regions.base_extractor import BaseExtractor, RawExtraction
from src.regions.italy.extractor import ItalyExtractor
from src.regions.switzerland.extractor import SwitzerlandExtractor


# ---------------------------------------------------------------------------
# Extractor registry — add new regions here
# ---------------------------------------------------------------------------

_EXTRACTOR_REGISTRY: dict[str, type[BaseExtractor]] = {
    "IT": ItalyExtractor,
    "CH": SwitzerlandExtractor,
}


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class UnsupportedJurisdictionError(RuntimeError):
    def __init__(self, jurisdiction: str) -> None:
        super().__init__(
            f"No extractor registered for jurisdiction '{jurisdiction}'. "
            f"Supported jurisdictions: {list(_EXTRACTOR_REGISTRY.keys())}."
        )


# ---------------------------------------------------------------------------
# Pipeline output
# ---------------------------------------------------------------------------

class FullAuditResult(BaseModel):
    """
    Single object returned to the UI by run_audit_pipeline().

    status
      "success"           — pipeline completed; all fields verified ≥ threshold
      "needs_human_review"— pipeline completed; one or more fields below threshold
      "error"             — pipeline aborted; inspect error_message

    Fields are None when status="error".
    """

    status: Literal["success", "needs_human_review", "error"]
    error_message: str = ""

    # Document-level classification (Layer 1)
    contract_context: DocumentContext | None = None
    invoice_context: DocumentContext | None = None

    # Fully assembled, Critic-enriched models (Layers 2 + 3)
    contract: ContractRules | None = None
    invoice: InvoiceData | None = None

    # Raw Critic reports — exposed for the UI's "source evidence" panel
    contract_critic: CriticReport | None = None
    invoice_critic: CriticReport | None = None

    # Non-fatal warnings from regional extractors
    contract_warnings: list[str] = []
    invoice_warnings: list[str] = []

    # Final audit result
    audit_report: AuditReport | None = None

    @model_validator(mode="after")
    def _derive_status(self) -> "FullAuditResult":
        """Auto-upgrade status to needs_human_review if any field is flagged."""
        if self.status == "success" and (self.contract or self.invoice):
            flagged = (
                (self.contract.fields_needing_review if self.contract else [])
                + (self.invoice.fields_needing_review if self.invoice else [])
            )
            if flagged:
                self.status = "needs_human_review"
        return self


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_extractor(jurisdiction: str) -> BaseExtractor:
    cls = _EXTRACTOR_REGISTRY.get(jurisdiction)
    if cls is None:
        raise UnsupportedJurisdictionError(jurisdiction)
    return cls()


def _fv_index(critic: CriticReport) -> dict[str, FieldVerification]:
    """Build a fast lookup dict from the Critic's field verifications."""
    return {fv.field_name: fv for fv in critic.field_verifications}


def _make_field(
    name: str,
    raw_value: Any,
    fv_index: dict[str, FieldVerification],
    *,
    coerce: type | None = None,
) -> FieldResult:
    """
    Assemble a single FieldResult by merging the raw extractor value with
    the Critic's verdict.

    Priority order:
      1. Critic's corrected_value (only when Critic is certain of a fix)
      2. Raw extractor value
      3. Fallback default for the given type

    Confidence comes exclusively from the Critic.  If the Critic skipped
    the field the guard in critic.py already inserted a 0.20-confidence
    placeholder, so this path should be rare.
    """
    fv = fv_index.get(name)

    if fv is not None:
        value = fv.final_value          # returns corrected_value or extracted_value
        confidence = fv.confidence
        source_quote = fv.source_quote
    else:
        value = raw_value
        confidence = 0.20               # unverified field
        source_quote = ""

    # Type coercion — LLM output is JSON so sometimes int arrives as float etc.
    if coerce is not None and value is not None:
        try:
            value = coerce(value)
        except (ValueError, TypeError):
            confidence = min(confidence, 0.25)   # coercion failure lowers confidence

    return FieldResult(value=value, confidence=confidence, source_quote=source_quote)


def _build_metadata(context: DocumentContext, doc_type: str) -> ExtractionMetadata:
    return ExtractionMetadata(
        jurisdiction=context.jurisdiction,
        language=context.language,
        currency=context.currency,
        document_type=doc_type,
        router_confidence=context.router_confidence,
    )


def _assemble_contract(
    raw: RawExtraction,
    critic: CriticReport,
    context: DocumentContext,
) -> ContractRules:
    """Merge Layer-2 raw values with Layer-3 confidence scores into ContractRules."""
    f = raw.fields
    idx = _fv_index(critic)

    return ContractRules(
        country=_make_field("country", f.get("country", "UNKNOWN"), idx, coerce=str),
        currency=_make_field("currency", f.get("currency", "UNKNOWN"), idx, coerce=str),
        base_monthly_fee=_make_field("base_monthly_fee", f.get("base_monthly_fee", 0.0), idx, coerce=float),
        extra_hourly_rate=_make_field("extra_hourly_rate", f.get("extra_hourly_rate", 0.0), idx, coerce=float),
        urgent_callout_fee=_make_field("urgent_callout_fee", f.get("urgent_callout_fee", 0.0), idx, coerce=float),
        sla_penalty_per_day=_make_field("sla_penalty_per_day", f.get("sla_penalty_per_day", 0.0), idx, coerce=float),
        indexation_allowed_month=_make_field("indexation_allowed_month", f.get("indexation_allowed_month", ""), idx, coerce=str),
        vat_rate=_make_field("vat_rate", f.get("vat_rate", 0.0), idx, coerce=float),
        metadata=_build_metadata(context, "CONTRACT"),
    )


def _assemble_invoice(
    raw: RawExtraction,
    critic: CriticReport,
    context: DocumentContext,
) -> InvoiceData:
    """Merge Layer-2 raw values with Layer-3 confidence scores into InvoiceData."""
    f = raw.fields
    idx = _fv_index(critic)

    return InvoiceData(
        billed_base_fee=_make_field("billed_base_fee", f.get("billed_base_fee", 0.0), idx, coerce=float),
        billed_extra_hours=_make_field("billed_extra_hours", f.get("billed_extra_hours", 0.0), idx, coerce=float),
        billed_extra_hourly_rate=_make_field("billed_extra_hourly_rate", f.get("billed_extra_hourly_rate", 0.0), idx, coerce=float),
        billed_urgent_callout_fee=_make_field("billed_urgent_callout_fee", f.get("billed_urgent_callout_fee", 0.0), idx, coerce=float),
        unauthorized_services_total=_make_field("unauthorized_services_total", f.get("unauthorized_services_total", 0.0), idx, coerce=float),
        indexation_applied=_make_field("indexation_applied", f.get("indexation_applied", False), idx, coerce=bool),
        invoice_month=_make_field("invoice_month", f.get("invoice_month", ""), idx, coerce=str),
        sla_penalty_deducted=_make_field("sla_penalty_deducted", f.get("sla_penalty_deducted", 0.0), idx, coerce=float),
        detected_delay_days=_make_field("detected_delay_days", f.get("detected_delay_days", 0), idx, coerce=int),
        billed_vat_rate=_make_field("billed_vat_rate", f.get("billed_vat_rate", 0.0), idx, coerce=float),
        metadata=_build_metadata(context, "INVOICE"),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_audit_pipeline(
    contract_text: str,
    invoice_text: str,
) -> FullAuditResult:
    """
    Execute the full 3-layer audit pipeline and return a FullAuditResult.

    This function never raises — all exceptions are converted into a
    FullAuditResult with status="error" so the UI remains stable.

    Parameters
    ----------
    contract_text : Full extracted text of the contract PDF.
    invoice_text  : Full extracted text of the invoice PDF.
    """

    # ------------------------------------------------------------------
    # Layer 1 — Route both documents
    # ------------------------------------------------------------------
    try:
        contract_context = classify_document(contract_text)
    except LowConfidenceError as exc:
        return FullAuditResult(
            status="error",
            error_message=(
                f"Could not classify the contract with sufficient confidence.\n{exc}"
            ),
            contract_context=exc.context,
        )

    try:
        invoice_context = classify_document(invoice_text)
    except LowConfidenceError as exc:
        return FullAuditResult(
            status="error",
            error_message=(
                f"Could not classify the invoice with sufficient confidence.\n{exc}"
            ),
            contract_context=contract_context,
            invoice_context=exc.context,
        )

    # ------------------------------------------------------------------
    # Layer 2 — Regional extraction
    # ------------------------------------------------------------------
    try:
        extractor = _get_extractor(contract_context.jurisdiction)
    except UnsupportedJurisdictionError as exc:
        return FullAuditResult(
            status="error",
            error_message=str(exc),
            contract_context=contract_context,
            invoice_context=invoice_context,
        )

    try:
        raw_contract: RawExtraction = extractor.extract_contract(
            contract_text, contract_context
        )
        raw_invoice: RawExtraction = extractor.extract_invoice(
            invoice_text, invoice_context
        )
    except Exception as exc:
        return FullAuditResult(
            status="error",
            error_message=(
                f"Extraction failed (extractor: {extractor.EXTRACTOR_ID}).\n"
                f"{traceback.format_exc()}"
            ),
            contract_context=contract_context,
            invoice_context=invoice_context,
        )

    # ------------------------------------------------------------------
    # Layer 3 — Critic verification
    # ------------------------------------------------------------------
    try:
        contract_critic: CriticReport = verify_extraction(
            document_text=contract_text,
            extracted_fields=raw_contract.fields,
            document_type="contract",
        )
        invoice_critic: CriticReport = verify_extraction(
            document_text=invoice_text,
            extracted_fields=raw_invoice.fields,
            document_type="invoice",
        )
    except Exception:
        return FullAuditResult(
            status="error",
            error_message=(
                "Critic verification failed.\n"
                f"{traceback.format_exc()}"
            ),
            contract_context=contract_context,
            invoice_context=invoice_context,
            contract_warnings=raw_contract.extraction_warnings,
            invoice_warnings=raw_invoice.extraction_warnings,
        )

    # ------------------------------------------------------------------
    # Assembly — merge raw values + Critic verdicts into final models
    # ------------------------------------------------------------------
    try:
        contract = _assemble_contract(raw_contract, contract_critic, contract_context)
        invoice = _assemble_invoice(raw_invoice, invoice_critic, invoice_context)
    except Exception:
        return FullAuditResult(
            status="error",
            error_message=(
                "Failed to assemble verified models.\n"
                f"{traceback.format_exc()}"
            ),
            contract_context=contract_context,
            invoice_context=invoice_context,
            contract_critic=contract_critic,
            invoice_critic=invoice_critic,
            contract_warnings=raw_contract.extraction_warnings,
            invoice_warnings=raw_invoice.extraction_warnings,
        )

    # ------------------------------------------------------------------
    # Audit — deterministic anomaly detection on verified data
    # ------------------------------------------------------------------
    try:
        audit_report: AuditReport = audit_invoice(contract, invoice)
    except Exception:
        return FullAuditResult(
            status="error",
            error_message=(
                "Audit step failed.\n"
                f"{traceback.format_exc()}"
            ),
            contract_context=contract_context,
            invoice_context=invoice_context,
            contract=contract,
            invoice=invoice,
            contract_critic=contract_critic,
            invoice_critic=invoice_critic,
            contract_warnings=raw_contract.extraction_warnings,
            invoice_warnings=raw_invoice.extraction_warnings,
        )

    # ------------------------------------------------------------------
    # Return — status is auto-derived by FullAuditResult's model_validator
    # ------------------------------------------------------------------
    return FullAuditResult(
        status="success",               # validator upgrades to needs_human_review if needed
        contract_context=contract_context,
        invoice_context=invoice_context,
        contract=contract,
        invoice=invoice,
        contract_critic=contract_critic,
        invoice_critic=invoice_critic,
        audit_report=audit_report,
        contract_warnings=raw_contract.extraction_warnings,
        invoice_warnings=raw_invoice.extraction_warnings,
    )
