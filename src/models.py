"""
src/models.py — Centralised Pydantic models for the Adaptive Audit Engine IT/CH.

All domain types live here. External modules import exclusively from this file.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, Field, model_validator

from src.core.money import money as _money

# ── Base exception ────────────────────────────────────────────────────────────

class AuditError(Exception):
    """Base exception for all audit engine errors."""


# ── Constants ─────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.90
T = TypeVar("T")

DocumentType = Literal["CONTRACT", "INVOICE", "DDT", "UNKNOWN"]
BadgeStatus = Literal["OK", "Warning", "Anomaly"]
FindingType = Literal["FISCAL", "CONTRACTUAL", "SLA", "PRICING", "QUANTITY", "OTHER"]


# ── Primitives ────────────────────────────────────────────────────────────────

class Money(BaseModel):
    """Immutable monetary value. Always quantized to 2 d.p. ROUND_HALF_UP."""

    amount: Decimal
    currency: Literal["EUR", "CHF"]
    country: Literal["IT", "CH", "UNKNOWN"] = "UNKNOWN"

    @model_validator(mode="after")
    def _quantize(self) -> "Money":
        self.amount = _money(self.amount)
        return self

    def __add__(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError(f"Cannot add {self.currency} and {other.currency} — convert first")
        return Money(
            amount=_money(self.amount + other.amount),
            currency=self.currency,
            country=self.country if self.country == other.country else "UNKNOWN",
        )

    def __sub__(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError(f"Cannot subtract {self.currency} and {other.currency} — convert first")
        return Money(
            amount=_money(self.amount - other.amount),
            currency=self.currency,
            country=self.country if self.country == other.country else "UNKNOWN",
        )

    def __repr__(self) -> str:
        return f"{self.currency} {self.amount:.2f} ({self.country})"


class Evidence(BaseModel):
    """Verbatim proof attached to a finding or extracted fact."""

    source_document: str
    page_number: Optional[int] = None
    extracted_text: str = ""
    matched_clause: Optional[str] = None
    invoice_field: Optional[str] = None
    confidence: float = 1.0


class FieldResult(BaseModel, Generic[T]):
    """LLM-extracted value with confidence score and optional evidence."""

    value: T
    confidence: float = 1.0
    needs_review: bool = False
    evidence: Optional[Evidence] = None

    @model_validator(mode="after")
    def _auto_flag(self) -> "FieldResult[T]":
        if self.confidence < CONFIDENCE_THRESHOLD:
            self.needs_review = True
        return self


# ── Document components ───────────────────────────────────────────────────────

class LineItem(BaseModel):
    """Single line in any document: invoice line, contract schedule item, DDT entry."""

    description: str
    quantity: Decimal
    unit_price: Money
    total: Money
    evidence: Optional[Evidence] = None
    sku: Optional[str] = None  # Product code / SKU / part number extracted by LLM


class DocumentMeta(BaseModel):
    """Extraction context carried by every document."""

    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_number: Optional[str] = None
    issue_date: Optional[date] = None
    jurisdiction: Literal["IT", "CH", "UNKNOWN"] = "UNKNOWN"
    language: str = "unknown"
    source_file: Optional[str] = None
    extraction_confidence: float = 1.0


class ContractClause(BaseModel):
    """A named clause from a contract with verbatim text and optional evidence."""

    title: str
    text: str
    evidence: Optional[Evidence] = None


# ── Core document model ───────────────────────────────────────────────────────

class DocumentFact(BaseModel):
    """
    Unified representation of any extracted document.

    doc_type drives audit logic. Optional fields are populated only when
    relevant to the detected document type.
    """

    meta: DocumentMeta
    doc_type: DocumentType
    vendor: FieldResult[str]
    client: FieldResult[str]
    currency: FieldResult[str]
    line_items: list[LineItem] = []

    # Contract-specific
    start_date: Optional[FieldResult[date]] = None
    end_date: Optional[FieldResult[date]] = None
    clauses: list[ContractClause] = []

    # Invoice-specific
    subtotal: Optional[Money] = None
    vat_rate: Optional[FieldResult[Decimal]] = None
    vat_amount: Optional[Money] = None
    total: Optional[Money] = None

    is_truncated: bool = False


# ── Audit models ──────────────────────────────────────────────────────────────

class Finding(BaseModel):
    """
    Standalone audit result. Produced by any agent, consumed by any output layer.

    motivation must be a concrete sentence referencing actual document values.
    """

    finding_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    finding_type: FindingType
    status: BadgeStatus
    motivation: str
    impact_value: Optional[Money] = None
    evidence: list[Evidence] = []
    source_document_ids: list[str] = []
    rule_reference: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _anomaly_needs_motivation(self) -> "Finding":
        if self.status == "Anomaly" and not self.motivation.strip():
            raise ValueError("An Anomaly finding must include a non-empty motivation.")
        return self

    @model_validator(mode="after")
    def _ok_needs_evidence(self) -> "Finding":
        if self.status == "OK" and not self.evidence:
            raise ValueError(
                "An OK finding must include at least one Evidence entry. "
                "No evidence = no OK badge."
            )
        return self


class AuditResult(BaseModel):
    """Top-level container for one audit run."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: Literal["PENDING", "IN_PROGRESS", "COMPLETE", "FAILED"] = "PENDING"
    documents: list[DocumentFact] = []
    findings: list[Finding] = []

    def add_document(self, doc: DocumentFact) -> None:
        self.documents.append(doc)

    def add_finding(self, finding: Finding) -> None:
        self.findings.append(finding)

    @property
    def contracts(self) -> list[DocumentFact]:
        return [d for d in self.documents if d.doc_type == "CONTRACT"]

    @property
    def invoices(self) -> list[DocumentFact]:
        return [d for d in self.documents if d.doc_type == "INVOICE"]

    @property
    def anomalies(self) -> list[Finding]:
        return [f for f in self.findings if f.status == "Anomaly"]

    @property
    def reviews_needed(self) -> list[Finding]:
        return [f for f in self.findings if f.status == "Warning"]

    @property
    def confirmed_ok(self) -> list[Finding]:
        return [f for f in self.findings if f.status == "OK"]

    @property
    def overall_badge(self) -> BadgeStatus:
        statuses = {f.status for f in self.findings}
        if "Anomaly" in statuses:
            return "Anomaly"
        if "Warning" in statuses:
            return "Warning"
        if "OK" in statuses:
            return "OK"
        return "Warning"

    @property
    def total_impact(self) -> Optional[Decimal]:
        impacts = [f.impact_value.amount for f in self.findings if f.impact_value]
        if not impacts:
            return None
        return sum(impacts, Decimal("0.00"))
