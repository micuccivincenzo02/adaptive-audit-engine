"""
tests/test_evidence.py — Evidence model and Finding constraint tests

Coverage:
  TestEvidenceModel        — Evidence fields, defaults, optional fields
  TestFindingConstraints   — Anomaly/OK/Warning badge rules (from CLAUDE.md)
  TestAuditEngineEvidence  — AuditEngine produces Findings with populated evidence
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from audit_engine import AuditEngine
from schemas.base import Evidence, FieldResult, Money
from schemas.document import Contract, DocumentMeta, Invoice, LineItem
from src.models import Finding


# ── Helpers ───────────────────────────────────────────────────────────────────

def _meta(jurisdiction: str = "IT", source_file: str = "test.pdf") -> DocumentMeta:
    return DocumentMeta(
        jurisdiction=jurisdiction,
        source_file=source_file,
        issue_date=date(2025, 1, 31),
    )


def _m(amount: str, currency: str = "EUR", country: str = "IT") -> Money:
    return Money(amount=Decimal(amount), currency=currency, country=country)


def _fr(value, confidence: float = 1.0) -> FieldResult:
    return FieldResult(value=value, confidence=confidence)


def _line(description: str, qty: str, unit: str, total: str) -> LineItem:
    return LineItem(
        description=description,
        quantity=Decimal(qty),
        unit_price=_m(unit),
        total=_m(total),
    )


def _invoice(line_items=None, vat_rate: str = "22.00", subtotal: str = "1000.00") -> Invoice:
    items = line_items or []
    vat_amt = Money(
        amount=Decimal(subtotal) * Decimal(vat_rate) / Decimal("100"),
        currency="EUR",
        country="IT",
    )
    return Invoice(
        meta=_meta(),
        doc_type="INVOICE",
        vendor=_fr("Vendor Ltd"),
        client=_fr("Client Corp"),
        currency=_fr("EUR"),
        line_items=items,
        subtotal=_m(subtotal),
        vat_rate=_fr(Decimal(vat_rate)),
        vat_amount=vat_amt,
        total=_m(str(Decimal(subtotal) + vat_amt.amount)),
    )


def _contract(line_items=None) -> Contract:
    return Contract(
        meta=_meta(source_file="contract.pdf"),
        doc_type="CONTRACT",
        vendor=_fr("Vendor Ltd"),
        client=_fr("Client Corp"),
        currency=_fr("EUR"),
        line_items=line_items or [],
    )


engine = AuditEngine()


# ── Evidence model ────────────────────────────────────────────────────────────

class TestEvidenceModel:

    def test_required_source_document(self):
        ev = Evidence(source_document="invoice.pdf")
        assert ev.source_document == "invoice.pdf"

    def test_extracted_text_default_empty(self):
        ev = Evidence(source_document="doc.pdf")
        assert ev.extracted_text == ""

    def test_page_number_optional_none_by_default(self):
        ev = Evidence(source_document="doc.pdf")
        assert ev.page_number is None

    def test_matched_clause_optional_none_by_default(self):
        ev = Evidence(source_document="doc.pdf")
        assert ev.matched_clause is None

    def test_invoice_field_optional_none_by_default(self):
        ev = Evidence(source_document="doc.pdf")
        assert ev.invoice_field is None

    def test_confidence_default_one(self):
        ev = Evidence(source_document="doc.pdf")
        assert ev.confidence == 1.0

    def test_all_fields_populated(self):
        ev = Evidence(
            source_document="invoice.pdf",
            page_number=3,
            extracted_text="Totale: EUR 1.220,00",
            matched_clause="Art. 5: Canone mensile EUR 1.000,00",
            invoice_field="total",
            confidence=0.92,
        )
        assert ev.page_number == 3
        assert ev.extracted_text == "Totale: EUR 1.220,00"
        assert ev.matched_clause == "Art. 5: Canone mensile EUR 1.000,00"
        assert ev.invoice_field == "total"
        assert ev.confidence == 0.92


# ── Finding badge constraints (CLAUDE.md badge rules) ────────────────────────

class TestFindingConstraints:

    def test_anomaly_empty_motivation_raises(self):
        with pytest.raises(ValidationError):
            Finding(finding_type="PRICING", status="Anomaly", motivation="")

    def test_anomaly_whitespace_motivation_raises(self):
        with pytest.raises(ValidationError):
            Finding(finding_type="PRICING", status="Anomaly", motivation="   ")

    def test_anomaly_with_motivation_valid(self):
        f = Finding(
            finding_type="PRICING",
            status="Anomaly",
            motivation="Unit price EUR 60 billed vs EUR 50 in contract.",
        )
        assert f.status == "Anomaly"

    def test_ok_without_evidence_raises(self):
        with pytest.raises(ValidationError):
            Finding(
                finding_type="FISCAL",
                status="OK",
                motivation="VAT is correct.",
                evidence=[],
            )

    def test_ok_with_evidence_valid(self):
        ev = Evidence(source_document="invoice.pdf", extracted_text="IVA 22%")
        f = Finding(
            finding_type="FISCAL",
            status="OK",
            motivation="VAT 22% matches the IT standard.",
            evidence=[ev],
        )
        assert f.status == "OK"
        assert len(f.evidence) == 1

    def test_warning_without_evidence_valid(self):
        f = Finding(
            finding_type="FISCAL",
            status="Warning",
            motivation="Jurisdiction unknown; VAT cannot be verified.",
        )
        assert f.status == "Warning"
        assert f.evidence == []

    def test_finding_id_auto_generated_non_empty(self):
        f = Finding(finding_type="OTHER", status="Warning", motivation="Test.")
        assert f.finding_id != ""

    def test_impact_value_optional_none_by_default(self):
        f = Finding(finding_type="CONTRACTUAL", status="Warning", motivation="No clause found.")
        assert f.impact_value is None

    def test_source_document_ids_default_empty(self):
        f = Finding(finding_type="OTHER", status="Warning", motivation="Test.")
        assert f.source_document_ids == []


# ── AuditEngine produces Findings with populated evidence ─────────────────────

class TestAuditEngineEvidence:

    def test_price_mismatch_has_two_evidence_entries(self):
        inv = _invoice([_line("IT Consulting", "1", "60.00", "60.00")])
        con = _contract([_line("IT Consulting", "1", "50.00", "50.00")])
        findings = engine.cross_check(inv, con)
        price = [f for f in findings if f.finding_type == "PRICING"]
        assert len(price) == 1
        assert len(price[0].evidence) == 2

    def test_price_mismatch_evidence_source_documents(self):
        inv = _invoice([_line("IT Consulting", "1", "60.00", "60.00")])
        con = _contract([_line("IT Consulting", "1", "50.00", "50.00")])
        findings = engine.cross_check(inv, con)
        price = next(f for f in findings if f.finding_type == "PRICING")
        sources = {e.source_document for e in price.evidence}
        assert "test.pdf" in sources
        assert "contract.pdf" in sources

    def test_price_mismatch_invoice_evidence_has_field(self):
        inv = _invoice([_line("IT Consulting", "1", "60.00", "60.00")])
        con = _contract([_line("IT Consulting", "1", "50.00", "50.00")])
        findings = engine.cross_check(inv, con)
        price = next(f for f in findings if f.finding_type == "PRICING")
        invoice_ev = next(e for e in price.evidence if e.source_document == "test.pdf")
        assert invoice_ev.invoice_field == "unit_price"

    def test_vat_ok_finding_has_evidence(self):
        inv = _invoice([], vat_rate="22.00")
        con = _contract([])
        findings = engine.cross_check(inv, con)
        vat = [f for f in findings if f.finding_type == "FISCAL"]
        assert len(vat) == 1
        assert vat[0].status == "OK"
        assert len(vat[0].evidence) >= 1

    def test_extra_item_evidence_invoice_field(self):
        inv = _invoice([_line("Unknown service", "1", "200.00", "200.00")])
        con = _contract([])
        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]
        assert len(contractual) == 1
        assert contractual[0].evidence[0].invoice_field == "line_items"

    def test_anomaly_motivation_references_actual_amounts(self):
        inv = _invoice([_line("IT Consulting", "1", "60.00", "60.00")])
        con = _contract([_line("IT Consulting", "1", "50.00", "50.00")])
        findings = engine.cross_check(inv, con)
        price = next(f for f in findings if f.finding_type == "PRICING")
        assert "60" in price.motivation
        assert "50" in price.motivation
