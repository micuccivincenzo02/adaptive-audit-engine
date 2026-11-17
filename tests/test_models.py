"""
tests/test_models.py — Pydantic model tests for src/models.py

Coverage:
  TestMoney         — quantization ROUND_HALF_UP, arithmetic, currency safety
  TestFieldResult   — confidence auto-flag, needs_review, evidence optional
  TestDocumentFact  — construction per doc_type, required fields, optional fields
  TestFinding       — Anomaly/OK constraints, fields
  TestAuditResult   — add_document, add_finding, filter properties, badge, impact
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.models import (
    AuditResult,
    ContractClause,
    DocumentFact,
    DocumentMeta,
    Evidence,
    FieldResult,
    Finding,
    LineItem,
    Money,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _meta(jurisdiction: str = "IT", source_file: str = "test.pdf") -> DocumentMeta:
    return DocumentMeta(jurisdiction=jurisdiction, source_file=source_file)


def _money(amount: str, currency: str = "EUR", country: str = "IT") -> Money:
    return Money(amount=Decimal(amount), currency=currency, country=country)


def _fr(value, confidence: float = 1.0) -> FieldResult:
    return FieldResult(value=value, confidence=confidence)


def _evidence(**kwargs) -> Evidence:
    defaults: dict = dict(source_document="invoice.pdf", extracted_text="Importo: EUR 100,00")
    defaults.update(kwargs)
    return Evidence(**defaults)


def _invoice_fact(**kwargs) -> DocumentFact:
    defaults: dict = dict(
        meta=_meta(),
        doc_type="INVOICE",
        vendor=_fr("Vendor Ltd"),
        client=_fr("Client Corp"),
        currency=_fr("EUR"),
        subtotal=_money("1000.00"),
        vat_rate=_fr(Decimal("22.00")),
        vat_amount=_money("220.00"),
        total=_money("1220.00"),
    )
    defaults.update(kwargs)
    return DocumentFact(**defaults)


def _contract_fact(**kwargs) -> DocumentFact:
    defaults: dict = dict(
        meta=_meta(),
        doc_type="CONTRACT",
        vendor=_fr("Vendor Ltd"),
        client=_fr("Client Corp"),
        currency=_fr("EUR"),
    )
    defaults.update(kwargs)
    return DocumentFact(**defaults)


# ── Money ─────────────────────────────────────────────────────────────────────

class TestMoney:

    def test_quantization_round_half_up(self):
        m = Money(amount=Decimal("50.005"), currency="EUR")
        assert m.amount == Decimal("50.01")

    def test_quantization_truncate_below_half(self):
        m = Money(amount=Decimal("50.004"), currency="EUR")
        assert m.amount == Decimal("50.00")

    def test_amount_is_decimal_not_float(self):
        m = Money(amount=Decimal("100.00"), currency="EUR")
        assert isinstance(m.amount, Decimal)
        assert type(m.amount) is not float

    def test_add_same_currency(self):
        result = Money(amount=Decimal("100.00"), currency="EUR") + Money(amount=Decimal("50.00"), currency="EUR")
        assert result.amount == Decimal("150.00")
        assert result.currency == "EUR"

    def test_sub_same_currency(self):
        result = Money(amount=Decimal("100.00"), currency="EUR") - Money(amount=Decimal("30.00"), currency="EUR")
        assert result.amount == Decimal("70.00")

    def test_add_different_currency_raises(self):
        a = Money(amount=Decimal("100.00"), currency="EUR")
        b = Money(amount=Decimal("100.00"), currency="CHF")
        with pytest.raises(ValueError, match="Cannot add"):
            _ = a + b

    def test_sub_different_currency_raises(self):
        a = Money(amount=Decimal("100.00"), currency="EUR")
        b = Money(amount=Decimal("50.00"), currency="CHF")
        with pytest.raises(ValueError, match="Cannot subtract"):
            _ = a - b

    def test_country_defaults_unknown(self):
        m = Money(amount=Decimal("10.00"), currency="EUR")
        assert m.country == "UNKNOWN"

    def test_add_preserves_country_when_same(self):
        a = Money(amount=Decimal("10.00"), currency="EUR", country="IT")
        b = Money(amount=Decimal("5.00"), currency="EUR", country="IT")
        assert (a + b).country == "IT"

    def test_add_sets_unknown_when_countries_differ(self):
        a = Money(amount=Decimal("10.00"), currency="EUR", country="IT")
        b = Money(amount=Decimal("5.00"), currency="EUR", country="UNKNOWN")
        assert (a + b).country == "UNKNOWN"

    def test_chf_currency_accepted(self):
        m = Money(amount=Decimal("100.00"), currency="CHF", country="CH")
        assert m.currency == "CHF"


# ── FieldResult ───────────────────────────────────────────────────────────────

class TestFieldResult:

    def test_auto_flag_below_threshold(self):
        fr = FieldResult(value=Decimal("100.00"), confidence=0.89)
        assert fr.needs_review is True

    def test_auto_flag_at_threshold_not_flagged(self):
        fr = FieldResult(value=Decimal("100.00"), confidence=0.90)
        assert fr.needs_review is False

    def test_auto_flag_above_threshold_not_flagged(self):
        fr = FieldResult(value="EUR", confidence=1.0)
        assert fr.needs_review is False

    def test_confidence_default_one(self):
        fr = FieldResult(value="test")
        assert fr.confidence == 1.0

    def test_needs_review_default_false(self):
        fr = FieldResult(value="test")
        assert fr.needs_review is False

    def test_evidence_optional_none_by_default(self):
        fr = FieldResult(value="test")
        assert fr.evidence is None

    def test_evidence_can_be_set(self):
        ev = Evidence(source_document="doc.pdf", extracted_text="text")
        fr = FieldResult(value="test", evidence=ev)
        assert fr.evidence is not None
        assert fr.evidence.source_document == "doc.pdf"

    def test_str_value(self):
        fr = FieldResult(value="EUR", confidence=0.95)
        assert fr.value == "EUR"

    def test_decimal_value(self):
        fr = FieldResult(value=Decimal("22.00"))
        assert fr.value == Decimal("22.00")


# ── DocumentFact ──────────────────────────────────────────────────────────────

class TestDocumentFact:

    def test_invoice_fact_required_fields(self):
        doc = _invoice_fact()
        assert doc.doc_type == "INVOICE"
        assert doc.subtotal is not None
        assert doc.vat_rate is not None
        assert doc.vat_amount is not None
        assert doc.total is not None

    def test_contract_fact_invoice_fields_are_none(self):
        doc = _contract_fact()
        assert doc.doc_type == "CONTRACT"
        assert doc.subtotal is None
        assert doc.vat_rate is None
        assert doc.vat_amount is None
        assert doc.total is None

    def test_ddt_fact_creation(self):
        doc = DocumentFact(
            meta=_meta(),
            doc_type="DDT",
            vendor=_fr("Vendor"),
            client=_fr("Client"),
            currency=_fr("EUR"),
        )
        assert doc.doc_type == "DDT"

    def test_unknown_doc_type_accepted(self):
        doc = DocumentFact(
            meta=_meta(),
            doc_type="UNKNOWN",
            vendor=_fr("V"),
            client=_fr("C"),
            currency=_fr("EUR"),
        )
        assert doc.doc_type == "UNKNOWN"

    def test_missing_doc_type_raises(self):
        with pytest.raises(ValidationError):
            DocumentFact(
                meta=_meta(),
                vendor=_fr("V"),
                client=_fr("C"),
                currency=_fr("EUR"),
            )

    def test_missing_currency_raises(self):
        with pytest.raises(ValidationError):
            DocumentFact(
                meta=_meta(),
                doc_type="INVOICE",
                vendor=_fr("V"),
                client=_fr("C"),
            )

    def test_contract_optional_fields_for_invoice(self):
        doc = _invoice_fact()
        assert doc.start_date is None
        assert doc.end_date is None
        assert doc.clauses == []

    def test_contract_with_dates(self):
        doc = _contract_fact(
            start_date=_fr(date(2024, 1, 1)),
            end_date=_fr(date(2025, 12, 31)),
        )
        assert doc.start_date.value == date(2024, 1, 1)
        assert doc.end_date.value == date(2025, 12, 31)

    def test_contract_with_clauses(self):
        doc = _contract_fact(
            clauses=[ContractClause(title="Pagamento", text="Entro 30 giorni")],
        )
        assert len(doc.clauses) == 1
        assert doc.clauses[0].title == "Pagamento"

    def test_line_items_default_empty(self):
        doc = _invoice_fact()
        assert doc.line_items == []

    def test_line_item_amount_is_decimal(self):
        item = LineItem(
            description="Service A",
            quantity=Decimal("2"),
            unit_price=_money("50.00"),
            total=_money("100.00"),
        )
        assert isinstance(item.unit_price.amount, Decimal)
        assert isinstance(item.total.amount, Decimal)

    def test_meta_jurisdiction_propagated(self):
        doc = _invoice_fact(meta=_meta(jurisdiction="CH"))
        assert doc.meta.jurisdiction == "CH"


# ── Finding ───────────────────────────────────────────────────────────────────

class TestFinding:

    def test_anomaly_with_motivation_valid(self):
        f = Finding(
            finding_type="PRICING",
            status="Anomaly",
            motivation="Unit price EUR 60 billed vs EUR 50 in contract.",
        )
        assert f.status == "Anomaly"

    def test_anomaly_empty_motivation_raises(self):
        with pytest.raises(ValidationError):
            Finding(finding_type="PRICING", status="Anomaly", motivation="")

    def test_anomaly_whitespace_motivation_raises(self):
        with pytest.raises(ValidationError):
            Finding(finding_type="PRICING", status="Anomaly", motivation="   ")

    def test_ok_with_evidence_valid(self):
        f = Finding(
            finding_type="FISCAL",
            status="OK",
            motivation="VAT 22% matches IT standard.",
            evidence=[_evidence()],
        )
        assert f.status == "OK"

    def test_ok_without_evidence_raises(self):
        with pytest.raises(ValidationError):
            Finding(
                finding_type="FISCAL",
                status="OK",
                motivation="VAT is correct.",
                evidence=[],
            )

    def test_warning_without_evidence_valid(self):
        f = Finding(
            finding_type="FISCAL",
            status="Warning",
            motivation="Cannot verify VAT: jurisdiction unknown.",
        )
        assert f.status == "Warning"

    def test_finding_id_auto_generated(self):
        f = Finding(finding_type="OTHER", status="Warning", motivation="Test.")
        assert f.finding_id != ""

    def test_impact_value_optional(self):
        f = Finding(finding_type="CONTRACTUAL", status="Warning", motivation="No clause.")
        assert f.impact_value is None

    def test_impact_value_monetary(self):
        f = Finding(
            finding_type="PRICING",
            status="Anomaly",
            motivation="Overcharge of EUR 50.",
            impact_value=Money(amount=Decimal("50.00"), currency="EUR"),
        )
        assert f.impact_value.amount == Decimal("50.00")


# ── AuditResult ───────────────────────────────────────────────────────────────

class TestAuditResult:

    def test_add_document_appends(self):
        session = AuditResult(name="Test")
        session.add_document(_invoice_fact())
        assert len(session.documents) == 1

    def test_contracts_property_filters_by_type(self):
        session = AuditResult(name="Test")
        session.add_document(_contract_fact())
        session.add_document(_invoice_fact())
        assert len(session.contracts) == 1
        assert session.contracts[0].doc_type == "CONTRACT"

    def test_invoices_property_filters_by_type(self):
        session = AuditResult(name="Test")
        session.add_document(_contract_fact())
        session.add_document(_invoice_fact())
        assert len(session.invoices) == 1
        assert session.invoices[0].doc_type == "INVOICE"

    def test_anomalies_property(self):
        session = AuditResult(name="Test")
        session.add_finding(Finding(finding_type="PRICING", status="Anomaly", motivation="Overcharge."))
        session.add_finding(Finding(finding_type="FISCAL", status="Warning", motivation="Unverified."))
        assert len(session.anomalies) == 1

    def test_reviews_needed_property(self):
        session = AuditResult(name="Test")
        session.add_finding(Finding(finding_type="FISCAL", status="Warning", motivation="Unverified."))
        assert len(session.reviews_needed) == 1

    def test_confirmed_ok_property(self):
        session = AuditResult(name="Test")
        session.add_finding(Finding(
            finding_type="FISCAL", status="OK",
            motivation="VAT 22% correct.",
            evidence=[_evidence()],
        ))
        assert len(session.confirmed_ok) == 1

    def test_overall_badge_anomaly_wins(self):
        session = AuditResult(name="Test")
        session.add_finding(Finding(finding_type="PRICING", status="Anomaly", motivation="Overcharge."))
        session.add_finding(Finding(
            finding_type="FISCAL", status="OK",
            motivation="VAT OK.",
            evidence=[_evidence()],
        ))
        assert session.overall_badge == "Anomaly"

    def test_overall_badge_warning_without_anomaly(self):
        session = AuditResult(name="Test")
        session.add_finding(Finding(finding_type="FISCAL", status="Warning", motivation="Unverified."))
        assert session.overall_badge == "Warning"

    def test_overall_badge_ok_when_all_ok(self):
        session = AuditResult(name="Test")
        session.add_finding(Finding(
            finding_type="FISCAL", status="OK",
            motivation="VAT 22% correct.",
            evidence=[_evidence()],
        ))
        assert session.overall_badge == "OK"

    def test_total_impact_sums_decimal(self):
        session = AuditResult(name="Test")
        session.add_finding(Finding(
            finding_type="PRICING", status="Anomaly",
            motivation="Overcharge EUR 100.",
            impact_value=Money(amount=Decimal("100.00"), currency="EUR"),
        ))
        session.add_finding(Finding(
            finding_type="PRICING", status="Anomaly",
            motivation="Overcharge EUR 50.",
            impact_value=Money(amount=Decimal("50.00"), currency="EUR"),
        ))
        assert session.total_impact == Decimal("150.00")
        assert isinstance(session.total_impact, Decimal)

    def test_total_impact_none_when_no_impact(self):
        session = AuditResult(name="Test")
        session.add_finding(Finding(finding_type="FISCAL", status="Warning", motivation="Unverified."))
        assert session.total_impact is None

    def test_session_id_auto_generated(self):
        session = AuditResult(name="Test")
        assert session.session_id != ""

    def test_status_default_pending(self):
        session = AuditResult(name="Test")
        assert session.status == "PENDING"
