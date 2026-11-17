"""
tests/test_audit_engine.py — unit tests for AuditEngine.cross_check_ddt_invoice()

Covers:
  - quantity match (no finding)
  - quantity mismatch → Anomaly with correct motivation and impact
  - unit price match (no finding)
  - price mismatch slight (≤ threshold) → Warning
  - price mismatch severe (> threshold) → Anomaly
  - extra invoice item (absent from DDT and Contract) → Anomaly
  - item in contract only (not in DDT) → not flagged as extra
  - VAT OK — IT 22%
  - VAT OK — CH 8.1%
  - VAT MISMATCH → Anomaly with impact
  - VAT unknown jurisdiction → Warning
  - OK finding has evidence (model constraint)
  - Anomaly motivation is non-empty (model constraint)
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from audit_engine import AuditEngine
from schemas.base import Evidence, FieldResult, Money
from schemas.document import Contract, ContractClause, DDT, DocumentMeta, Invoice, LineItem

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _meta(jurisdiction: str = "IT", source_file: str = "test.pdf") -> DocumentMeta:
    return DocumentMeta(
        jurisdiction=jurisdiction,
        source_file=source_file,
        issue_date=date(2025, 1, 31),
    )


def _money(amount: str, currency: str = "EUR", country: str = "IT") -> Money:
    return Money(amount=Decimal(amount), currency=currency, country=country)


def _fr(value, confidence: float = 1.0) -> FieldResult:
    return FieldResult(value=value, confidence=confidence)


def _line(
    description: str,
    qty: str,
    unit: str,
    total: str,
    currency: str = "EUR",
    country: str = "IT",
) -> LineItem:
    return LineItem(
        description=description,
        quantity=Decimal(qty),
        unit_price=_money(unit, currency, country),
        total=_money(total, currency, country),
    )


def _invoice(
    line_items: list[LineItem],
    vat_rate: str = "22.00",
    jurisdiction: str = "IT",
    subtotal: str = "1000.00",
) -> Invoice:
    sub = _money(subtotal)
    vat_amount = Money(
        amount=Decimal(subtotal) * Decimal(vat_rate) / Decimal("100"),
        currency="EUR",
        country=jurisdiction,
    )
    return Invoice(
        meta=_meta(jurisdiction=jurisdiction),
        doc_type="INVOICE",
        vendor=_fr("Vendor Ltd"),
        client=_fr("Client Corp"),
        currency=_fr("EUR"),
        line_items=line_items,
        subtotal=sub,
        vat_amount=vat_amount,
        total=_money(str(Decimal(subtotal) + vat_amount.amount)),
        vat_rate=_fr(Decimal(vat_rate)),
    )


def _ddt(line_items: list[LineItem]) -> DDT:
    return DDT(
        meta=_meta(source_file="ddt.pdf"),
        doc_type="DDT",
        vendor=_fr("Vendor Ltd"),
        client=_fr("Client Corp"),
        currency=_fr("EUR"),
        line_items=line_items,
    )


def _contract(
    line_items: list[LineItem],
    jurisdiction: str = "IT",
) -> Contract:
    return Contract(
        meta=_meta(jurisdiction=jurisdiction),
        doc_type="CONTRACT",
        vendor=_fr("Vendor Ltd"),
        client=_fr("Client Corp"),
        currency=_fr("EUR"),
        line_items=line_items,
    )


engine = AuditEngine()


# ---------------------------------------------------------------------------
# Quantity checks
# ---------------------------------------------------------------------------

class TestQuantityCheck:
    def test_matching_quantities_produce_no_finding(self):
        inv = _invoice([_line("IT Consulting", "8", "50.00", "400.00")])
        ddt_ = _ddt([_line("IT Consulting", "8", "50.00", "400.00")])
        con = _contract([_line("IT Consulting", "8", "50.00", "400.00")])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        qty_findings = [f for f in findings if f.finding_type == "QUANTITY"]
        assert qty_findings == []

    def test_excess_quantity_is_anomalia(self):
        inv = _invoice([_line("IT Consulting", "10", "50.00", "500.00")])
        ddt_ = _ddt([_line("IT Consulting", "8", "50.00", "400.00")])
        con = _contract([_line("IT Consulting", "8", "50.00", "400.00")])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        qty_findings = [f for f in findings if f.finding_type == "QUANTITY"]

        assert len(qty_findings) == 1
        f = qty_findings[0]
        assert f.status == "Anomaly"
        assert "10" in f.motivation and "8" in f.motivation
        assert f.impact_value is not None
        assert f.impact_value.amount == Decimal("100.00")  # 2 × 50.00

    def test_deficit_quantity_is_anomalia_no_impact(self):
        inv = _invoice([_line("IT Consulting", "6", "50.00", "300.00")])
        ddt_ = _ddt([_line("IT Consulting", "8", "50.00", "400.00")])
        con = _contract([_line("IT Consulting", "8", "50.00", "400.00")])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        qty_findings = [f for f in findings if f.finding_type == "QUANTITY"]

        assert len(qty_findings) == 1
        f = qty_findings[0]
        assert f.status == "Anomaly"
        assert f.impact_value is None  # underbilling: no overbilling impact

    def test_item_absent_from_ddt_not_flagged_as_quantity_mismatch(self):
        # Item only in invoice and contract but not in DDT → extra item, not qty mismatch
        inv = _invoice([_line("Extra service", "1", "100.00", "100.00")])
        ddt_ = _ddt([])
        con = _contract([_line("Extra service", "1", "100.00", "100.00")])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        qty_findings = [f for f in findings if f.finding_type == "QUANTITY"]
        assert qty_findings == []

    def test_description_normalized_case_insensitive(self):
        inv = _invoice([_line("consulenza IT", "5", "50.00", "250.00")])
        ddt_ = _ddt([_line("IT Consulting", "5", "50.00", "250.00")])
        con = _contract([_line("CONSULENZA IT", "5", "50.00", "250.00")])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        qty_findings = [f for f in findings if f.finding_type == "QUANTITY"]
        assert qty_findings == []


# ---------------------------------------------------------------------------
# Price checks
# ---------------------------------------------------------------------------

class TestPriceCheck:
    def test_matching_price_produces_no_finding(self):
        inv = _invoice([_line("Support", "1", "200.00", "200.00")])
        ddt_ = _ddt([_line("Support", "1", "200.00", "200.00")])
        con = _contract([_line("Support", "1", "200.00", "200.00")])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        price_findings = [f for f in findings if f.finding_type == "PRICING"]
        assert price_findings == []

    def test_slight_price_increase_is_dubbio(self):
        # 1% above contract price: within 2% threshold → Dubbio
        inv = _invoice([_line("Support", "1", "202.00", "202.00")])
        ddt_ = _ddt([_line("Support", "1", "202.00", "202.00")])
        con = _contract([_line("Support", "1", "200.00", "200.00")])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        price_findings = [f for f in findings if f.finding_type == "PRICING"]

        assert len(price_findings) == 1
        assert price_findings[0].status == "Warning"

    def test_severe_price_increase_is_anomalia(self):
        # 10% above contract price: above 2% threshold → Anomalia
        inv = _invoice([_line("Support", "1", "220.00", "220.00")])
        ddt_ = _ddt([_line("Support", "1", "220.00", "220.00")])
        con = _contract([_line("Support", "1", "200.00", "200.00")])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        price_findings = [f for f in findings if f.finding_type == "PRICING"]

        assert len(price_findings) == 1
        f = price_findings[0]
        assert f.status == "Anomaly"
        assert "220" in f.motivation and "200" in f.motivation
        assert f.impact_value is not None
        assert f.impact_value.amount == Decimal("20.00")  # 1 unit × 20.00 delta

    def test_price_impact_multiplied_by_quantity(self):
        # 3 units × (110 - 100) = 30.00 impact
        inv = _invoice([_line("Technical Support", "3", "110.00", "330.00")])
        ddt_ = _ddt([_line("Technical Support", "3", "110.00", "330.00")])
        con = _contract([_line("Technical Support", "3", "100.00", "300.00")])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        price_findings = [f for f in findings if f.finding_type == "PRICING"]

        assert len(price_findings) == 1
        assert price_findings[0].impact_value.amount == Decimal("30.00")

    def test_custom_threshold_respected(self):
        strict_engine = AuditEngine(price_dubbio_threshold=Decimal("0.005"))
        # 1% deviation → Anomalia with strict threshold of 0.5%
        inv = _invoice([_line("Support", "1", "201.00", "201.00")])
        ddt_ = _ddt([_line("Support", "1", "201.00", "201.00")])
        con = _contract([_line("Support", "1", "200.00", "200.00")])

        findings = strict_engine.cross_check_ddt_invoice(ddt_, inv, con)
        price_findings = [f for f in findings if f.finding_type == "PRICING"]
        assert price_findings[0].status == "Anomaly"


# ---------------------------------------------------------------------------
# Extra item checks
# ---------------------------------------------------------------------------

class TestExtraItemCheck:
    def test_extra_item_not_in_ddt_or_contract_is_anomalia(self):
        inv = _invoice([_line("Mystery item", "1", "500.00", "500.00")])
        ddt_ = _ddt([])
        con = _contract([])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        extra = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert len(extra) == 1
        f = extra[0]
        assert f.status == "Anomaly"
        assert "Mystery item" in f.motivation
        assert f.impact_value.amount == Decimal("500.00")

    def test_item_in_contract_only_is_not_flagged_as_extra(self):
        inv = _invoice([_line("Consulenza", "1", "100.00", "100.00")])
        ddt_ = _ddt([])
        con = _contract([_line("Consulenza", "1", "100.00", "100.00")])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        extra = [f for f in findings if f.finding_type == "CONTRACTUAL"]
        assert extra == []

    def test_item_in_ddt_only_is_not_flagged_as_extra(self):
        inv = _invoice([_line("Shipping", "1", "80.00", "80.00")])
        ddt_ = _ddt([_line("Shipping", "1", "80.00", "80.00")])
        con = _contract([])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        extra = [f for f in findings if f.finding_type == "CONTRACTUAL"]
        assert extra == []

    def test_amount_match_on_unit_price_is_ok(self):
        """Description differs but unit_price matches a contract rate → OK badge."""
        inv = _invoice([_line("Canone Manutenzione Mensile Esteso", "1", "350.00", "350.00")])
        ddt_ = _ddt([])
        con = _contract([_line("Canone base mensile", "1", "350.00", "350.00")])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert len(contractual) == 1
        f = contractual[0]
        assert f.status == "OK"
        assert len(f.evidence) >= 2

    def test_amount_match_motivation_references_both_descriptions(self):
        """OK motivation must cite both invoice and contract line descriptions."""
        inv = _invoice([_line("Abbonamento Premium Plus", "1", "199.00", "199.00")])
        ddt_ = _ddt([])
        con = _contract([_line("Piano Standard", "1", "199.00", "199.00")])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        f = next(x for x in findings if x.finding_type == "CONTRACTUAL")

        assert f.status == "OK"
        assert "Abbonamento Premium Plus" in f.motivation
        assert "Piano Standard" in f.motivation
        assert "199" in f.motivation

    def test_amount_match_on_line_total_is_ok(self):
        """Truly different descriptions, same amount → OK (qty-1 subscription reconciled by amount)."""
        inv = _invoice([_line("Pagamento Abbonamento Enterprise", "1", "600.00", "600.00")])
        ddt_ = _ddt([])
        con = _contract([_line("Servizio annuale", "1", "600.00", "600.00")])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]
        assert contractual[0].status == "OK"

    def test_no_description_no_amount_match_remains_anomaly(self):
        """No description match AND no amount match → Anomaly with impact."""
        inv = _invoice([_line("Voce sconosciuta", "1", "999.00", "999.00")])
        ddt_ = _ddt([])
        con = _contract([_line("Canone base", "1", "350.00", "350.00")])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert len(contractual) == 1
        f = contractual[0]
        assert f.status == "Anomaly"
        assert f.impact_value is not None
        assert f.impact_value.amount == Decimal("999.00")

    def test_zero_price_invoice_item_no_description_match_is_ok(self):
        """Zero-value invoice item with no description match → OK (informative), not Anomaly."""
        inv = _invoice([_line("Ispezione trimestrale inclusa", "1", "0.00", "0.00")])
        ddt_ = _ddt([])
        con = _contract([])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert len(contractual) == 1
        f = contractual[0]
        assert f.status == "OK"
        assert "0.00" in f.motivation
        assert "Ispezione trimestrale inclusa" in f.motivation
        assert len(f.evidence) >= 1

    def test_zero_price_invoice_item_not_in_anomaly_counter(self):
        """Zero-value items must not increment the Anomaly counter."""
        inv = _invoice([
            _line("Calibrazione inclusa", "1", "0.00", "0.00"),
            _line("Canone base",          "1", "350.00", "350.00"),
        ])
        ddt_ = _ddt([])
        con = _contract([_line("Canone base", "1", "350.00", "350.00")])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        assert all(f.status != "Anomaly" for f in findings)

    def test_zero_unit_price_also_caught_as_informative(self):
        """unit_price=0.00 (regardless of total) triggers the zero-value guard."""
        inv = _invoice([_line("Nota piano manutenzione", "3", "0.00", "0.00")])
        ddt_ = _ddt([])
        con = _contract([_line("Canone mensile", "1", "200.00", "200.00")])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]
        assert contractual[0].status == "OK"

    def test_zero_price_contract_item_not_used_in_amount_matching(self):
        """A zero-priced contract item must not give a false amount match for a non-zero invoice item."""
        inv = _invoice([_line("Voce costosa", "1", "999.00", "999.00")])
        ddt_ = _ddt([])
        con = _contract([_line("Riga vuota", "1", "0.00", "0.00")])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]
        assert len(contractual) == 1
        assert contractual[0].status == "Anomaly"


# ---------------------------------------------------------------------------
# VAT checks
# ---------------------------------------------------------------------------

class TestVATCheck:
    def test_it_correct_vat_22_is_ok(self):
        inv = _invoice([], vat_rate="22.00", jurisdiction="IT")
        ddt_ = _ddt([])
        con = _contract([], jurisdiction="IT")

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        vat_findings = [f for f in findings if f.finding_type == "FISCAL"]

        assert len(vat_findings) == 1
        assert vat_findings[0].status == "OK"
        assert "22" in vat_findings[0].motivation

    def test_ch_correct_vat_81_is_ok(self):
        inv = Invoice(
            meta=_meta(jurisdiction="CH", source_file="invoice_ch.pdf"),
            doc_type="INVOICE",
            vendor=_fr("Lieferant AG"),
            client=_fr("Kunde GmbH"),
            currency=_fr("CHF"),
            line_items=[],
            subtotal=Money(amount=Decimal("1000.00"), currency="CHF", country="CH"),
            vat_amount=Money(amount=Decimal("81.00"), currency="CHF", country="CH"),
            total=Money(amount=Decimal("1081.00"), currency="CHF", country="CH"),
            vat_rate=_fr(Decimal("8.10")),
        )
        ddt_ = DDT(
            meta=_meta(jurisdiction="CH"),
            doc_type="DDT",
            vendor=_fr("Lieferant AG"),
            client=_fr("Kunde GmbH"),
            currency=_fr("CHF"),
        )
        con = Contract(
            meta=_meta(jurisdiction="CH"),
            doc_type="CONTRACT",
            vendor=_fr("Lieferant AG"),
            client=_fr("Kunde GmbH"),
            currency=_fr("CHF"),
        )

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        vat_findings = [f for f in findings if f.finding_type == "FISCAL"]

        assert len(vat_findings) == 1
        assert vat_findings[0].status == "OK"
        assert "8.10" in vat_findings[0].motivation or "8.1" in vat_findings[0].motivation

    def test_it_wrong_vat_produces_anomalia_with_impact(self):
        # Invoice claims 10% VAT but IT standard is 22%
        inv = _invoice([], vat_rate="10.00", jurisdiction="IT", subtotal="1000.00")
        ddt_ = _ddt([])
        con = _contract([], jurisdiction="IT")

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        vat_findings = [f for f in findings if f.finding_type == "FISCAL"]

        assert len(vat_findings) == 1
        f = vat_findings[0]
        assert f.status == "Anomaly"
        assert "10" in f.motivation and "22" in f.motivation
        assert f.impact_value is not None
        # Expected VAT: 1000 × 22% = 220; Actual VAT: 1000 × 10% = 100; delta = 120
        assert f.impact_value.amount == Decimal("120.00")

    def test_unknown_jurisdiction_produces_dubbio(self):
        inv = _invoice([], vat_rate="20.00", jurisdiction="UNKNOWN")
        ddt_ = _ddt([])
        con = _contract([])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        vat_findings = [f for f in findings if f.finding_type == "FISCAL"]

        assert len(vat_findings) == 1
        assert vat_findings[0].status == "Warning"


# ---------------------------------------------------------------------------
# Model constraint guards (should never be bypassed)
# ---------------------------------------------------------------------------

class TestModelConstraints:
    def test_ok_finding_without_evidence_raises(self):
        from schemas.audit import Finding

        with pytest.raises(Exception, match="evidence"):
            Finding(
                finding_type="FISCAL",
                status="OK",
                motivation="All good",
                evidence=[],
            )

    def test_anomaly_without_motivation_raises(self):
        from schemas.audit import Finding
        from schemas.base import Evidence

        with pytest.raises(Exception, match="motivation"):
            Finding(
                finding_type="FISCAL",
                status="Anomaly",
                motivation="  ",
                evidence=[Evidence(source_document="f.pdf", extracted_text="x")],
            )


# ---------------------------------------------------------------------------
# Fuzzy amount matching (ERP rounding tolerance)
# ---------------------------------------------------------------------------

class TestFuzzyAmountMatching:
    def test_exact_amount_match_no_fuzzy_label(self):
        """Exact amount match (Δ = 0.00) with truly different descriptions → OK without ERP label."""
        inv = _invoice([_line("Addebito mensile servizi", "1", "350.00", "350.00")])
        con = _contract([_line("Canone mensile base", "1", "350.00", "350.00")])

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert len(contractual) == 1
        assert contractual[0].status == "OK"
        assert "Arrotondamento ERP" not in contractual[0].motivation

    def test_fuzzy_match_within_tolerance_is_ok(self):
        """Δ = 0.03 (within CH 0.05 Rappen tolerance) → OK with ERP label."""
        inv = _invoice([_line("Servizio Tecnico Extended", "1", "350.03", "350.03")], jurisdiction="CH")
        con = _contract([_line("Servizio Tecnico Base", "1", "350.00", "350.00")], jurisdiction="CH")

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert len(contractual) == 1
        f = contractual[0]
        assert f.status == "OK"
        assert "Arrotondamento ERP" in f.motivation
        assert "350.03" in f.motivation or "350" in f.motivation

    def test_fuzzy_match_at_exact_tolerance_boundary_is_ok(self):
        """Δ = 0.05 (exactly at CH Rappen tolerance boundary) → OK with ERP label."""
        inv = _invoice([_line("Canone Q4", "1", "200.05", "200.05")], jurisdiction="CH")
        con = _contract([_line("Canone trimestrale", "1", "200.00", "200.00")], jurisdiction="CH")

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert contractual[0].status == "OK"
        assert "Arrotondamento ERP" in contractual[0].motivation

    def test_delta_just_above_tolerance_is_anomaly(self):
        """Δ = 0.06 (above 0.05 tolerance) → no amount match → Anomaly."""
        inv = _invoice([_line("Canone Speciale", "1", "200.06", "200.06")])
        con = _contract([_line("Canone trimestrale", "1", "200.00", "200.00")])

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert len(contractual) == 1
        assert contractual[0].status == "Anomaly"

    def test_fuzzy_match_underbilling_also_accepted(self):
        """Δ = -0.04 (invoice cheaper, within CH 0.05 tolerance) → OK with ERP label."""
        inv = _invoice([_line("Manutenzione ridotta", "1", "499.96", "499.96")], jurisdiction="CH")
        con = _contract([_line("Manutenzione standard", "1", "500.00", "500.00")], jurisdiction="CH")

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert contractual[0].status == "OK"
        assert "Arrotondamento ERP" in contractual[0].motivation

    def test_fuzzy_match_has_two_evidence_entries(self):
        """A fuzzy amount match (CH, Δ=0.02 ≤ 0.05) must include evidence from both documents."""
        inv = _invoice([_line("Canone Plus v2", "1", "350.02", "350.02")], jurisdiction="CH")
        con = _contract([_line("Canone base v1", "1", "350.00", "350.00")], jurisdiction="CH")

        findings = engine.cross_check(inv, con)
        f = next(x for x in findings if x.finding_type == "CONTRACTUAL")

        assert f.status == "OK"
        assert len(f.evidence) >= 2


# ---------------------------------------------------------------------------
# SKU / product-code matching
# ---------------------------------------------------------------------------

class TestSKUMatching:
    def test_sku_in_description_matches_contract_item(self):
        """SKU matches but price 999.99 vs 450.00 deviates far beyond tolerance → PRICING Anomaly."""
        inv = _invoice([_line("Product No. 500026135 - Service Bundle", "1", "999.99", "999.99")])
        con = _contract([_line("Manutenzione 500026135 Piano Premium", "1", "450.00", "450.00")])

        findings = engine.cross_check(inv, con)
        pricing = [f for f in findings if f.finding_type == "PRICING"]

        assert len(pricing) == 1
        f = pricing[0]
        assert f.status == "Anomaly"
        assert "500026135" in f.motivation or "Match SKU" in f.motivation
        assert "999.99" in f.motivation and "450.00" in f.motivation

    def test_explicit_sku_field_matches(self):
        """Explicit sku field on both items → OK with SKU label."""
        inv_item = LineItem(
            description="Replacement Part",
            quantity=Decimal("1"),
            unit_price=_money("75.00"),
            total=_money("75.00"),
            sku="ABC12345",
        )
        con_item = LineItem(
            description="Ricambio speciale",
            quantity=Decimal("1"),
            unit_price=_money("75.00"),
            total=_money("75.00"),
            sku="ABC12345",
        )
        inv = _invoice([inv_item])
        con = _contract([con_item])

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert len(contractual) == 1
        f = contractual[0]
        assert f.status == "OK"
        assert "SKU" in f.motivation or "ABC12345" in f.motivation

    def test_sku_mismatch_falls_through_to_anomaly(self):
        """Invoice SKU not found in contract → Anomaly (no false OK)."""
        inv = _invoice([_line("Product 999999999 Mystery", "1", "500.00", "500.00")])
        con = _contract([_line("Canone base 123456789", "1", "350.00", "350.00")])

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert len(contractual) == 1
        assert contractual[0].status == "Anomaly"

    def test_short_numeric_token_not_treated_as_sku(self):
        """A 4-digit year token ('2024') is too short to be a SKU — no false match."""
        inv = _invoice([_line("Service 2024 edition", "1", "300.00", "300.00")])
        con = _contract([_line("Item 2024 standard", "1", "350.00", "350.00")])

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        # Amount mismatch (300 vs 350, Δ=50 >> 0.05) → Anomaly
        assert contractual[0].status == "Anomaly"

    def test_sku_match_has_evidence_from_both_documents(self):
        """SKU explicit field matches but price 120.00 vs 100.00 (20%) → PRICING Anomaly with 2 evidences."""
        inv_item = LineItem(
            description="Manutenzione Preventiva Strutture",
            quantity=Decimal("1"),
            unit_price=_money("120.00"),
            total=_money("120.00"),
            sku="500026135",
        )
        con_item = LineItem(
            description="Canone Piano Annuale Infrastruttura",
            quantity=Decimal("1"),
            unit_price=_money("100.00"),
            total=_money("100.00"),
            sku="500026135",
        )
        inv = _invoice([inv_item])
        con = _contract([con_item])

        findings = engine.cross_check(inv, con)
        f = next(x for x in findings if x.finding_type == "PRICING")

        assert f.status == "Anomaly"
        assert len(f.evidence) >= 2

    def test_sku_match_with_matching_price_is_ok(self):
        """SKU matches and price is identical → OK (price sanity passed)."""
        inv_item = LineItem(
            description="Prodotto Speciale Q4 Bundle",
            quantity=Decimal("1"),
            unit_price=_money("200.00"),
            total=_money("200.00"),
            sku="AB123456",
        )
        con_item = LineItem(
            description="Canone Mensile Infrastruttura",
            quantity=Decimal("1"),
            unit_price=_money("200.00"),
            total=_money("200.00"),
            sku="AB123456",
        )
        inv = _invoice([inv_item])
        con = _contract([con_item])

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert len(contractual) == 1
        f = contractual[0]
        assert f.status == "OK"
        assert "AB123456" in f.motivation or "SKU" in f.motivation


# ---------------------------------------------------------------------------
# Structural row filter (ERP subtotals / tax rows)
# ---------------------------------------------------------------------------

class TestStructuralRowFilter:
    def test_total_row_produces_no_finding(self):
        """A line item whose description is 'Total' is silently skipped."""
        inv = _invoice([
            _line("Canone mensile", "1", "100.00", "100.00"),
            _line("Total", "1", "100.00", "100.00"),
        ])
        con = _contract([_line("Canone mensile", "1", "100.00", "100.00")])

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]
        assert contractual == []

    def test_iva_row_produces_no_finding(self):
        """'IVA 22%' is an ERP tax row and must not generate a CONTRACTUAL finding."""
        inv = _invoice([
            _line("Servizio base", "1", "200.00", "200.00"),
            _line("IVA 22%", "1", "44.00", "44.00"),
        ])
        con = _contract([_line("Servizio base", "1", "200.00", "200.00")])

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]
        assert contractual == []

    def test_mwst_row_produces_no_finding(self):
        """'MwSt 8.1%' (Swiss VAT ERP row) must not generate a CONTRACTUAL finding."""
        inv = _invoice([
            _line("Service de maintenance", "1", "500.00", "500.00"),
            _line("MwSt 8.1%", "1", "40.50", "40.50"),
        ])
        con = _contract([_line("Service de maintenance", "1", "500.00", "500.00")])

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]
        assert contractual == []

    def test_remise_row_produces_no_finding(self):
        """'Remise 5%' discount summary row must be silently skipped."""
        inv = _invoice([
            _line("Software license", "1", "1000.00", "1000.00"),
            _line("Remise 5%", "1", "50.00", "50.00"),
        ])
        con = _contract([_line("Software license", "1", "1000.00", "1000.00")])

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]
        assert contractual == []

    def test_subtotal_row_produces_no_finding(self):
        """'Subtotal' row is a structural summary and must be skipped."""
        inv = _invoice([
            _line("Consulenza", "1", "500.00", "500.00"),
            _line("Subtotal", "1", "500.00", "500.00"),
        ])
        con = _contract([_line("Consulenza", "1", "500.00", "500.00")])

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]
        assert contractual == []

    def test_legitimate_service_with_word_total_in_name_not_filtered(self):
        """'Total maintenance fee monthly' is a service description — must NOT be filtered."""
        inv = _invoice([_line("Total maintenance fee monthly", "1", "300.00", "300.00")])
        con = _contract([_line("Total maintenance fee monthly", "1", "300.00", "300.00")])

        findings = engine.cross_check(inv, con)
        # Description matches contract exactly → no CONTRACTUAL finding, no false filter
        assert all(f.status != "Anomaly" for f in findings)

    def test_structural_filter_does_not_affect_pricing_check(self):
        """A structural row in invoice also skips the price cross-check — no PRICING finding."""
        inv = _invoice([
            _line("Canone mensile", "1", "200.00", "200.00"),
            _line("Total", "1", "9999.00", "9999.00"),
        ])
        ddt_ = _ddt([_line("Canone mensile", "1", "200.00", "200.00")])
        con = _contract([_line("Canone mensile", "1", "200.00", "200.00")])

        findings = engine.cross_check_ddt_invoice(ddt_, inv, con)
        pricing = [f for f in findings if f.finding_type == "PRICING"]
        assert pricing == []


# ---------------------------------------------------------------------------
# Fuzzy description matching (token-subset + SequenceMatcher)
# ---------------------------------------------------------------------------

class TestFuzzyDescriptionMatching:
    def test_token_subset_match_produces_warning(self):
        """Token-subset match ('Canone di manutenzione' ⊆ 'Canone mensile di manutenzione') → Warning."""
        inv = _invoice([_line("Canone mensile di manutenzione", "1", "350.00", "350.00")])
        con = _contract([_line("Canone di manutenzione", "1", "350.00", "350.00")])

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert len(contractual) == 1
        f = contractual[0]
        assert f.status == "Warning"
        assert "Revisione Manuale" in f.motivation
        assert "Canone mensile di manutenzione" in f.motivation
        assert "Canone di manutenzione" in f.motivation

    def test_token_subset_match_with_price_deviation_includes_price_in_motivation(self):
        """Fuzzy description match + price deviation → Warning with price context."""
        inv = _invoice([_line("Canone mensile di manutenzione", "1", "420.00", "420.00")])
        con = _contract([_line("Canone di manutenzione", "1", "350.00", "350.00")])

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert len(contractual) == 1
        f = contractual[0]
        assert f.status == "Warning"
        assert "420" in f.motivation and "350" in f.motivation
        assert f.impact_value is not None
        assert f.impact_value.amount == Decimal("70.00")

    def test_exact_match_produces_no_contractual_finding(self):
        """Exact description match → no CONTRACTUAL finding (prices identical, no PRICING either)."""
        inv = _invoice([_line("Manutenzione impianti", "1", "350.00", "350.00")])
        con = _contract([_line("Manutenzione impianti", "1", "350.00", "350.00")])

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]
        assert contractual == []

    def test_no_description_match_remains_anomaly(self):
        """No description, SKU, or amount match → Anomaly unchanged."""
        inv = _invoice([_line("Voce completamente sconosciuta", "1", "777.00", "777.00")])
        con = _contract([_line("Canone base annuale", "1", "200.00", "200.00")])

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert len(contractual) == 1
        assert contractual[0].status == "Anomaly"
        assert contractual[0].impact_value.amount == Decimal("777.00")


# ---------------------------------------------------------------------------
# Region-aware ERP rounding tolerance (IT=0.01, CH=0.05, UNKNOWN=0.02)
# ---------------------------------------------------------------------------

class TestRegionAwareTolerance:
    def test_it_delta_002_is_anomaly(self):
        """IT invoice: delta 0.02 > IT tolerance 0.01 → no amount match → Anomaly."""
        inv = _invoice(
            [_line("Servizio fatturato Italia", "1", "100.02", "100.02")],
            jurisdiction="IT",
        )
        con = _contract([_line("Altro servizio IT", "1", "100.00", "100.00")], jurisdiction="IT")

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert len(contractual) == 1
        assert contractual[0].status == "Anomaly"

    def test_ch_delta_004_is_ok(self):
        """CH invoice: delta 0.04 ≤ CH tolerance 0.05 → fuzzy amount match → OK."""
        inv = Invoice(
            meta=_meta(jurisdiction="CH", source_file="inv_ch.pdf"),
            doc_type="INVOICE",
            vendor=_fr("Lieferant AG"),
            client=_fr("Kunde GmbH"),
            currency=_fr("CHF"),
            line_items=[
                LineItem(
                    description="Wartung Gebäudetechnik",
                    quantity=Decimal("1"),
                    unit_price=Money(amount=Decimal("500.04"), currency="CHF", country="CH"),
                    total=Money(amount=Decimal("500.04"), currency="CHF", country="CH"),
                )
            ],
            subtotal=Money(amount=Decimal("500.04"), currency="CHF", country="CH"),
            vat_amount=Money(amount=Decimal("40.50"), currency="CHF", country="CH"),
            total=Money(amount=Decimal("540.54"), currency="CHF", country="CH"),
            vat_rate=_fr(Decimal("8.10")),
        )
        con = Contract(
            meta=_meta(jurisdiction="CH"),
            doc_type="CONTRACT",
            vendor=_fr("Lieferant AG"),
            client=_fr("Kunde GmbH"),
            currency=_fr("CHF"),
            line_items=[
                LineItem(
                    description="Pauschale Unterhalt",
                    quantity=Decimal("1"),
                    unit_price=Money(amount=Decimal("500.00"), currency="CHF", country="CH"),
                    total=Money(amount=Decimal("500.00"), currency="CHF", country="CH"),
                )
            ],
        )

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert len(contractual) == 1
        assert contractual[0].status == "OK"
        assert "Arrotondamento ERP" in contractual[0].motivation

    def test_unknown_delta_002_is_ok(self):
        """UNKNOWN jurisdiction: delta 0.02 ≤ UNKNOWN tolerance 0.02 → OK (boundary)."""
        inv = _invoice(
            [_line("Bollatura Annuale Registro", "1", "200.02", "200.02")],
            jurisdiction="UNKNOWN",
        )
        con = _contract([_line("Assistenza Tecnica Continuativa", "1", "200.00", "200.00")])

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert len(contractual) == 1
        assert contractual[0].status == "OK"


# ---------------------------------------------------------------------------
# Rounding adjustment detection (Arrondis / Rounding / Arrotondamento)
# ---------------------------------------------------------------------------

class TestRoundingAdjustmentDetection:
    """Rounding adjustment line items (e.g., Swiss Rappen rounding) must not be Anomaly."""

    def _ch_invoice(self, line_items: list[LineItem]) -> Invoice:
        return Invoice(
            meta=_meta(jurisdiction="CH", source_file="inv_ch.pdf"),
            doc_type="INVOICE",
            vendor=_fr("Fournisseur SA"),
            client=_fr("Composites Busch SA"),
            currency=_fr("CHF"),
            line_items=line_items,
            subtotal=Money(amount=Decimal("9990.00"), currency="CHF", country="CH"),
            vat_amount=Money(amount=Decimal("809.19"), currency="CHF", country="CH"),
            total=Money(amount=Decimal("10799.15"), currency="CHF", country="CH"),
            vat_rate=_fr(Decimal("8.10")),
        )

    def _ch_contract(self, line_items: list[LineItem]) -> Contract:
        return Contract(
            meta=_meta(jurisdiction="CH"),
            doc_type="CONTRACT",
            vendor=_fr("Fournisseur SA"),
            client=_fr("Composites Busch SA"),
            currency=_fr("CHF"),
            line_items=line_items,
        )

    def _chf_item(self, description: str, amount: str) -> LineItem:
        return LineItem(
            description=description,
            quantity=Decimal("1"),
            unit_price=Money(amount=Decimal(amount), currency="CHF", country="CH"),
            total=Money(amount=Decimal(amount), currency="CHF", country="CH"),
        )

    def test_arrondis_negative_ch_is_ok(self):
        """Arrondis CHF -0.04 (client-favorable) → OK finding, not Anomaly."""
        inv = self._ch_invoice([self._chf_item("Arrondis", "-0.04")])
        con = self._ch_contract([])

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert len(contractual) == 1
        f = contractual[0]
        assert f.status == "OK"
        assert "favore del cliente" in f.motivation
        assert len(f.evidence) >= 1

    def test_arrondi_variant_without_s_is_ok(self):
        """'Arrondi' (without final s) also detected as rounding adjustment → OK."""
        inv = self._ch_invoice([self._chf_item("Arrondi", "-0.02")])
        con = self._ch_contract([])

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert len(contractual) == 1
        assert contractual[0].status == "OK"

    def test_rounding_english_negative_is_ok(self):
        """English 'Rounding' with negative amount → OK."""
        inv = self._ch_invoice([self._chf_item("Rounding", "-0.02")])
        con = self._ch_contract([])

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert len(contractual) == 1
        assert contractual[0].status == "OK"

    def test_arrondis_positive_within_tolerance_is_warning(self):
        """Arrondis CHF +0.03 (positive, within tolerance 0.05) → Warning, not Anomaly."""
        inv = self._ch_invoice([self._chf_item("Arrondis", "0.03")])
        con = self._ch_contract([])

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert len(contractual) == 1
        f = contractual[0]
        assert f.status == "Warning"
        assert f.impact_value is not None
        assert f.impact_value.amount == Decimal("0.03")

    def test_arrondis_over_tolerance_falls_through_to_anomaly(self):
        """Arrondis CHF -0.10 (exceeds CH tolerance 0.05) → Anomaly (too large for rounding)."""
        inv = self._ch_invoice([self._chf_item("Arrondis", "-0.10")])
        con = self._ch_contract([])

        findings = engine.cross_check(inv, con)
        contractual = [f for f in findings if f.finding_type == "CONTRACTUAL"]

        assert len(contractual) == 1
        assert contractual[0].status == "Anomaly"

    def test_busch_composites_exact_scenario(self):
        """Exact Busch Composites scenario: Arrondis CHF -0.04 must produce no Anomaly."""
        inv = self._ch_invoice([self._chf_item("Arrondis", "-0.04")])
        con = self._ch_contract([])

        findings = engine.cross_check(inv, con)
        assert not any(f.status == "Anomaly" for f in findings)
