"""
tests/test_fiscal_rules.py — unit tests for src/fiscal_rules.py

Coverage:
  TestTaxRuleModel         — TaxRule creation, rate as Decimal, valid_to optional
  TestGetApplicableTaxRule — date filtering, country filtering, most-recent selection
  TestIsVerified           — VERIFIED / UNVERIFIED
  TestValidateVATRate      — OK (IT), OK (CH), MISMATCH, RULE_NOT_FOUND, UNVERIFIED,
                             valid_from / valid_to boundary dates
  TestNoFloat              — zero float in fiscal_rules module and return values
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from src.fiscal_rules import (
    KNOWN_TAX_RULES,
    TaxRule,
    VATValidationResult,
    get_applicable_tax_rule,
    is_tax_rule_verified,
    validate_vat_rate,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _unverified_it_rule(
    rate: str = "0.22",
    valid_from: date = date(2020, 1, 1),
    valid_to: date | None = None,
) -> TaxRule:
    return TaxRule(
        country="IT",
        tax_type="VAT",
        rate=Decimal(rate),
        valid_from=valid_from,
        valid_to=valid_to,
        source="test-only unverified rule",
        verification_status="UNVERIFIED",
    )


# ── TaxRule model ─────────────────────────────────────────────────────────────

class TestTaxRuleModel:

    def test_it_rule_creation(self):
        rule = TaxRule(
            country="IT",
            tax_type="VAT",
            rate=Decimal("0.22"),
            valid_from=date(2013, 10, 1),
            source="test",
            verification_status="VERIFIED",
        )
        assert rule.country == "IT"
        assert rule.rate == Decimal("0.22")

    def test_ch_rule_creation(self):
        rule = TaxRule(
            country="CH",
            tax_type="VAT",
            rate=Decimal("0.081"),
            valid_from=date(2024, 1, 1),
            source="test",
            verification_status="VERIFIED",
        )
        assert rule.country == "CH"
        assert rule.rate == Decimal("0.081")

    def test_rate_is_decimal_not_float(self):
        rule = TaxRule(
            country="IT",
            tax_type="VAT",
            rate=Decimal("0.22"),
            valid_from=date(2013, 10, 1),
            source="test",
            verification_status="VERIFIED",
        )
        assert isinstance(rule.rate, Decimal)
        assert type(rule.rate) is not float

    def test_valid_to_is_optional(self):
        rule = TaxRule(
            country="IT",
            tax_type="VAT",
            rate=Decimal("0.22"),
            valid_from=date(2013, 10, 1),
            source="test",
            verification_status="VERIFIED",
        )
        assert rule.valid_to is None

    def test_valid_to_can_be_set(self):
        rule = TaxRule(
            country="IT",
            tax_type="VAT",
            rate=Decimal("0.22"),
            valid_from=date(2013, 10, 1),
            valid_to=date(2025, 12, 31),
            source="test",
            verification_status="VERIFIED",
        )
        assert rule.valid_to == date(2025, 12, 31)


# ── get_applicable_tax_rule ───────────────────────────────────────────────────

class TestGetApplicableTaxRule:

    def test_it_rule_found_today(self):
        rule = get_applicable_tax_rule("IT", date(2025, 6, 1))
        assert rule is not None
        assert rule.country == "IT"

    def test_ch_rule_found_today(self):
        rule = get_applicable_tax_rule("CH", date(2025, 6, 1))
        assert rule is not None
        assert rule.country == "CH"

    def test_unknown_country_returns_none(self):
        assert get_applicable_tax_rule("XX", date(2025, 6, 1)) is None

    def test_date_before_it_valid_from_returns_none(self):
        assert get_applicable_tax_rule("IT", date(2013, 9, 30)) is None

    def test_date_on_it_valid_from_returns_rule(self):
        rule = get_applicable_tax_rule("IT", date(2013, 10, 1))
        assert rule is not None
        assert rule.country == "IT"

    def test_date_after_valid_to_returns_none(self):
        limited_rule = TaxRule(
            country="IT",
            tax_type="VAT",
            rate=Decimal("0.22"),
            valid_from=date(2020, 1, 1),
            valid_to=date(2022, 12, 31),
            source="test",
            verification_status="VERIFIED",
        )
        assert get_applicable_tax_rule("IT", date(2023, 1, 1), rules=[limited_rule]) is None

    def test_date_within_valid_to_returns_rule(self):
        limited_rule = TaxRule(
            country="IT",
            tax_type="VAT",
            rate=Decimal("0.22"),
            valid_from=date(2020, 1, 1),
            valid_to=date(2022, 12, 31),
            source="test",
            verification_status="VERIFIED",
        )
        assert get_applicable_tax_rule("IT", date(2021, 6, 1), rules=[limited_rule]) is not None

    def test_selects_most_recent_rule(self):
        old_rule = TaxRule(
            country="IT", tax_type="VAT", rate=Decimal("0.21"),
            valid_from=date(2011, 9, 1), source="old", verification_status="VERIFIED",
        )
        new_rule = TaxRule(
            country="IT", tax_type="VAT", rate=Decimal("0.22"),
            valid_from=date(2013, 10, 1), source="new", verification_status="VERIFIED",
        )
        result = get_applicable_tax_rule("IT", date(2025, 1, 1), rules=[old_rule, new_rule])
        assert result is not None
        assert result.rate == Decimal("0.22")


# ── is_tax_rule_verified ──────────────────────────────────────────────────────

class TestIsVerified:

    def test_verified_returns_true(self):
        rule = TaxRule(
            country="IT", tax_type="VAT", rate=Decimal("0.22"),
            valid_from=date(2013, 10, 1), source="test",
            verification_status="VERIFIED",
        )
        assert is_tax_rule_verified(rule) is True

    def test_unverified_returns_false(self):
        assert is_tax_rule_verified(_unverified_it_rule()) is False


# ── validate_vat_rate ─────────────────────────────────────────────────────────

class TestValidateVATRate:

    def test_vat_it_corretta(self):
        assert validate_vat_rate("IT", date(2025, 1, 1), Decimal("22.00")).status == "OK"

    def test_vat_ch_corretta(self):
        assert validate_vat_rate("CH", date(2025, 1, 1), Decimal("8.10")).status == "OK"

    def test_vat_errata_it(self):
        assert validate_vat_rate("IT", date(2025, 1, 1), Decimal("10.00")).status == "MISMATCH"

    def test_vat_errata_ch(self):
        assert validate_vat_rate("CH", date(2025, 1, 1), Decimal("22.00")).status == "MISMATCH"

    def test_vat_rule_mancante_paese_sconosciuto(self):
        assert validate_vat_rate("IT", date(2010, 1, 1), Decimal("20.00")).status == "RULE_NOT_FOUND"

    def test_vat_rule_mancante_data_precedente(self):
        assert validate_vat_rate("IT", date(2012, 6, 1), Decimal("21.00")).status == "RULE_NOT_FOUND"

    def test_vat_rule_unverified(self):
        result = validate_vat_rate(
            "IT", date(2025, 1, 1), Decimal("22.00"),
            rules=[_unverified_it_rule()],
        )
        assert result.status == "UNVERIFIED"

    def test_valid_from_boundary_inclusive(self):
        assert validate_vat_rate("IT", date(2013, 10, 1), Decimal("22.00")).status == "OK"

    def test_valid_to_boundary_inclusive(self):
        limited_rule = TaxRule(
            country="IT", tax_type="VAT", rate=Decimal("0.22"),
            valid_from=date(2020, 1, 1), valid_to=date(2022, 12, 31),
            source="test", verification_status="VERIFIED",
        )
        assert validate_vat_rate("IT", date(2022, 12, 31), Decimal("22.00"),
                                 rules=[limited_rule]).status == "OK"

    def test_valid_to_boundary_exclusive_after(self):
        limited_rule = TaxRule(
            country="IT", tax_type="VAT", rate=Decimal("0.22"),
            valid_from=date(2020, 1, 1), valid_to=date(2022, 12, 31),
            source="test", verification_status="VERIFIED",
        )
        assert validate_vat_rate("IT", date(2023, 1, 1), Decimal("22.00"),
                                 rules=[limited_rule]).status == "RULE_NOT_FOUND"

    def test_mismatch_expected_rate_pct_set(self):
        result = validate_vat_rate("IT", date(2025, 1, 1), Decimal("10.00"))
        assert result.expected_rate_pct == Decimal("22.00")

    def test_ok_extracted_rate_pct_set(self):
        result = validate_vat_rate("IT", date(2025, 1, 1), Decimal("22.00"))
        assert result.extracted_rate_pct == Decimal("22.00")

    def test_rule_not_found_rule_field_is_none(self):
        assert validate_vat_rate("IT", date(2010, 1, 1), Decimal("20.00")).rule is None

    def test_ok_rule_field_populated(self):
        result = validate_vat_rate("IT", date(2025, 1, 1), Decimal("22.00"))
        assert result.rule is not None
        assert result.rule.country == "IT"


# ── No float ──────────────────────────────────────────────────────────────────

class TestNoFloat:

    def test_known_rules_rates_are_decimal(self):
        for rule in KNOWN_TAX_RULES:
            assert isinstance(rule.rate, Decimal), f"rule {rule.country} rate is not Decimal"
            assert type(rule.rate) is not float

    def test_validate_expected_rate_pct_is_decimal(self):
        result = validate_vat_rate("IT", date(2025, 1, 1), Decimal("22.00"))
        assert result.expected_rate_pct is not None
        assert isinstance(result.expected_rate_pct, Decimal)
        assert type(result.expected_rate_pct) is not float

    def test_validate_extracted_rate_pct_is_decimal(self):
        result = validate_vat_rate("IT", date(2025, 1, 1), Decimal("22.00"))
        assert isinstance(result.extracted_rate_pct, Decimal)
        assert type(result.extracted_rate_pct) is not float

    def test_rule_not_found_extracted_rate_pct_is_decimal(self):
        result = validate_vat_rate("IT", date(2010, 1, 1), Decimal("20.00"))
        assert isinstance(result.extracted_rate_pct, Decimal)
        assert type(result.extracted_rate_pct) is not float
