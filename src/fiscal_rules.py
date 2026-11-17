"""
Fiscal rules registry — AI Audit Engine IT/CH.

Tax rules are versioned, explicit, configurable, and never hardcoded in
prompts or scattered across the codebase.

Rate representation
-------------------
TaxRule.rate is stored as a decimal fraction: Decimal("0.22") = 22%.

validate_vat_rate() receives *extracted_rate* in PERCENTAGE form
(e.g. Decimal("22.00")), consistent with ContractRules.vat_rate and
InvoiceData.billed_vat_rate.  The conversion factor × 100 is applied
internally and never leaks to callers.

No float is used anywhere in this module.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal, Optional

from pydantic import BaseModel

from src.core.money import is_discrepancy, money


# ---------------------------------------------------------------------------
# TaxRule — single versioned fiscal rule
# ---------------------------------------------------------------------------

class TaxRule(BaseModel):
    country: Literal["IT", "CH"]
    tax_type: Literal["VAT"]
    rate: Decimal                          # fraction: Decimal("0.22") = 22%
    valid_from: date
    valid_to: Optional[date] = None
    source: str
    verification_status: Literal["VERIFIED", "UNVERIFIED"]


# ---------------------------------------------------------------------------
# VATValidationResult — returned by validate_vat_rate()
# ---------------------------------------------------------------------------

class VATValidationResult(BaseModel):
    status: Literal["OK", "MISMATCH", "UNVERIFIED", "RULE_NOT_FOUND"]
    expected_rate_pct: Optional[Decimal] = None   # percentage form for reporting
    extracted_rate_pct: Decimal                   # percentage form as-received
    rule: Optional[TaxRule] = None


# ---------------------------------------------------------------------------
# Known rules — extend when new rules are officially verified
# ---------------------------------------------------------------------------

KNOWN_TAX_RULES: list[TaxRule] = [
    TaxRule(
        country="IT",
        tax_type="VAT",
        rate=Decimal("0.22"),
        valid_from=date(2013, 10, 1),
        source=(
            "DL 98/2011 art.2 c.2-bis — standard VAT rate raised to 22% "
            "from 01.10.2013 (previously 21%)"
        ),
        verification_status="VERIFIED",
    ),
    TaxRule(
        country="CH",
        tax_type="VAT",
        rate=Decimal("0.081"),
        valid_from=date(2024, 1, 1),
        source=(
            "LIVA art.25 — standard VAT rate 8.1% in force from 01.01.2024 "
            "(previously 7.7%)"
        ),
        verification_status="VERIFIED",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_applicable_tax_rule(
    country: str,
    issue_date: date,
    rules: list[TaxRule] | None = None,
) -> TaxRule | None:
    """
    Return the most recent applicable VAT rule for *country* on *issue_date*.

    Returns None when no rule covers the requested country and date.
    Pass *rules* to override KNOWN_TAX_RULES (useful in tests).
    """
    _rules = rules if rules is not None else KNOWN_TAX_RULES
    candidates = [
        r for r in _rules
        if r.country == country
        and r.tax_type == "VAT"
        and r.valid_from <= issue_date
        and (r.valid_to is None or issue_date <= r.valid_to)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.valid_from)


def is_tax_rule_verified(rule: TaxRule) -> bool:
    """Return True only when the rule carries VERIFIED status."""
    return rule.verification_status == "VERIFIED"


def validate_vat_rate(
    country: str,
    issue_date: date,
    extracted_rate: Decimal,
    rules: list[TaxRule] | None = None,
) -> VATValidationResult:
    """
    Validate *extracted_rate* (percentage form, e.g. Decimal("22.00")) against
    the applicable fiscal rule for *country* on *issue_date*.

    Status semantics:
      OK             — rule found, verified, rate matches within 0.01 pct points
      MISMATCH       — rule found, verified, rate differs by >= 0.01 pct points
      UNVERIFIED     — rule found but UNVERIFIED; callers must cap badge at Warning
      RULE_NOT_FOUND — no applicable rule; callers must cap badge at Warning

    No float is introduced or returned by this function.
    """
    extracted_pct = money(extracted_rate)
    rule = get_applicable_tax_rule(country, issue_date, rules)

    if rule is None:
        return VATValidationResult(
            status="RULE_NOT_FOUND",
            extracted_rate_pct=extracted_pct,
        )

    expected_pct = money(rule.rate * Decimal("100"))

    if not is_tax_rule_verified(rule):
        return VATValidationResult(
            status="UNVERIFIED",
            expected_rate_pct=expected_pct,
            extracted_rate_pct=extracted_pct,
            rule=rule,
        )

    if is_discrepancy(expected_pct, extracted_pct):
        return VATValidationResult(
            status="MISMATCH",
            expected_rate_pct=expected_pct,
            extracted_rate_pct=extracted_pct,
            rule=rule,
        )

    return VATValidationResult(
        status="OK",
        expected_rate_pct=expected_pct,
        extracted_rate_pct=extracted_pct,
        rule=rule,
    )
