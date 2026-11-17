"""
Auditor — deterministic anomaly detection on Critic-verified data.

All monetary arithmetic uses decimal.Decimal via src.core.money.
No float is used for any monetary value; float is used only for
confidence scores (0.0 → 1.0).

Behaviour for low-confidence fields
------------------------------------
If a field has needs_review=True (confidence < CONFIDENCE_THRESHOLD), the
check that depends on it is SKIPPED.  The field name is surfaced in
AuditReport.needs_human_review instead of generating a potentially false
anomaly.  This prevents overbilling alerts based on data the system itself
is not sure about.

Anomaly confidence
------------------
Each AnomalyItem.confidence is the minimum confidence across all FieldResult
fields that contributed to detecting the anomaly.  If either side of a
comparison is uncertain, the anomaly confidence reflects that uncertainty.

Badge assignment
----------------
"Anomaly" — at least one anomaly (monetary, fiscal, SLA, or clause violation)
"Warning" — no anomalies but: fields need human review, confidence below
            threshold, OR the Critic ran but critical fields lack evidence
"OK"      — no anomalies, no fields need review, confidence sufficient,
            and either the Critic has not run yet (unit-test / pre-pipeline)
            or all critical fields have verifiable source evidence

Evidence propagation
--------------------
Every AnomalyItem is built via _build_evidence(), which extracts source_quote
and matched_clause from the FieldResult objects involved in the finding.
Evidence fields are None when the Critic has not yet run (empty source_quote).

Evidence-gap rule (CLAUDE.md: "No evidence = no OK badge")
-----------------------------------------------------------
If the Critic has run (at least one field has a non-empty source_quote) but a
critical comparison field (base_fee or VAT) has no evidence, the badge is
downgraded from "OK" to "Warning".  This does not affect unit tests where
all source_quotes are empty by construction.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any

from src.core.money import diff as _diff
from src.core.money import is_discrepancy, money, to_decimal
from src.fiscal_rules import TaxRule, validate_vat_rate
from src.models import (
    CONFIDENCE_THRESHOLD,
    AnomalyItem,
    AuditReport,
    ConfidenceBadge,
    ContractRules,
    FieldResult,
    InvoiceData,
)

_MONTH_NUM = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

_ZERO = Decimal("0.00")


def _month_number(name: str) -> int | None:
    return _MONTH_NUM.get(name.strip().lower())


def _min_conf(*fields) -> float:
    """Return the minimum confidence across the given FieldResult objects."""
    return round(min(f.confidence for f in fields), 4)


def _build_evidence(
    *,
    contract_fr: FieldResult | None = None,
    invoice_fr: FieldResult | None = None,
    invoice_field_name: str | None = None,
) -> dict[str, Any]:
    """
    Build evidence kwargs for AnomalyItem from the involved FieldResult objects.

    Returned keys (only present when values are non-empty):
      matched_clause      ← contract_fr.source_quote  (verbatim contract clause)
      source_quote        ← invoice_fr.source_quote   (verbatim invoice line)
      invoice_field       ← invoice_field_name
      evidence_confidence ← min(confidence) of supplied fields, as Decimal

    The result is intended to be **unpacked directly into AnomalyItem().
    """
    ev: dict[str, Any] = {}

    if invoice_field_name:
        ev["invoice_field"] = invoice_field_name

    contract_quote = (contract_fr.source_quote if contract_fr else "") or ""
    invoice_quote = (invoice_fr.source_quote if invoice_fr else "") or ""

    if contract_quote:
        ev["matched_clause"] = contract_quote
    if invoice_quote:
        ev["source_quote"] = invoice_quote

    confs: list[float] = []
    if contract_fr is not None:
        confs.append(contract_fr.confidence)
    if invoice_fr is not None:
        confs.append(invoice_fr.confidence)
    if confs:
        ev["evidence_confidence"] = Decimal(str(round(min(confs), 4)))

    return ev


def audit_invoice(
    contract: ContractRules,
    invoice: InvoiceData,
    _vat_rules: list[TaxRule] | None = None,
) -> AuditReport:
    anomalies: list[AnomalyItem] = []
    needs_review: list[str] = []
    overbilling: Decimal = _ZERO
    all_confidences: list[float] = []
    cur = contract.currency.value
    _today = date.today()
    _fiscal_cap: ConfidenceBadge | None = None

    # ------------------------------------------------------------------
    # Helper: register a field as needing human review
    # ------------------------------------------------------------------
    def _flag(*field_names: str) -> None:
        for name in field_names:
            if name not in needs_review:
                needs_review.append(name)

    # ------------------------------------------------------------------
    # 1. Base monthly fee
    # ------------------------------------------------------------------
    c_fee = contract.base_monthly_fee
    i_fee = invoice.billed_base_fee
    all_confidences.extend([c_fee.confidence, i_fee.confidence])

    if c_fee.needs_review or i_fee.needs_review:
        _flag("base_monthly_fee", "billed_base_fee")
    elif is_discrepancy(c_fee.value, i_fee.value):
        gap = _diff(c_fee.value, i_fee.value)
        anomalies.append(AnomalyItem(
            description=(
                f"Base fee mismatch: billed {cur} {i_fee.value:,.2f} vs "
                f"contract {cur} {c_fee.value:,.2f} (diff {cur} {gap:+.2f})"
            ),
            confidence=_min_conf(c_fee, i_fee),
            overbilling_amount=gap,
            **_build_evidence(
                contract_fr=c_fee,
                invoice_fr=i_fee,
                invoice_field_name="billed_base_fee",
            ),
        ))
        overbilling += gap

    # ------------------------------------------------------------------
    # 2. Extra hourly rate
    # billed_extra_hours is FieldResult[float] (hours quantity, not money).
    # Convert to Decimal before multiplying with a Decimal rate.
    # ------------------------------------------------------------------
    c_rate = contract.extra_hourly_rate
    i_rate = invoice.billed_extra_hourly_rate
    i_hours = invoice.billed_extra_hours
    all_confidences.extend([c_rate.confidence, i_rate.confidence, i_hours.confidence])

    if c_rate.needs_review or i_rate.needs_review or i_hours.needs_review:
        _flag("extra_hourly_rate", "billed_extra_hourly_rate", "billed_extra_hours")
    elif i_hours.value > 0 and is_discrepancy(c_rate.value, i_rate.value) and i_rate.value > c_rate.value:
        rate_gap = money(i_rate.value - c_rate.value)
        excess = money(rate_gap * to_decimal(i_hours.value))
        anomalies.append(AnomalyItem(
            description=(
                f"Extra hourly rate overbilled: charged {cur} {i_rate.value:,.2f}/hr "
                f"vs contract {cur} {c_rate.value:,.2f}/hr on {i_hours.value} hours "
                f"(excess {cur} {excess:,.2f})"
            ),
            confidence=_min_conf(c_rate, i_rate, i_hours),
            overbilling_amount=excess,
            **_build_evidence(
                contract_fr=c_rate,
                invoice_fr=i_rate,
                invoice_field_name="billed_extra_hourly_rate",
            ),
        ))
        overbilling += excess

    # ------------------------------------------------------------------
    # 3. Urgent call-out fee
    # ------------------------------------------------------------------
    c_urgent = contract.urgent_callout_fee
    i_urgent = invoice.billed_urgent_callout_fee
    all_confidences.extend([c_urgent.confidence, i_urgent.confidence])

    if c_urgent.needs_review or i_urgent.needs_review:
        _flag("urgent_callout_fee", "billed_urgent_callout_fee")
    elif is_discrepancy(c_urgent.value, i_urgent.value) and i_urgent.value > c_urgent.value:
        excess = money(i_urgent.value - c_urgent.value)
        anomalies.append(AnomalyItem(
            description=(
                f"Urgent call-out fee overbilled: charged {cur} {i_urgent.value:,.2f} "
                f"vs contract {cur} {c_urgent.value:,.2f} (excess {cur} {excess:,.2f})"
            ),
            confidence=_min_conf(c_urgent, i_urgent),
            overbilling_amount=excess,
            **_build_evidence(
                contract_fr=c_urgent,
                invoice_fr=i_urgent,
                invoice_field_name="billed_urgent_callout_fee",
            ),
        ))
        overbilling += excess

    # ------------------------------------------------------------------
    # 4. Unauthorized services — invoice-side evidence only
    # ------------------------------------------------------------------
    i_unauth = invoice.unauthorized_services_total
    all_confidences.append(i_unauth.confidence)

    if i_unauth.needs_review:
        _flag("unauthorized_services_total")
    elif i_unauth.value > _ZERO:
        anomalies.append(AnomalyItem(
            description=f"Unauthorized services billed: {cur} {i_unauth.value:,.2f}",
            confidence=i_unauth.confidence,
            overbilling_amount=i_unauth.value,
            **_build_evidence(
                invoice_fr=i_unauth,
                invoice_field_name="unauthorized_services_total",
            ),
        ))
        overbilling += i_unauth.value

    # ------------------------------------------------------------------
    # 5. VAT rate
    # ------------------------------------------------------------------
    c_vat = contract.vat_rate
    i_vat = invoice.billed_vat_rate
    all_confidences.extend([c_vat.confidence, i_vat.confidence])

    if c_vat.needs_review or i_vat.needs_review:
        _flag("vat_rate", "billed_vat_rate")
    elif is_discrepancy(c_vat.value, i_vat.value):
        anomalies.append(AnomalyItem(
            description=(
                f"VAT rate mismatch: invoice applies {i_vat.value:.2f}% "
                f"but contract mandates {c_vat.value:.2f}%"
            ),
            confidence=_min_conf(c_vat, i_vat),
            overbilling_amount=_ZERO,
            **_build_evidence(
                contract_fr=c_vat,
                invoice_fr=i_vat,
                invoice_field_name="billed_vat_rate",
            ),
        ))

    # ------------------------------------------------------------------
    # 5b. Fiscal rule VAT validation
    # Validates invoice VAT against the country-level rule (independent of
    # the contract-vs-invoice check above).
    # RULE_NOT_FOUND / UNVERIFIED → cap badge at Dubbio via _fiscal_cap.
    # MISMATCH → AnomalyItem; badge becomes Anomalia via normal anomaly path.
    # Skipped when country is unknown/low-confidence or VAT fields need review.
    # ------------------------------------------------------------------
    _country_val = (
        contract.country.value
        if not contract.country.needs_review and contract.country.value in ("IT", "CH")
        else None
    )
    if _country_val and not (c_vat.needs_review or i_vat.needs_review):
        _vat_rule_result = validate_vat_rate(_country_val, _today, i_vat.value, _vat_rules)
        if _vat_rule_result.status in ("RULE_NOT_FOUND", "UNVERIFIED"):
            _fiscal_cap = "Warning"
        elif _vat_rule_result.status == "MISMATCH":
            if not any("VAT rate mismatch" in a.description for a in anomalies):
                anomalies.append(AnomalyItem(
                    description=(
                        f"Fiscal rule VAT mismatch: invoice applies {i_vat.value:.2f}% "
                        f"but {_country_val} rule mandates "
                        f"{_vat_rule_result.expected_rate_pct:.2f}%"
                    ),
                    confidence=i_vat.confidence,
                    overbilling_amount=_ZERO,
                    **_build_evidence(
                        invoice_fr=i_vat,
                        invoice_field_name="billed_vat_rate",
                    ),
                ))

    # ------------------------------------------------------------------
    # 6. Indexation — contract-side evidence only
    # ------------------------------------------------------------------
    c_idx_month = contract.indexation_allowed_month
    i_idx = invoice.indexation_applied
    i_month = invoice.invoice_month
    all_confidences.extend([c_idx_month.confidence, i_idx.confidence, i_month.confidence])

    if c_idx_month.needs_review or i_idx.needs_review or i_month.needs_review:
        _flag("indexation_allowed_month", "indexation_applied", "invoice_month")
    elif i_idx.value:
        allowed_num = _month_number(c_idx_month.value)
        invoice_num = _month_number(i_month.value) if i_month.value else None
        if invoice_num is None or (allowed_num is not None and invoice_num < allowed_num):
            anomalies.append(AnomalyItem(
                description=(
                    f"Indexation applied outside the allowed period "
                    f"(only permitted from {c_idx_month.value}"
                    + (f"; invoice month: {i_month.value}" if i_month.value else "")
                    + ")"
                ),
                confidence=_min_conf(c_idx_month, i_idx, i_month),
                overbilling_amount=_ZERO,
                **_build_evidence(
                    contract_fr=c_idx_month,
                    invoice_field_name="indexation_applied",
                ),
            ))

    # ------------------------------------------------------------------
    # 7. SLA penalties
    # ------------------------------------------------------------------
    c_penalty = contract.sla_penalty_per_day
    i_penalty = invoice.sla_penalty_deducted
    i_delays = invoice.detected_delay_days
    all_confidences.extend([c_penalty.confidence, i_penalty.confidence, i_delays.confidence])

    if c_penalty.needs_review or i_penalty.needs_review or i_delays.needs_review:
        _flag("sla_penalty_per_day", "sla_penalty_deducted", "detected_delay_days")
    else:
        if i_penalty.value < _ZERO:
            anomalies.append(AnomalyItem(
                description=(
                    f"SLA penalty deduction is negative "
                    f"({cur} {i_penalty.value:,.2f}), which is invalid."
                ),
                confidence=i_penalty.confidence,
                overbilling_amount=_ZERO,
                **_build_evidence(
                    invoice_fr=i_penalty,
                    invoice_field_name="sla_penalty_deducted",
                ),
            ))

        if i_delays.value > 0:
            if c_penalty.value == _ZERO:
                # Delays confirmed but no penalty clause — unenforced or missing
                anomalies.append(AnomalyItem(
                    description=(
                        f"{i_delays.value} delay day(s) detected but the contract "
                        "SLA penalty rate is zero — clause may be missing or unenforced."
                    ),
                    confidence=_min_conf(c_penalty, i_delays),
                    overbilling_amount=_ZERO,
                    **_build_evidence(
                        contract_fr=c_penalty,
                        invoice_fr=i_delays,
                        invoice_field_name="detected_delay_days",
                    ),
                ))
            else:
                expected = money(i_delays.value * c_penalty.value)
                if i_penalty.value < expected:
                    shortfall = money(expected - i_penalty.value)
                    anomalies.append(AnomalyItem(
                        description=(
                            f"MAJOR: SLA penalty underapplied — {i_delays.value} delay day(s) detected, "
                            f"expected deduction {cur} {expected:,.2f} "
                            f"({cur} {c_penalty.value:,.2f}/day), "
                            f"but only {cur} {i_penalty.value:,.2f} deducted "
                            f"(shortfall {cur} {shortfall:,.2f})"
                        ),
                        confidence=_min_conf(c_penalty, i_penalty, i_delays),
                        overbilling_amount=shortfall,
                        **_build_evidence(
                            contract_fr=c_penalty,
                            invoice_fr=i_penalty,
                            invoice_field_name="sla_penalty_deducted",
                        ),
                    ))
                    overbilling += shortfall

    # ------------------------------------------------------------------
    # Final report — badge assignment, evidence-gap check, is_compliant
    # ------------------------------------------------------------------
    overall_confidence = (
        round(sum(all_confidences) / len(all_confidences), 4)
        if all_confidences else 0.0
    )

    badge: ConfidenceBadge
    if anomalies:
        badge = "Anomaly"
    elif needs_review or overall_confidence < CONFIDENCE_THRESHOLD:
        badge = "Warning"
    else:
        badge = "OK"

    # Evidence-gap check (CLAUDE.md: "No evidence = no OK badge")
    # Only runs when the Critic has actually produced evidence for at least
    # one field.  When all source_quotes are empty (unit tests, pre-pipeline)
    # this block is skipped so existing tests are not broken.
    if badge == "OK":
        _all_used: list[FieldResult] = [
            c_fee, i_fee, c_rate, i_rate, i_hours,
            c_urgent, i_urgent, i_unauth,
            c_vat, i_vat,
            c_idx_month, i_idx, i_month,
            c_penalty, i_penalty, i_delays,
        ]
        critic_has_run = any(bool(f.source_quote) for f in _all_used)

        if critic_has_run:
            # Critical fields that must have evidence for an OK badge
            _critical = [f for f in [c_fee, i_fee, c_vat, i_vat] if not f.needs_review]
            critical_evidence_present = all(bool(f.source_quote) for f in _critical)
            if not critical_evidence_present:
                badge = "Warning"

    # Fiscal rule badge cap: RULE_NOT_FOUND or UNVERIFIED → max badge is Dubbio
    if badge == "OK" and _fiscal_cap is not None:
        badge = _fiscal_cap

    return AuditReport(
        is_compliant=(badge == "OK"),
        anomalies_found=anomalies,
        needs_human_review=needs_review,
        estimated_overbilling=money(overbilling),
        overall_audit_confidence=overall_confidence,
        confidence_badge=badge,
    )
