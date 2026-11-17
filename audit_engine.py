"""
audit_engine.py — cross-check Invoice and Contract, optionally DDT.

Entry point: AuditEngine.cross_check(invoice, contract, ddt=None)

Checks performed:
  1. Quantity mismatch   — DDT vs Invoice  (only when DDT is provided)
  2. Unit price mismatch — Invoice vs Contract
       0 < delta ≤ price_warning_threshold  → Warning
       delta > price_warning_threshold      → Anomaly
  3. Extra invoice items — absent from Contract (and DDT when present)
  4. VAT rate            — Invoice vs RegionConfig (IT 22%, CH 8.1%)

All monetary arithmetic uses Decimal; float is never introduced.
vat_rate is expected in percentage form: Decimal("22.00") = 22%.
"""

from __future__ import annotations

import difflib
import re
from datetime import date
from decimal import Decimal
from typing import NamedTuple, Optional

from config.regions import RegionConfig, get_region
from schemas.audit import Finding, FindingType
from schemas.base import Evidence, Money
from schemas.document import Contract, DDT, Invoice, LineItem
from src.core.money import diff, is_discrepancy, money
from src.fiscal_rules import validate_vat_rate

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Price deviations up to this fraction of the contract price → Warning.
# Above this fraction → Anomaly.  2% is the default; override per session.
_DEFAULT_PRICE_WARNING_MAX_PCT = Decimal("0.02")

# Maximum amount delta (in currency units) accepted as ERP rounding — not a real discrepancy.
# Applies to Swiss Rappen (CHF 0.05 rounding) and equivalent EUR precision loss.
# Used as the global fallback; prefer _FUZZY_TOLERANCE_BY_JURISDICTION for jurisdiction-aware logic.
_FUZZY_ROUNDING_TOLERANCE = Decimal("0.05")

# Per-jurisdiction ERP rounding tolerances (amount, not percentage).
# IT: centesimal (0.01) — Italian ERPs have no sub-cent rounding.
# CH: 0.05 Rappen — Swiss standard for cash rounding.
# UNKNOWN: conservative middle ground.
_FUZZY_TOLERANCE_BY_JURISDICTION: dict[str, Decimal] = {
    "IT":      Decimal("0.01"),
    "CH":      Decimal("0.05"),
    "UNKNOWN": Decimal("0.02"),
}

# Descriptions that start with these tokens are accounting summaries, not billable services.
# Check is case-insensitive and requires the token to be a full prefix word.
_STRUCTURAL_PREFIXES: tuple[str, ...] = (
    "total", "totale", "subtotal", "sub-total", "sous-total",
    "gesamtbetrag", "zwischensumme",
    "total ht", "total ttc", "total tva", "total mwst",
    "montant total", "montant ht", "montant ttc",
    "tva", "iva", "mwst", "mehrwertsteuer",
    "netto", "brutto",
    "remise", "discount", "rabatt", "escompte", "sconto",
)

# Descriptions matching these tokens are standard accounting rounding adjustments.
# Negative amounts (client-favorable) → OK. Positive amounts within tolerance → Warning.
_ROUNDING_KEYWORDS: frozenset[str] = frozenset({
    "arrondis", "arrondi",              # FR (Svizzera, standard CH)
    "rounding", "rounding adjustment",  # EN
    "arrotondamento",                   # IT
    "rundung", "abrundung",             # DE
})

# Matches 6+ character alphanumeric tokens (uppercase) that contain at least 2 digits.
# Used for SKU / product-code detection in description fields.
_SKU_PATTERN = re.compile(r"\b([A-Z0-9]{6,})\b")


class _AmountMatchResult(NamedTuple):
    item: "LineItem"  # noqa: F821 — resolved at runtime
    is_fuzzy: bool    # True when delta is 0.01–0.05 (ERP rounding), False for exact


class _DescriptionMatch(NamedTuple):
    item: "LineItem"  # noqa: F821
    is_fuzzy: bool    # False = exact, True = token-subset or similarity match
    similarity: float  # 1.0 for exact; 0.82–<1.0 for fuzzy


# Stopwords filtered before token-subset matching (IT/EN/FR/DE).
_STOPWORDS: frozenset[str] = frozenset({
    "di", "il", "la", "lo", "le", "gli", "i", "a", "in", "con", "su", "per",
    "tra", "fra", "e", "o", "ma", "che", "non", "dei", "delle", "del", "della",
    "un", "una", "uno", "da", "al", "ai", "agli", "alle", "alla",
    "the", "an", "of", "for", "and", "or", "with", "to", "at", "by", "is",
    "de", "du", "des", "les", "et", "ou", "avec", "pour", "sur",
    "der", "die", "das", "und", "oder", "mit", "von", "für", "auf",
})


def _significant_tokens(text: str) -> frozenset[str]:
    """Return lowercase, non-stopword tokens with length > 1."""
    return frozenset(
        t for t in text.lower().split()
        if t not in _STOPWORDS and len(t) > 1
    )


# ---------------------------------------------------------------------------
# AuditEngine
# ---------------------------------------------------------------------------

class AuditEngine:
    """
    Stateless audit engine.  Instantiate once; call cross_check() for each
    (Invoice, Contract) pair — DDT is optional.

    Parameters
    ----------
    price_dubbio_threshold : Decimal
        Maximum price deviation (as a fraction of contract price) that
        produces a Warning badge.  Anything above this threshold produces
        Anomaly.  Default: 0.02 (2%).
    """

    def __init__(
        self,
        price_dubbio_threshold: Decimal = _DEFAULT_PRICE_WARNING_MAX_PCT,
    ) -> None:
        self._price_dubbio_threshold = price_dubbio_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cross_check(
        self,
        invoice: Invoice,
        contract: Contract,
        ddt: Optional[DDT] = None,
    ) -> list[Finding]:
        """
        Run all applicable cross-checks and return a flat list of Findings.

        When ddt is None, quantity checks are skipped silently and extra-item
        checks run against the contract only.
        """
        findings: list[Finding] = []
        if ddt is not None:
            findings.extend(self._check_quantities(ddt, invoice))
        findings.extend(self._check_prices(invoice, contract))
        findings.extend(self._check_extra_items(invoice, contract, ddt=ddt))
        findings.extend(self._check_vat(invoice, contract))
        return findings

    def cross_check_ddt_invoice(
        self,
        ddt: DDT,
        invoice: Invoice,
        contract: Contract,
    ) -> list[Finding]:
        """Backward-compatible wrapper — prefer cross_check()."""
        return self.cross_check(invoice, contract, ddt=ddt)

    # ------------------------------------------------------------------
    # Check 1 — Quantities: DDT vs Invoice
    # ------------------------------------------------------------------

    def _check_quantities(self, ddt: DDT, invoice: Invoice) -> list[Finding]:
        findings: list[Finding] = []

        for inv_item in invoice.line_items:
            if self._is_structural_row(inv_item.description):
                continue
            ddt_match = self._match_item(inv_item.description, ddt.line_items)
            if ddt_match is None:
                # absence is caught by _check_extra_items
                continue

            inv_qty = inv_item.quantity
            ddt_qty = ddt_match.item.quantity

            if inv_qty == ddt_qty:
                continue

            delta_qty = inv_qty - ddt_qty
            impact_amount = money(abs(delta_qty) * inv_item.unit_price.amount)
            currency = inv_item.unit_price.currency
            country = inv_item.unit_price.country

            if delta_qty > 0:
                motivation = (
                    f"Warning: {inv_qty} units of '{inv_item.description}' billed, "
                    f"but the delivery note (DDT) records {ddt_qty} units. "
                    f"Excess of {delta_qty} units ({currency} {impact_amount} overbilled)."
                )
            else:
                motivation = (
                    f"Note: {inv_qty} units of '{inv_item.description}' billed, "
                    f"but the DDT records {ddt_qty} units. "
                    f"{abs(delta_qty)} units present in the DDT do not appear on the invoice."
                )

            findings.append(
                Finding(
                    finding_type="QUANTITY",
                    status="Anomaly",
                    motivation=motivation,
                    impact_value=Money(
                        amount=impact_amount,
                        currency=currency,
                        country=country,
                    ) if delta_qty > 0 else None,
                    evidence=[
                        Evidence(
                            source_document=invoice.meta.source_file or "invoice",
                            extracted_text=(
                                f"Quantity billed: {inv_qty} — {inv_item.description}"
                            ),
                            invoice_field="line_items",
                        ),
                        Evidence(
                            source_document=ddt.meta.source_file or "DDT",
                            extracted_text=(
                                f"Quantity on DDT: {ddt_qty} — {ddt_match.item.description}"
                            ),
                        ),
                    ],
                    source_document_ids=[
                        invoice.meta.document_id,
                        ddt.meta.document_id,
                    ],
                )
            )

        return findings

    # ------------------------------------------------------------------
    # Check 2 — Unit prices: Invoice vs Contract
    # ------------------------------------------------------------------

    def _check_prices(self, invoice: Invoice, contract: Contract) -> list[Finding]:
        findings: list[Finding] = []

        for inv_item in invoice.line_items:
            if self._is_structural_row(inv_item.description):
                continue
            desc_match = self._match_item(inv_item.description, contract.line_items)
            if desc_match is None or desc_match.is_fuzzy:
                # absence and fuzzy matches are handled by _check_extra_items
                continue

            inv_price = money(inv_item.unit_price.amount)
            contract_price = money(desc_match.item.unit_price.amount)
            currency = inv_item.unit_price.currency
            country = inv_item.unit_price.country

            if not is_discrepancy(contract_price, inv_price):
                continue

            price_delta = money(inv_price - contract_price)

            # Percentage deviation vs contract price (avoid division by zero)
            if contract_price != Decimal("0.00"):
                pct_dev = abs(price_delta) / contract_price
            else:
                pct_dev = Decimal("1.00")  # anything vs zero = 100% deviation

            status = (
                "Anomaly"
                if pct_dev >= self._price_dubbio_threshold
                else "Warning"
            )
            pct_display = money(pct_dev * Decimal("100"))

            if price_delta > Decimal("0.00"):
                motivation = (
                    f"Unit price billed for '{inv_item.description}' "
                    f"is {currency} {inv_price}, "
                    f"but the contract specifies {currency} {contract_price}. "
                    f"Overcharge of {currency} {price_delta} ({pct_display}%)."
                    + (
                        " Significant deviation: verify whether an unauthorised "
                        "price change was applied."
                        if status == "Anomaly"
                        else " Minor deviation: may be a rounding difference or "
                        "a rate review. Please verify."
                    )
                )
            else:
                motivation = (
                    f"Unit price billed for '{inv_item.description}' "
                    f"is {currency} {inv_price}, "
                    f"below the contract price {currency} {contract_price} "
                    f"by {currency} {abs(price_delta)} ({pct_display}%). "
                    f"Possible undocumented discount or price list error."
                )

            # Total financial impact for this line: Δprice × quantity
            total_impact = money(abs(price_delta) * inv_item.quantity)

            findings.append(
                Finding(
                    finding_type="PRICING",
                    status=status,
                    motivation=motivation,
                    impact_value=Money(
                        amount=total_impact,
                        currency=currency,
                        country=country,
                    ),
                    evidence=[
                        Evidence(
                            source_document=invoice.meta.source_file or "invoice",
                            extracted_text=(
                                f"Unit price billed: {currency} {inv_price}"
                            ),
                            invoice_field="unit_price",
                        ),
                        Evidence(
                            source_document=contract.meta.source_file or "contract",
                            extracted_text=(
                                f"Contract price: {currency} {contract_price}"
                            ),
                            matched_clause=desc_match.item.description,
                        ),
                    ],
                    source_document_ids=[
                        invoice.meta.document_id,
                        contract.meta.document_id,
                    ],
                )
            )

        return findings

    # ------------------------------------------------------------------
    # Check 3 — Extra invoice items absent from Contract (and DDT if present)
    # ------------------------------------------------------------------

    def _check_extra_items(
        self,
        invoice: Invoice,
        contract: Contract,
        ddt: Optional[DDT] = None,
    ) -> list[Finding]:
        findings: list[Finding] = []

        tol = self._get_fuzzy_tolerance(invoice.meta.jurisdiction)

        for inv_item in invoice.line_items:
            # ── Structural row filter ─────────────────────────────────────────
            if self._is_structural_row(inv_item.description):
                continue

            # ── Description matching ──────────────────────────────────────────
            desc_match_contract = self._match_item(inv_item.description, contract.line_items)
            desc_match_ddt = (
                self._match_item(inv_item.description, ddt.line_items)
                if ddt is not None
                else None
            )

            exact_in_contract = (
                desc_match_contract is not None and not desc_match_contract.is_fuzzy
            )
            exact_in_ddt = (
                desc_match_ddt is not None and not desc_match_ddt.is_fuzzy
            )

            # Exact match in contract or DDT → _check_prices() handles price validation
            if exact_in_contract or exact_in_ddt:
                continue

            # Fuzzy description match in contract (not exact) → Warning for manual review.
            # Price context included so the reviewer has complete information.
            if desc_match_contract is not None:
                currency_f = inv_item.unit_price.currency
                item_unit_f = money(inv_item.unit_price.amount)
                contract_price_f = money(desc_match_contract.item.unit_price.amount)
                price_delta_f = money(item_unit_f - contract_price_f)

                if abs(price_delta_f) > tol:
                    price_note = (
                        f" Prezzo fatturato: {currency_f} {item_unit_f}, "
                        f"prezzo contrattuale: {currency_f} {contract_price_f} "
                        f"(differenza: {currency_f} {abs(price_delta_f)})."
                    )
                    fuzzy_impact = Money(
                        amount=money(abs(price_delta_f) * inv_item.quantity),
                        currency=currency_f,
                        country=inv_item.unit_price.country,
                    )
                else:
                    price_note = ""
                    fuzzy_impact = None

                findings.append(
                    Finding(
                        finding_type="CONTRACTUAL",
                        status="Warning",
                        motivation=(
                            f"🟡 Revisione Manuale - Descrizione Simile: la voce in fattura "
                            f"'{inv_item.description}' è simile ma non identica a "
                            f"'{desc_match_contract.item.description}' nel contratto "
                            f"(corrispondenza {desc_match_contract.similarity:.0%}).{price_note} "
                            f"Verificare se si tratta della stessa prestazione prima di approvare."
                        ),
                        impact_value=fuzzy_impact,
                        evidence=[
                            Evidence(
                                source_document=invoice.meta.source_file or "invoice",
                                extracted_text=(
                                    f"{inv_item.description} — "
                                    f"{currency_f} {item_unit_f}"
                                ),
                                invoice_field="line_items",
                                confidence=float(desc_match_contract.similarity),
                            ),
                            Evidence(
                                source_document=contract.meta.source_file or "contract",
                                extracted_text=(
                                    f"Voce simile in contratto: "
                                    f"{desc_match_contract.item.description}"
                                ),
                                matched_clause=desc_match_contract.item.description,
                                confidence=float(desc_match_contract.similarity),
                            ),
                        ],
                        source_document_ids=[
                            invoice.meta.document_id,
                            contract.meta.document_id,
                        ],
                    )
                )
                continue

            # ── Compute amounts for remaining checks ──────────────────────────
            item_total = money(inv_item.total.amount)
            item_unit  = money(inv_item.unit_price.amount)
            currency = inv_item.total.currency
            country  = inv_item.total.country

            # ── Zero-value items: included services, warranties, descriptive entries ──
            if item_unit == Decimal("0.00") or item_total == Decimal("0.00"):
                findings.append(
                    Finding(
                        finding_type="CONTRACTUAL",
                        status="OK",
                        motivation=(
                            f"Voce a valore zero (EUR 0.00): '{inv_item.description}'. "
                            "Identificata come prestazione inclusa nel canone principale, "
                            "estensione di garanzia o nota descrittiva del fornitore."
                        ),
                        evidence=[
                            Evidence(
                                source_document=invoice.meta.source_file or "invoice",
                                extracted_text=f"{inv_item.description} — EUR 0.00",
                                invoice_field="line_items",
                                confidence=1.0,
                            ),
                        ],
                        source_document_ids=[invoice.meta.document_id],
                    )
                )
                continue

            # ── SKU / product-code reconciliation with price sanity check ─────
            sku_match = self._match_item_by_sku(inv_item, contract.line_items)
            if sku_match is not None:
                inv_sku = inv_item.sku or next(iter(self._extract_skus(inv_item.description)), "")
                sku_contract_price = money(sku_match.unit_price.amount)
                price_delta_sku = money(item_unit - sku_contract_price)

                if abs(price_delta_sku) <= tol:
                    findings.append(
                        Finding(
                            finding_type="CONTRACTUAL",
                            status="OK",
                            motivation=(
                                f"🟢 Conforme (Match SKU): la voce '{inv_item.description}' "
                                f"è riconciliata con '{sku_match.description}' tramite codice "
                                f"prodotto ({inv_sku}). La corrispondenza per codice articolo "
                                f"è considerata valida indipendentemente dalla descrizione testuale."
                            ),
                            evidence=[
                                Evidence(
                                    source_document=invoice.meta.source_file or "invoice",
                                    extracted_text=(
                                        f"{inv_item.description} — "
                                        f"{currency} {item_unit} (unit) / {item_total} (totale)"
                                    ),
                                    invoice_field="line_items",
                                    confidence=0.90,
                                ),
                                Evidence(
                                    source_document=contract.meta.source_file or "contract",
                                    extracted_text=(
                                        f"Voce contrattuale: {sku_match.description} — "
                                        f"Codice: {inv_sku}"
                                    ),
                                    matched_clause=sku_match.description,
                                    confidence=0.95,
                                ),
                            ],
                            source_document_ids=[
                                invoice.meta.document_id,
                                contract.meta.document_id,
                            ],
                        )
                    )
                else:
                    # SKU matches but price deviates → PRICING Anomaly
                    findings.append(
                        Finding(
                            finding_type="PRICING",
                            status="Anomaly",
                            motivation=(
                                f"Match SKU rilevato (Codice corretto: {inv_sku}), "
                                f"ma rilevata discrepanza di prezzo non autorizzata "
                                f"(Fatturato: {currency} {item_unit}, "
                                f"Contrattato: {currency} {sku_contract_price}). "
                                f"Il codice prodotto corrisponde al contratto ma il prezzo "
                                f"applicato non è stato autorizzato. Verificare urgentemente."
                            ),
                            impact_value=Money(
                                amount=money(abs(price_delta_sku) * inv_item.quantity),
                                currency=currency,
                                country=country,
                            ),
                            evidence=[
                                Evidence(
                                    source_document=invoice.meta.source_file or "invoice",
                                    extracted_text=(
                                        f"{inv_item.description} — "
                                        f"{currency} {item_unit} (unit) / {item_total} (totale)"
                                    ),
                                    invoice_field="line_items",
                                    confidence=0.90,
                                ),
                                Evidence(
                                    source_document=contract.meta.source_file or "contract",
                                    extracted_text=(
                                        f"Voce contrattuale: {sku_match.description} — "
                                        f"Prezzo contrattuale: {currency} {sku_contract_price} — "
                                        f"Codice: {inv_sku}"
                                    ),
                                    matched_clause=sku_match.description,
                                    confidence=0.95,
                                ),
                            ],
                            source_document_ids=[
                                invoice.meta.document_id,
                                contract.meta.document_id,
                            ],
                        )
                    )
                continue

            # ── Amount-based reconciliation (flat rates / canoni fissi) ───────
            amount_match = self._match_item_by_amount(inv_item, contract.line_items, tol)
            if amount_match is not None:
                matched_price = money(amount_match.item.unit_price.amount)

                if amount_match.is_fuzzy:
                    rounding_delta = abs(money(item_unit - matched_price))
                    motivation = (
                        f"🟢 Conforme (Arrotondamento ERP): la voce '{inv_item.description}' "
                        f"non corrisponde per denominazione a '{amount_match.item.description}', "
                        f"ma l'importo fatturato ({currency} {item_unit}) è compatibile con "
                        f"il canone contrattuale ({currency} {matched_price}) entro la "
                        f"tolleranza di arrotondamento ERP ({currency} {rounding_delta} su "
                        f"{currency} 0.05 massimi). Voce conforme al contratto."
                    )
                    evidence_confidence_inv = Decimal("0.85")
                    evidence_confidence_con = Decimal("0.90")
                else:
                    motivation = (
                        f"La voce '{inv_item.description}' non corrisponde per "
                        f"denominazione a '{amount_match.item.description}', "
                        f"ma l'importo ({currency} {item_unit}) coincide "
                        f"esattamente con il canone/tariffa prevista dal contratto "
                        f"({currency} {matched_price}). "
                        f"Riconciliazione per importo: voce conforme al contratto."
                    )
                    evidence_confidence_inv = Decimal("0.85")
                    evidence_confidence_con = Decimal("0.95")

                findings.append(
                    Finding(
                        finding_type="CONTRACTUAL",
                        status="OK",
                        motivation=motivation,
                        evidence=[
                            Evidence(
                                source_document=invoice.meta.source_file or "invoice",
                                extracted_text=(
                                    f"{inv_item.description} — "
                                    f"{currency} {item_unit} (unit) / {item_total} (totale)"
                                ),
                                invoice_field="line_items",
                                confidence=float(evidence_confidence_inv),
                            ),
                            Evidence(
                                source_document=contract.meta.source_file or "contract",
                                extracted_text=(
                                    f"Prezzo contrattuale: {currency} {matched_price}"
                                ),
                                matched_clause=amount_match.item.description,
                                confidence=float(evidence_confidence_con),
                            ),
                        ],
                        source_document_ids=[
                            invoice.meta.document_id,
                            contract.meta.document_id,
                        ],
                    )
                )
                continue

            # ── Rounding adjustment detection ─────────────────────────────────
            # Standard accounting entries for cash/ERP rounding (e.g., Swiss Rappen).
            # Negative amounts (client-favorable) → OK. Positive within tolerance → Warning.
            desc_lower = inv_item.description.lower().strip()
            is_rounding_adj = any(kw in desc_lower for kw in _ROUNDING_KEYWORDS)
            if is_rounding_adj and abs(item_total) <= tol:
                if item_total <= Decimal("0.00"):
                    findings.append(
                        Finding(
                            finding_type="CONTRACTUAL",
                            status="OK",
                            motivation=(
                                f"Voce di arrotondamento a favore del cliente: "
                                f"'{inv_item.description}' ({currency} {item_total}). "
                                f"Aggiustamento tecnico standard (es. arrotondamento ai "
                                f"5 centesimi svizzeri) che riduce il totale fattura. "
                                f"Nessuna anomalia: importo entro la soglia ERP "
                                f"({currency} {tol}), a vantaggio del cliente."
                            ),
                            evidence=[
                                Evidence(
                                    source_document=invoice.meta.source_file or "invoice",
                                    extracted_text=(
                                        f"{inv_item.description} — {currency} {item_total}"
                                    ),
                                    invoice_field="line_items",
                                    confidence=1.0,
                                ),
                            ],
                            source_document_ids=[invoice.meta.document_id],
                        )
                    )
                else:
                    findings.append(
                        Finding(
                            finding_type="CONTRACTUAL",
                            status="Warning",
                            motivation=(
                                f"Voce di arrotondamento positiva: "
                                f"'{inv_item.description}' ({currency} {item_total}). "
                                f"Non coperta dal contratto, ma l'importo è entro la "
                                f"soglia di tolleranza ERP ({currency} {tol}). "
                                f"Compatibile con arrotondamento tecnico. "
                                f"Revisione manuale consigliata."
                            ),
                            impact_value=Money(
                                amount=item_total,
                                currency=currency,
                                country=country,
                            ),
                            evidence=[
                                Evidence(
                                    source_document=invoice.meta.source_file or "invoice",
                                    extracted_text=(
                                        f"{inv_item.description} — {currency} {item_total}"
                                    ),
                                    invoice_field="line_items",
                                    confidence=1.0,
                                ),
                            ],
                            source_document_ids=[invoice.meta.document_id],
                        )
                    )
                continue

            # ── No match by description, SKU, or amount → Anomaly ────────────
            if ddt is not None:
                motivation = (
                    f"La voce '{inv_item.description}' è presente in fattura "
                    f"(importo: {currency} {item_total}) "
                    f"ma è assente sia nel documento di trasporto che nel contratto. "
                    f"Voce non autorizzata: verificare urgentemente l'origine di questa spesa."
                )
            else:
                motivation = (
                    f"La voce '{inv_item.description}' è presente in fattura "
                    f"(importo: {currency} {item_total}) "
                    f"ma non è coperta dal contratto né riconciliabile per importo "
                    f"con alcun canone o tariffa contrattuale. "
                    f"Verificare urgentemente l'origine di questa spesa."
                )

            findings.append(
                Finding(
                    finding_type="CONTRACTUAL",
                    status="Anomaly",
                    motivation=motivation,
                    impact_value=Money(
                        amount=item_total,
                        currency=currency,
                        country=country,
                    ),
                    evidence=[
                        Evidence(
                            source_document=invoice.meta.source_file or "invoice",
                            extracted_text=(
                                f"{inv_item.description} — {currency} {item_total}"
                            ),
                            invoice_field="line_items",
                        ),
                    ],
                    source_document_ids=[invoice.meta.document_id],
                )
            )

        return findings

    # ------------------------------------------------------------------
    # Check 4 — VAT rate vs RegionConfig
    # ------------------------------------------------------------------

    def _check_vat(self, invoice: Invoice, contract: Contract) -> list[Finding]:
        """
        Validate invoice VAT rate against the applicable fiscal rule.

        invoice.vat_rate.value must be in PERCENTAGE form: Decimal("22.00") = 22%.
        This is consistent with validate_vat_rate() in src/fiscal_rules.py.
        """
        jurisdiction = invoice.meta.jurisdiction
        extracted_rate = invoice.vat_rate.value  # percentage form
        source_file = invoice.meta.source_file or "invoice"

        if jurisdiction == "UNKNOWN":
            return [
                Finding(
                    finding_type="FISCAL",
                    status="Warning",
                    motivation=(
                        f"Unable to verify VAT ({extracted_rate}%): "
                        f"the invoice jurisdiction was not detected. "
                        f"Manually verify that the rate is correct "
                        f"for the supplier's country."
                    ),
                    evidence=[
                        Evidence(
                            source_document=source_file,
                            extracted_text=f"VAT applied: {extracted_rate}%",
                            invoice_field="vat_rate",
                            confidence=invoice.vat_rate.confidence,
                        )
                    ],
                    source_document_ids=[invoice.meta.document_id],
                )
            ]

        issue_date = invoice.meta.issue_date or date.today()
        result = validate_vat_rate(
            country=jurisdiction,
            issue_date=issue_date,
            extracted_rate=extracted_rate,
        )

        currency = invoice.subtotal.currency

        # --- OK --------------------------------------------------------
        if result.status == "OK":
            return [
                Finding(
                    finding_type="FISCAL",
                    status="OK",
                    motivation=(
                        f"VAT applied ({extracted_rate}%) matches the standard "
                        f"{jurisdiction} rate ({result.expected_rate_pct}%). "
                        f"No fiscal anomaly detected."
                    ),
                    evidence=[
                        Evidence(
                            source_document=source_file,
                            extracted_text=f"VAT: {extracted_rate}%",
                            invoice_field="vat_rate",
                            confidence=invoice.vat_rate.confidence,
                        ),
                        Evidence(
                            source_document=(
                                result.rule.source if result.rule else "fiscal rule"
                            ),
                            extracted_text=(
                                f"Expected {jurisdiction} rate: "
                                f"{result.expected_rate_pct}%"
                            ),
                            matched_clause=(
                                result.rule.source if result.rule else None
                            ),
                        ),
                    ],
                    source_document_ids=[invoice.meta.document_id],
                )
            ]

        # --- MISMATCH --------------------------------------------------
        if result.status == "MISMATCH":
            rate_delta = money(extracted_rate - result.expected_rate_pct)
            vat_impact = self._compute_vat_impact(
                invoice, extracted_rate, result.expected_rate_pct, currency, jurisdiction
            )

            impact_sentence = (
                f" VAT amount {'overcharged' if rate_delta > 0 else 'undercharged'} "
                f"is {currency} {vat_impact.amount}."
                if vat_impact
                else ""
            )

            return [
                Finding(
                    finding_type="FISCAL",
                    status="Anomaly",
                    motivation=(
                        f"VAT applied on the invoice ({extracted_rate}%) does not match "
                        f"the standard {jurisdiction} rate "
                        f"({result.expected_rate_pct}%). "
                        f"Difference: {abs(rate_delta)} percentage points.{impact_sentence}"
                    ),
                    impact_value=vat_impact,
                    evidence=[
                        Evidence(
                            source_document=source_file,
                            extracted_text=f"VAT applied: {extracted_rate}%",
                            invoice_field="vat_rate",
                            confidence=invoice.vat_rate.confidence,
                        ),
                        Evidence(
                            source_document=(
                                result.rule.source if result.rule else "fiscal rule"
                            ),
                            extracted_text=(
                                f"Expected {jurisdiction} rate: "
                                f"{result.expected_rate_pct}%"
                            ),
                            matched_clause=(
                                result.rule.source if result.rule else None
                            ),
                        ),
                    ],
                    source_document_ids=[invoice.meta.document_id],
                )
            ]

        # --- UNVERIFIED / RULE_NOT_FOUND -------------------------------
        region = get_region(jurisdiction)
        region_hint = (
            f"For {jurisdiction}, the expected standard VAT rate is "
            f"{money(region.standard_vat_rate * Decimal('100'))}%. "
            if region
            else ""
        )

        return [
            Finding(
                finding_type="FISCAL",
                status="Warning",
                motivation=(
                    f"Unable to reliably verify the VAT applied ({extracted_rate}%) "
                    f"for {jurisdiction}: no verified fiscal rule available. "
                    f"{region_hint}"
                    f"Manual review required before approving the invoice."
                ),
                evidence=[
                    Evidence(
                        source_document=source_file,
                        extracted_text=f"VAT applied: {extracted_rate}%",
                        invoice_field="vat_rate",
                        confidence=invoice.vat_rate.confidence,
                    )
                ],
                source_document_ids=[invoice.meta.document_id],
            )
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase + collapse whitespace for deterministic description matching."""
        return " ".join(text.lower().split())

    def _match_item(
        self, description: str, items: list[LineItem]
    ) -> Optional[_DescriptionMatch]:
        """
        Match description against items with three levels of precision:

        1. Exact (after normalisation)     → is_fuzzy=False, similarity=1.0
        2. Token-subset (significant words) → is_fuzzy=True,  similarity=0.90
        3. SequenceMatcher ratio ≥ 0.82    → is_fuzzy=True,  similarity=ratio

        Returns the best match found, or None.
        Exact match causes an immediate return; fuzzy matches accumulate and the
        highest-scoring one is returned.
        """
        needle = self._normalize(description)
        needle_tokens = _significant_tokens(needle)
        best_fuzzy: Optional[_DescriptionMatch] = None

        for item in items:
            haystack = self._normalize(item.description)

            # Level 1 — exact
            if needle == haystack:
                return _DescriptionMatch(item=item, is_fuzzy=False, similarity=1.0)

            candidate_sim = 0.0

            # Level 2 — token-subset: all significant words of the shorter
            # description appear in the longer one (handles "Canone manutenzione"
            # vs "Canone mensile di manutenzione").
            haystack_tokens = _significant_tokens(haystack)
            if (needle_tokens and haystack_tokens
                    and (needle_tokens.issubset(haystack_tokens)
                         or haystack_tokens.issubset(needle_tokens))):
                candidate_sim = max(candidate_sim, 0.90)

            # Level 3 — character-level similarity
            ratio = difflib.SequenceMatcher(None, needle, haystack).ratio()
            if ratio >= 0.82:
                candidate_sim = max(candidate_sim, ratio)

            if candidate_sim > 0 and (
                best_fuzzy is None or best_fuzzy.similarity < candidate_sim
            ):
                best_fuzzy = _DescriptionMatch(
                    item=item, is_fuzzy=True, similarity=candidate_sim
                )

        return best_fuzzy

    @staticmethod
    def _is_structural_row(description: str) -> bool:
        """Return True when the line is an accounting summary row, not a billable service."""
        normalized = " ".join(description.lower().split())
        for kw in _STRUCTURAL_PREFIXES:
            if normalized == kw or normalized.startswith(kw + " "):
                return True
        return False

    @staticmethod
    def _extract_skus(text: str) -> frozenset[str]:
        """Extract potential SKU/product codes from text (uppercase, ≥6 chars, ≥2 digits)."""
        tokens = _SKU_PATTERN.findall(text.upper())
        return frozenset(t for t in tokens if sum(1 for c in t if c.isdigit()) >= 2)

    @staticmethod
    def _match_item_by_sku(
        inv_item: LineItem, contract_items: list[LineItem]
    ) -> Optional[LineItem]:
        """
        Match an invoice line to a contract line via SKU / product code.

        Priority: explicit sku field → regex extraction from description.
        Returns the first matching contract item, or None.
        """
        inv_sku_field = inv_item.sku
        if inv_sku_field:
            inv_skus: frozenset[str] = frozenset([inv_sku_field.upper().strip()])
        else:
            inv_skus = AuditEngine._extract_skus(inv_item.description)

        if not inv_skus:
            return None

        for ci in contract_items:
            ci_sku_field = ci.sku
            if ci_sku_field and ci_sku_field.upper().strip() in inv_skus:
                return ci
            if AuditEngine._extract_skus(ci.description) & inv_skus:
                return ci

        return None

    @staticmethod
    def _get_fuzzy_tolerance(jurisdiction: str) -> Decimal:
        """Return the ERP-rounding tolerance for the given jurisdiction."""
        return _FUZZY_TOLERANCE_BY_JURISDICTION.get(jurisdiction, Decimal("0.02"))

    @staticmethod
    def _match_item_by_amount(
        inv_item: LineItem,
        contract_items: list[LineItem],
        tolerance: Decimal = _FUZZY_ROUNDING_TOLERANCE,
    ) -> Optional[_AmountMatchResult]:
        """
        Amount-based reconciliation fallback for fixed fees and flat rates.

        When description and SKU matching fail, check whether the invoice line's
        unit price OR its line total is within tolerance of any contract item's
        unit price.

        Returns an _AmountMatchResult with is_fuzzy=False for exact matches
        (delta < 0.01) and is_fuzzy=True for ERP-rounding matches (0.01 ≤ delta ≤ tolerance).
        Returns None when no match is found.
        """
        inv_unit  = money(inv_item.unit_price.amount)
        inv_total = money(inv_item.total.amount)

        for ci in contract_items:
            ci_price = money(ci.unit_price.amount)
            if ci_price == Decimal("0.00"):
                continue

            unit_delta  = abs(money(inv_unit  - ci_price))
            total_delta = abs(money(inv_total - ci_price))

            # Exact match: delta < 0.01 (centesimal precision)
            if unit_delta < Decimal("0.01"):
                return _AmountMatchResult(item=ci, is_fuzzy=False)
            if total_delta < Decimal("0.01"):
                return _AmountMatchResult(item=ci, is_fuzzy=False)

            # Fuzzy match: 0.01 ≤ delta ≤ tolerance (ERP rounding)
            if unit_delta <= tolerance:
                return _AmountMatchResult(item=ci, is_fuzzy=True)
            if total_delta <= tolerance:
                return _AmountMatchResult(item=ci, is_fuzzy=True)

        return None

    @staticmethod
    def _compute_vat_impact(
        invoice: Invoice,
        actual_rate_pct: Decimal,
        expected_rate_pct: Decimal,
        currency: str,
        country: str,
    ) -> Optional[Money]:
        """
        Compute the monetary delta between actual and expected VAT on the subtotal.

        Returns None when the subtotal is unavailable.
        """
        subtotal = invoice.subtotal.amount
        if subtotal == Decimal("0.00"):
            return None

        actual_vat = money(subtotal * actual_rate_pct / Decimal("100"))
        expected_vat = money(subtotal * expected_rate_pct / Decimal("100"))
        delta = money(actual_vat - expected_vat)

        return Money(amount=abs(delta), currency=currency, country=country)
