"""
config/regions.py — swappable regional fiscal configuration.

Design rules:
- Each region is a self-contained RegionConfig instance.
- Adding a new region (DE, FR, EU...) requires only a new RegionConfig entry here.
- No fiscal rate is hardcoded in agents, extractors, or prompts.
- All rates are stored as Decimal fractions (0.22 = 22%). Percentage conversion
  happens at the display or validation layer, never here.
- reduced_vat_rates lists all known reduced rates for that jurisdiction, from
  the highest to zero. Extend per official source only.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal, Optional

from pydantic import BaseModel

from src.fiscal_rules import KNOWN_TAX_RULES, TaxRule


class RegionConfig(BaseModel):
    """Complete fiscal and currency configuration for one jurisdiction."""

    country: Literal["IT", "CH"]
    currency: Literal["EUR", "CHF"]
    standard_vat_rate: Decimal
    reduced_vat_rates: list[Decimal] = []
    fiscal_rules: list[TaxRule] = []
    notes: Optional[str] = None


ITALY = RegionConfig(
    country="IT",
    currency="EUR",
    standard_vat_rate=Decimal("0.22"),
    reduced_vat_rates=[
        Decimal("0.10"),
        Decimal("0.05"),
        Decimal("0.04"),
        Decimal("0.00"),
    ],
    fiscal_rules=[r for r in KNOWN_TAX_RULES if r.country == "IT"],
    notes=(
        "Standard VAT rate 22% (DL 98/2011). "
        "Reduced rates: 10% (food, construction works, etc.), "
        "5% (social/healthcare services), 4% (essential goods), 0% (specific exemptions)."
    ),
)

SWITZERLAND = RegionConfig(
    country="CH",
    currency="CHF",
    standard_vat_rate=Decimal("0.081"),
    reduced_vat_rates=[
        Decimal("0.026"),
        Decimal("0.038"),
        Decimal("0.000"),
    ],
    fiscal_rules=[r for r in KNOWN_TAX_RULES if r.country == "CH"],
    notes=(
        "Standard VAT rate 8.1% (LIVA art.25, in force from 01.01.2024). "
        "Reduced rate: 2.6% (essential goods). "
        "Special rate: 3.8% (accommodation)."
    ),
)

REGION_REGISTRY: dict[str, RegionConfig] = {
    "IT": ITALY,
    "CH": SWITZERLAND,
}


def get_region(country: str) -> Optional[RegionConfig]:
    """Return the RegionConfig for *country*, or None if not registered."""
    return REGION_REGISTRY.get(country)
