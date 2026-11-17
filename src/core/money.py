"""
Centralised monetary arithmetic for AI Audit Engine IT/CH.

Rules enforced here:
- All monetary values are decimal.Decimal, never float.
- Rounding is always ROUND_HALF_UP to 2 decimal places.
- float inputs are accepted ONLY at system boundaries (LLM JSON output) and
  converted via str() to avoid binary floating-point artifacts.
- Every function returns Decimal. No float ever leaves this module.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Union

# ---------------------------------------------------------------------------
# Internal constants — never expose as float
# ---------------------------------------------------------------------------

_CENT = Decimal("0.01")
_ZERO = Decimal("0.00")

# Public alias so callers can write money.ZERO without importing Decimal
ZERO: Decimal = _ZERO

# Accepted input types
MonetaryInput = Union[str, int, float, Decimal]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def to_decimal(value: MonetaryInput) -> Decimal:
    """
    Convert *value* to Decimal safely.

    - str / int  : converted via Decimal(str(value)).
    - Decimal    : returned as-is.
    - float      : converted via str() to avoid binary precision loss.
                   This path exists solely for values arriving from LLM JSON
                   output. Internal code must never pass float here.
    - Any other type raises TypeError.
    - Non-numeric strings raise ValueError.
    """
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, int):
        return Decimal(str(value))
    if isinstance(value, str):
        try:
            return Decimal(value.strip())
        except InvalidOperation as exc:
            raise ValueError(
                f"Cannot convert string {value!r} to Decimal: {exc}"
            ) from exc
    raise TypeError(
        f"Monetary values must be str, int, float, or Decimal — "
        f"got {type(value).__name__!r}"
    )


def money(value: MonetaryInput) -> Decimal:
    """
    Return *value* quantized to 2 decimal places with ROUND_HALF_UP.

    This is the canonical representation for any stored or compared monetary
    amount. Always call money() before storing a result or comparing two values.
    """
    return to_decimal(value).quantize(_CENT, rounding=ROUND_HALF_UP)


def diff(expected: MonetaryInput, actual: MonetaryInput) -> Decimal:
    """
    Return (actual − expected), both operands quantized before subtraction.

    A positive result means actual > expected (overbilling).
    A negative result means actual < expected (underbilling / credit).
    """
    return money(actual) - money(expected)


def is_discrepancy(
    expected: MonetaryInput,
    actual: MonetaryInput,
    threshold: Decimal = Decimal("0.01"),
) -> bool:
    """
    Return True when abs(actual − expected) >= *threshold*.

    Default threshold is Decimal("0.01") — one euro/franc cent — as required
    by the fiscal calculation rules in CLAUDE.md.
    """
    return abs(diff(expected, actual)) >= threshold
