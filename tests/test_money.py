"""
pytest — unit tests for src/core/money.py

Coverage:
  - to_decimal: str, int, Decimal, float (boundary), invalid str, unsupported type
  - money: quantization, ROUND_HALF_UP edge cases, negatives, large values
  - diff: positive, negative, zero
  - is_discrepancy: above / below / at threshold, custom threshold
  - Return type guarantees: every function must return Decimal, never float
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.core.money import diff, is_discrepancy, money, to_decimal


# ---------------------------------------------------------------------------
# to_decimal — valid inputs
# ---------------------------------------------------------------------------

class TestToDecimalValidInputs:

    def test_string_integer(self):
        assert to_decimal("1000") == Decimal("1000")

    def test_string_decimal(self):
        assert to_decimal("22.10") == Decimal("22.10")

    def test_string_with_whitespace(self):
        assert to_decimal("  99.99  ") == Decimal("99.99")

    def test_int(self):
        assert to_decimal(5000) == Decimal("5000")

    def test_int_zero(self):
        assert to_decimal(0) == Decimal("0")

    def test_decimal_passthrough(self):
        d = Decimal("123.45")
        assert to_decimal(d) is d

    def test_decimal_negative(self):
        assert to_decimal(Decimal("-50.00")) == Decimal("-50.00")


# ---------------------------------------------------------------------------
# to_decimal — float boundary (LLM JSON output)
# ---------------------------------------------------------------------------

class TestToDecimalFloat:

    def test_float_whole_number(self):
        result = to_decimal(22.0)
        assert result == Decimal("22.0")
        assert isinstance(result, Decimal)

    def test_float_no_binary_artifact(self):
        # Decimal(22.1) produces binary artifact; to_decimal(22.1) must not
        result = to_decimal(22.1)
        assert result == Decimal("22.1")
        assert result != Decimal(22.1)   # proves str() path was taken

    def test_float_vat_italy(self):
        assert to_decimal(22.0) == Decimal("22.0")

    def test_float_vat_switzerland(self):
        assert to_decimal(8.1) == Decimal("8.1")

    def test_float_returns_decimal_not_float(self):
        result = to_decimal(100.0)
        assert type(result) is not float
        assert isinstance(result, Decimal)


# ---------------------------------------------------------------------------
# to_decimal — error cases
# ---------------------------------------------------------------------------

class TestToDecimalErrors:

    def test_invalid_string_raises_value_error(self):
        with pytest.raises(ValueError):
            to_decimal("not_a_number")

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            to_decimal("")

    def test_list_raises_type_error(self):
        with pytest.raises(TypeError):
            to_decimal([100])  # type: ignore[arg-type]

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError):
            to_decimal(None)  # type: ignore[arg-type]

    def test_dict_raises_type_error(self):
        with pytest.raises(TypeError):
            to_decimal({"amount": 100})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# money — quantization and ROUND_HALF_UP
# ---------------------------------------------------------------------------

class TestMoney:

    def test_already_two_decimals(self):
        assert money("1000.00") == Decimal("1000.00")

    def test_truncate_below_half(self):
        # 1.004 → 1.00  (4 < 5 → round down)
        assert money("1.004") == Decimal("1.00")

    def test_round_up_at_half(self):
        # 1.005 → 1.01  (ROUND_HALF_UP: 5 rounds up)
        assert money("1.005") == Decimal("1.01")

    def test_round_up_above_half(self):
        # 1.006 → 1.01
        assert money("1.006") == Decimal("1.01")

    def test_round_up_carry(self):
        # 1.995 → 2.00
        assert money("1.995") == Decimal("2.00")

    def test_negative_round_half_up(self):
        # -1.005: ROUND_HALF_UP rounds away from zero → -1.01
        assert money("-1.005") == Decimal("-1.01")

    def test_large_value(self):
        assert money("999999.999") == Decimal("1000000.00")

    def test_zero(self):
        assert money("0") == Decimal("0.00")

    def test_int_input(self):
        assert money(100) == Decimal("100.00")

    def test_float_input_vat_22(self):
        # VAT rate from LLM JSON arrives as float 22.0
        assert money(22.0) == Decimal("22.00")

    def test_return_type_is_decimal(self):
        result = money("50.50")
        assert type(result) is not float
        assert isinstance(result, Decimal)

    def test_not_round_half_even(self):
        # Python's built-in round() uses ROUND_HALF_EVEN (banker's rounding).
        # money("2.5") must be 2.5 → 2.50 (no rounding needed at cent level).
        # For 2.505 → ROUND_HALF_UP gives 2.51, ROUND_HALF_EVEN would give 2.50.
        assert money("2.505") == Decimal("2.51")
        # Verify built-in round() would give different result (documents the difference)
        assert round(2.505, 2) != 2.51   # float arithmetic, shows why we need Decimal


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

class TestDiff:

    def test_overbilling(self):
        # Billed 5100, expected 5000 → diff = +100.00
        result = diff("5000.00", "5100.00")
        assert result == Decimal("100.00")

    def test_underbilling(self):
        # Billed 4900, expected 5000 → diff = -100.00
        result = diff("5000.00", "4900.00")
        assert result == Decimal("-100.00")

    def test_exact_match(self):
        result = diff("1234.56", "1234.56")
        assert result == Decimal("0.00")

    def test_one_cent_difference(self):
        result = diff("100.00", "100.01")
        assert result == Decimal("0.01")

    def test_fractional_inputs_quantized_before_subtract(self):
        # Both inputs are quantized to cent before subtraction
        result = diff("100.004", "100.009")
        # money("100.004") = 100.00, money("100.009") = 100.01
        assert result == Decimal("0.01")

    def test_return_type_is_decimal(self):
        result = diff("10.00", "20.00")
        assert type(result) is not float
        assert isinstance(result, Decimal)

    def test_float_inputs(self):
        # Simulate LLM output: both arrive as float
        result = diff(5000.0, 5100.0)
        assert result == Decimal("100.00")


# ---------------------------------------------------------------------------
# is_discrepancy
# ---------------------------------------------------------------------------

class TestIsDiscrepancy:

    def test_above_threshold(self):
        assert is_discrepancy("100.00", "100.02") is True

    def test_below_threshold(self):
        # diff = 0.004 < 0.01 after quantization → both become same cent → False
        assert is_discrepancy("100.00", "100.004") is False

    def test_exactly_at_threshold(self):
        # diff = 0.01 == 0.01 → True (>= is inclusive)
        assert is_discrepancy("100.00", "100.01") is True

    def test_negative_discrepancy(self):
        # abs(-0.05) = 0.05 >= 0.01 → True
        assert is_discrepancy("100.05", "100.00") is True

    def test_exact_match_no_discrepancy(self):
        assert is_discrepancy("5000.00", "5000.00") is False

    def test_custom_threshold(self):
        # With threshold=0.10: diff of 0.05 should not be a discrepancy
        assert is_discrepancy("100.00", "100.05", threshold=Decimal("0.10")) is False

    def test_custom_threshold_triggered(self):
        # With threshold=0.10: diff of 0.10 should be a discrepancy
        assert is_discrepancy("100.00", "100.10", threshold=Decimal("0.10")) is True

    def test_float_inputs(self):
        # Simulate values arriving from LLM JSON as float
        assert is_discrepancy(5000.0, 5100.0) is True
        assert is_discrepancy(5000.0, 5000.0) is False

    def test_vat_mismatch(self):
        # Invoice applies 22.0%, contract mandates 8.1% → discrepancy
        assert is_discrepancy("8.1", "22.0") is True

    def test_vat_exact_match(self):
        assert is_discrepancy("22.0", "22.0") is False

    def test_return_type_is_bool(self):
        result = is_discrepancy("100.00", "100.01")
        assert isinstance(result, bool)
