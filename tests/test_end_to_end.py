"""
End-to-end tests — AI Audit Engine IT/CH.

These tests call run_audit_pipeline() on real synthetic PDFs and the live
Anthropic API.  They are intentionally separated from the deterministic
unit/integration suite.

Activation
----------
Two conditions must both be true for these tests to run:
  1. Environment variable RUN_E2E=1
  2. Environment variable ANTHROPIC_API_KEY set to a valid key

Run only e2e:    pytest tests/ -m e2e -v
Run only units:  pytest tests/ -m "not e2e" -v

Gap policy
----------
If a test fails because the synthetic PDF does not contain sufficient evidence
or because LLM output is non-deterministic, the failure is documented in the
test's docstring under "GAP:" rather than patched in core logic.
The pipeline's deterministic layers (Auditor, fiscal rules, money) are fully
covered by the unit suite; e2e tests verify the integration boundary only.
"""

from __future__ import annotations

import os
import pathlib
from decimal import Decimal

import pytest

# ---------------------------------------------------------------------------
# Gate — skip entire module unless explicitly enabled
# ---------------------------------------------------------------------------

_SKIP_REASON: str | None = None
if os.environ.get("RUN_E2E") != "1":
    _SKIP_REASON = "RUN_E2E != '1'  →  set RUN_E2E=1 to enable end-to-end tests"
elif not os.environ.get("ANTHROPIC_API_KEY"):
    _SKIP_REASON = "ANTHROPIC_API_KEY not set"

if _SKIP_REASON:
    pytest.skip(_SKIP_REASON, allow_module_level=True)

# ---------------------------------------------------------------------------
# Late imports — only reached when the gate passes
# ---------------------------------------------------------------------------

# NOTE: src.core.orchestrator and run_audit_pipeline were removed.
# These e2e tests need to be rewritten around UniversalExtractor + AuditEngine.
# pdfplumber has been removed; documents are read as raw bytes and sent to Vision API.
from src.core.orchestrator import FullAuditResult, run_audit_pipeline  # noqa: E402  # TODO: replace

pytestmark = pytest.mark.e2e

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATI_TEST = pathlib.Path(__file__).parent.parent / "Dati Test"
_CONTRACT_PDF = _DATI_TEST / "contract_sample.pdf"
_M1_PDF = _DATI_TEST / "invoice_m1_correct.pdf"
_M2_PDF = _DATI_TEST / "invoice_m2_errors.pdf"
_M3_PDF = _DATI_TEST / "invoice_m3_sla.pdf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_pdf(path: pathlib.Path) -> bytes:
    # TODO: update fixtures to pass bytes to UniversalExtractor.extract()
    return path.read_bytes()


# ---------------------------------------------------------------------------
# Session-scoped fixtures — one API call per invoice, shared across tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def contract_text() -> str:
    return _read_pdf(_CONTRACT_PDF)


@pytest.fixture(scope="session")
def result_m1(contract_text: str) -> FullAuditResult:
    """Pipeline result for the correct invoice (M1)."""
    return run_audit_pipeline(contract_text, _read_pdf(_M1_PDF))


@pytest.fixture(scope="session")
def result_m2(contract_text: str) -> FullAuditResult:
    """Pipeline result for the invoice with deliberate errors (M2)."""
    return run_audit_pipeline(contract_text, _read_pdf(_M2_PDF))


@pytest.fixture(scope="session")
def result_m3(contract_text: str) -> FullAuditResult:
    """Pipeline result for the invoice with SLA issues (M3)."""
    return run_audit_pipeline(contract_text, _read_pdf(_M3_PDF))


# ---------------------------------------------------------------------------
# TestPipelineSmoke — M1 (correct invoice)
# ---------------------------------------------------------------------------

class TestPipelineSmoke:
    """
    M1 — invoice_m1_correct.pdf

    Expected: pipeline completes without error; no Anomaly badge.
    The invoice matches the contract, so the engine should not flag anomalies.
    """

    def test_m1_pipeline_does_not_crash(self, result_m1: FullAuditResult) -> None:
        assert result_m1 is not None

    def test_m1_status_not_error(self, result_m1: FullAuditResult) -> None:
        assert result_m1.status != "error", (
            f"Pipeline returned error: {result_m1.error_message}"
        )

    def test_m1_status_is_valid(self, result_m1: FullAuditResult) -> None:
        assert result_m1.status in ("success", "needs_human_review")

    def test_m1_has_audit_report(self, result_m1: FullAuditResult) -> None:
        assert result_m1.audit_report is not None

    def test_m1_has_contract_and_invoice(self, result_m1: FullAuditResult) -> None:
        assert result_m1.contract is not None
        assert result_m1.invoice is not None

    def test_m1_badge_not_anomaly(self, result_m1: FullAuditResult) -> None:
        """
        GAP: LLM extraction may produce low-confidence fields or misread values
        from the synthetic PDF, causing the Auditor to flag false anomalies.
        If this test fails, the root cause is extraction quality, not Auditor logic.
        """
        if result_m1.audit_report is None:
            pytest.skip("No audit report — pipeline error upstream")
        badge = result_m1.audit_report.confidence_badge
        assert badge != "Anomaly", (
            f"M1 (correct invoice) produced Anomaly badge. "
            f"Anomalies: {[a.description for a in result_m1.audit_report.anomalies_found]}"
        )

    def test_m1_jurisdiction_detected(self, result_m1: FullAuditResult) -> None:
        """Router should classify the contract as IT or CH."""
        if result_m1.contract_context is None:
            pytest.skip("No contract context — pipeline error upstream")
        assert result_m1.contract_context.jurisdiction in ("IT", "CH"), (
            f"Unexpected jurisdiction: {result_m1.contract_context.jurisdiction}"
        )


# ---------------------------------------------------------------------------
# TestPipelineM2 — M2 (invoice with errors)
# ---------------------------------------------------------------------------

class TestPipelineM2:
    """
    M2 — invoice_m2_errors.pdf

    Expected: pipeline completes; at least one anomaly detected; badge Anomaly.
    The invoice contains deliberate billing errors vs the contract.

    GAP (known): if the synthetic PDF does not produce clearly mismatched numeric
    fields after LLM extraction, the Auditor may not flag any anomaly.
    This is an extraction quality gap, not an Auditor logic gap.
    """

    def test_m2_pipeline_does_not_crash(self, result_m2: FullAuditResult) -> None:
        assert result_m2 is not None

    def test_m2_status_not_error(self, result_m2: FullAuditResult) -> None:
        assert result_m2.status != "error", (
            f"Pipeline returned error: {result_m2.error_message}"
        )

    def test_m2_has_audit_report(self, result_m2: FullAuditResult) -> None:
        assert result_m2.audit_report is not None

    def test_m2_has_anomalies(self, result_m2: FullAuditResult) -> None:
        """
        GAP: extraction from synthetic PDF may not produce numeric mismatches
        large enough to trigger the Auditor (threshold >= 0.01 EUR).
        If this fails, check extractor output for base_fee / vat_rate fields.
        """
        if result_m2.audit_report is None:
            pytest.skip("No audit report — pipeline error upstream")
        assert len(result_m2.audit_report.anomalies_found) >= 1, (
            "M2 (invoice with errors) produced zero anomalies — "
            "possible extraction gap in synthetic PDF"
        )

    def test_m2_badge_is_anomaly(self, result_m2: FullAuditResult) -> None:
        """
        GAP: same as test_m2_has_anomalies — if extraction fails to surface
        the billing errors, badge will be Warning instead of Anomaly.
        """
        if result_m2.audit_report is None:
            pytest.skip("No audit report — pipeline error upstream")
        assert result_m2.audit_report.confidence_badge == "Anomaly", (
            f"Expected Anomaly, got {result_m2.audit_report.confidence_badge}. "
            f"Anomalies found: {[a.description for a in result_m2.audit_report.anomalies_found]}"
        )

    def test_m2_overbilling_is_decimal(self, result_m2: FullAuditResult) -> None:
        if result_m2.audit_report is None:
            pytest.skip("No audit report")
        assert isinstance(result_m2.audit_report.estimated_overbilling, Decimal)

    def test_m2_anomaly_descriptions_not_empty(self, result_m2: FullAuditResult) -> None:
        if result_m2.audit_report is None:
            pytest.skip("No audit report")
        for anomaly in result_m2.audit_report.anomalies_found:
            assert anomaly.description.strip(), "Anomaly has empty description"


# ---------------------------------------------------------------------------
# TestPipelineM3 — M3 (SLA issues)
# ---------------------------------------------------------------------------

class TestPipelineM3:
    """
    M3 — invoice_m3_sla.pdf

    Expected: pipeline completes; badge is Anomaly or Warning (never OK).
    The invoice contains SLA penalty issues.

    GAP: SLA delay detection depends on the LLM correctly identifying delay-day
    language in the synthetic PDF. If the PDF is ambiguous, the Auditor may
    not flag the SLA check and badge could be OK (false negative).
    """

    def test_m3_pipeline_does_not_crash(self, result_m3: FullAuditResult) -> None:
        assert result_m3 is not None

    def test_m3_status_not_error(self, result_m3: FullAuditResult) -> None:
        assert result_m3.status != "error", (
            f"Pipeline returned error: {result_m3.error_message}"
        )

    def test_m3_has_audit_report(self, result_m3: FullAuditResult) -> None:
        assert result_m3.audit_report is not None

    def test_m3_badge_not_ok(self, result_m3: FullAuditResult) -> None:
        """
        GAP: if delay-day evidence is not extracted from the synthetic PDF,
        the Auditor cannot flag SLA underapplication → badge may be OK.
        In that case the gap is in extraction quality, not Auditor logic.
        """
        if result_m3.audit_report is None:
            pytest.skip("No audit report — pipeline error upstream")
        badge = result_m3.audit_report.confidence_badge
        assert badge in ("Anomaly", "Warning"), (
            f"M3 (SLA invoice) produced badge OK — "
            f"possible extraction gap: SLA delay days not detected. "
            f"Anomalies: {[a.description for a in result_m3.audit_report.anomalies_found]}"
        )

    def test_m3_overbilling_is_decimal(self, result_m3: FullAuditResult) -> None:
        if result_m3.audit_report is None:
            pytest.skip("No audit report")
        assert isinstance(result_m3.audit_report.estimated_overbilling, Decimal)


# ---------------------------------------------------------------------------
# TestOutputTypes — cross-invoice type guarantees
# ---------------------------------------------------------------------------

class TestOutputTypes:
    """Verify that the pipeline produces correctly typed output regardless of content."""

    def test_audit_report_overbilling_decimal(self, result_m1: FullAuditResult) -> None:
        if result_m1.audit_report is None:
            pytest.skip("No audit report")
        assert isinstance(result_m1.audit_report.estimated_overbilling, Decimal)
        assert type(result_m1.audit_report.estimated_overbilling) is not float

    def test_audit_report_confidence_is_float(self, result_m1: FullAuditResult) -> None:
        if result_m1.audit_report is None:
            pytest.skip("No audit report")
        assert isinstance(result_m1.audit_report.overall_audit_confidence, float)

    def test_audit_report_badge_is_string(self, result_m1: FullAuditResult) -> None:
        if result_m1.audit_report is None:
            pytest.skip("No audit report")
        assert isinstance(result_m1.audit_report.confidence_badge, str)
        assert result_m1.audit_report.confidence_badge in ("OK", "Warning", "Anomaly")

    def test_full_result_is_pydantic_model(self, result_m2: FullAuditResult) -> None:
        assert isinstance(result_m2, FullAuditResult)

    def test_anomaly_overbilling_amounts_decimal(self, result_m2: FullAuditResult) -> None:
        if result_m2.audit_report is None:
            pytest.skip("No audit report")
        for anomaly in result_m2.audit_report.anomalies_found:
            assert isinstance(anomaly.overbilling_amount, Decimal), (
                f"overbilling_amount in '{anomaly.description}' is not Decimal"
            )
