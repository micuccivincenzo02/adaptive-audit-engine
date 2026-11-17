#!/usr/bin/env python3
"""
evaluate_copilot.py -- QA test runner for the FM Contract Auditor.

Reads fm_mvp_test_matrix.csv, generates a synthetic InvoiceData object via
the Anthropic API for each scenario, runs audit_invoice(), and prints a
terminal report with total pass/fail counts and detection accuracy.
"""

import os
import sys
import time

# -- Auto-install pandas if absent -----------------------------------------
try:
    import pandas as pd
except ImportError:
    import subprocess
    print("pandas not found — installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd

import anthropic
from dotenv import load_dotenv

from src.models import ContractRules, InvoiceData
from src.auditor import audit_invoice

load_dotenv()

# ── Golden contract used as the reference baseline for all 38 tests ───────
STANDARD_CONTRACT = ContractRules(
    base_monthly_fee=18_000.0,
    extra_hourly_rate=26.0,
    urgent_callout_fee=500.0,
    sla_penalty_per_day=500.0,
    indexation_allowed_month="January",
)

CSV_PATH = "Dati Test/fm_mvp_test_matrix.csv"

# ── ANSI colour helpers ────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}"


# ── Mock invoice generator ─────────────────────────────────────────────────
MOCK_TOOL_SCHEMA = {
    "name": "create_mock_invoice",
    "description": (
        "Build a synthetic InvoiceData object that faithfully represents "
        "the described test scenario against the standard FM contract."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "billed_base_fee": {
                "type": "number",
                "description": "Monthly base fee on this invoice (€). Contract value: 18000.",
            },
            "billed_extra_hours": {
                "type": "number",
                "description": "Extra hours billed. Use 0 if not relevant.",
            },
            "billed_extra_hourly_rate": {
                "type": "number",
                "description": "Hourly rate for extra hours (€). Use 0 if no extra hours. Contract rate: 26.",
            },
            "billed_urgent_callout_fee": {
                "type": "number",
                "description": "Urgent call-out fee billed (€). Use 0 if not applicable. Contract rate: 500.",
            },
            "unauthorized_services_total": {
                "type": "number",
                "description": (
                    "Total billed for unauthorized / out-of-scope services (€). "
                    "Use 0 if none. For scenarios involving unauthorized extras, "
                    "set a realistic positive amount."
                ),
            },
            "indexation_applied": {
                "type": "boolean",
                "description": (
                    "True if a price indexation or inflation adjustment was applied. "
                    "Contract allows indexation only from January."
                ),
            },
            "invoice_month": {
                "type": "string",
                "description": (
                    "The calendar month this invoice covers, as a full English month name "
                    "(e.g. 'January', 'March'). For early-indexation trap scenarios use the "
                    "month stated in the scenario (e.g. 'March'). For compliant indexation "
                    "scenarios use 'January' or later. Leave empty string if not relevant."
                ),
            },
            "sla_penalty_deducted": {
                "type": "number",
                "description": (
                    "Amount deducted for SLA penalties (€). Use 0 if none. "
                    "Contract rate: €500 per missed service day."
                ),
            },
            "detected_delay_days": {
                "type": "integer",
                "description": "Number of missed service days mentioned in the scenario. Use 0 if none.",
            },
        },
        "required": [
            "billed_base_fee",
            "billed_extra_hours",
            "billed_extra_hourly_rate",
            "billed_urgent_callout_fee",
            "unauthorized_services_total",
            "indexation_applied",
            "invoice_month",
            "sla_penalty_deducted",
            "detected_delay_days",
        ],
    },
}

MOCK_SYSTEM = (
    "You are a financial test-data generator for an FM contract auditing system. "
    "The reference contract has: base fee €18,000/month, extra hours at €26/hr, "
    "urgent call-out fee €500, SLA penalty €500/missed day, indexation only from January. "
    "For NEGATIVE scenarios inject the exact anomaly described. "
    "For POSITIVE scenarios return values that are 100 % contract-compliant."
)


def mock_invoice(client: anthropic.Anthropic, row: dict) -> InvoiceData:
    prompt = (
        f"Rule: {row['Rule_ID']} | Trap category: {row['Trap']} | "
        f"Type: {row['Scenario_Type']}\n"
        f"Scenario: {row['Test_Scenario']}\n"
        f"Expected finding: {row['Expected_AI_Finding']}"
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system=MOCK_SYSTEM,
        tools=[MOCK_TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": "create_mock_invoice"},
        messages=[{"role": "user", "content": prompt}],
    )
    tool_block = next(b for b in response.content if b.type == "tool_use")
    return InvoiceData(**tool_block.input)


# ── Main test runner ───────────────────────────────────────────────────────
def run_tests() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(c("ERROR: ANTHROPIC_API_KEY is not set in the environment.", RED))
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    df = pd.read_csv(CSV_PATH)
    total = len(df)

    print(f"\n{BOLD}{CYAN}{'=' * 72}{RESET}")
    print(f"{BOLD}{CYAN}   FM CONTRACT AUDITOR -- QA EVALUATION SUITE{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 72}{RESET}")
    print(f"   Reference contract : base E18,000 | extra E26/hr | callout E500 | SLA E500/day")
    print(f"   Test matrix        : {total} scenarios  ({total // 2} rule pairs)\n")

    passed = 0
    failed = 0
    results = []

    for _, row in df.iterrows():
        rule_id      = row["Rule_ID"]
        scenario_txt = row["Test_Scenario"]
        is_trap      = row["Scenario_Type"] == "Negative"
        label        = c("TRAP ", RED) if is_trap else c("CLEAN", GREEN)

        print(f"  [{rule_id}] {label} | {scenario_txt[:58]}...", end=" ", flush=True)

        try:
            invoice = mock_invoice(client, row)
            report  = audit_invoice(STANDARD_CONTRACT, invoice)

            # Negative scenario → expect an anomaly; Positive → expect clean pass
            test_passed = (not report.is_compliant) if is_trap else report.is_compliant

            status = c("PASS [OK]", GREEN) if test_passed else c("FAIL [X]", RED)
            detail = (
                f"{len(report.anomalies_found)} anomaly/ies | E{report.estimated_overbilling:.2f} flagged"
                if report.anomalies_found
                else "clean"
            )
            print(f"   {status}  [{detail}]")

            if test_passed:
                passed += 1
            else:
                failed += 1

            results.append({
                "rule_id":          rule_id,
                "trap":             row["Trap"],
                "scenario_type":    row["Scenario_Type"],
                "scenario":         scenario_txt,
                "expected_finding": row["Expected_AI_Finding"],
                "is_trap":          is_trap,
                "passed":           test_passed,
                "anomalies":        report.anomalies_found,
                "overbilling":      report.estimated_overbilling,
            })

        except Exception as exc:
            print(f"   {c('ERROR', YELLOW)}  [{exc}]")
            failed += 1
            results.append({
                "rule_id": rule_id, "trap": row["Trap"],
                "scenario_type": row["Scenario_Type"], "scenario": scenario_txt,
                "expected_finding": row["Expected_AI_Finding"],
                "is_trap": is_trap, "passed": False,
                "anomalies": [], "overbilling": 0.0, "error": str(exc),
            })

        time.sleep(0.4)   # stay within API rate limits

    # ── Summary report ────────────────────────────────────────────────────
    accuracy   = passed / total * 100 if total else 0.0
    trap_rows  = [r for r in results if r["is_trap"]]
    clean_rows = [r for r in results if not r["is_trap"]]
    trap_pass  = sum(1 for r in trap_rows  if r["passed"])
    clean_pass = sum(1 for r in clean_rows if r["passed"])

    acc_colour = GREEN if accuracy >= 80 else (YELLOW if accuracy >= 60 else RED)

    print(f"\n{BOLD}{CYAN}{'=' * 72}{RESET}")
    print(f"{BOLD}{CYAN}   EVALUATION REPORT{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 72}{RESET}")
    print(f"   {BOLD}Total Tests Run      :{RESET}  {total}")
    print(f"   {BOLD}Passed               :{RESET}  {c(str(passed), GREEN)}")
    print(f"   {BOLD}Failed               :{RESET}  {c(str(failed), RED)}")
    print(f"   {BOLD}Detection Accuracy   :{RESET}  {c(f'{accuracy:.1f} %', acc_colour + BOLD)}")
    print(f"{CYAN}{'-' * 72}{RESET}")

    trap_pct  = trap_pass  / len(trap_rows)  * 100 if trap_rows  else 0.0
    clean_pct = clean_pass / len(clean_rows) * 100 if clean_rows else 0.0
    print(f"   Negative (trap)  scenarios : {trap_pass:2d}/{len(trap_rows):2d} detected  "
          f"[{c(f'{trap_pct:.0f} %', GREEN if trap_pct == 100 else YELLOW)} detection rate]")
    print(f"   Positive (clean) scenarios : {clean_pass:2d}/{len(clean_rows):2d} correct   "
          f"[{c(f'{clean_pct:.0f} %', GREEN if clean_pct == 100 else YELLOW)} specificity]")

    failed_list = [r for r in results if not r["passed"]]
    if failed_list:
        print(f"\n{BOLD}{RED}   Failed / Coverage Gaps:{RESET}")
        print(f"   {'-' * 68}")
        for r in failed_list:
            if r["is_trap"]:
                tag = c("MISSED TRAP    ", RED)
                note = r.get("error") or "Auditor returned clean -- rule not in current model scope"
            else:
                tag = c("FALSE POSITIVE ", YELLOW)
                note = r["anomalies"][0][:65] if r["anomalies"] else r.get("error", "")
            print(f"   {BOLD}{r['rule_id']:<4}{RESET} [{tag}] {r['scenario'][:45]}")
            print(f"        Note : {note}")

    print(f"\n{BOLD}{CYAN}{'=' * 72}{RESET}\n")


if __name__ == "__main__":
    run_tests()
