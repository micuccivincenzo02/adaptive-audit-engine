# Adaptive Audit Engine IT/CH

An engine that reads a **contract and an invoice side by side** and explains, in plain language, where the numbers stop matching: wrong rates, unapplied penalty clauses, VAT and rounding errors, EUR/CHF issues.

**Case study page:** https://micuccivincenzo02.github.io/adaptive-audit-engine/

## Why
Facility-management and service contracts generate monthly invoices that nobody has time to check line by line against the contract terms. Small errors (a fee indexed a month early, a penalty clause not applied, a VAT rate off by one point) compound over multi-year contracts. Most small and mid-size teams have no dedicated auditor for this.

## How it works
Three steps, each with one job:

1. **Router** decides the jurisdiction (Italy or Switzerland) and applies the matching fiscal rules: VAT rates, rounding, currency.
2. **Extractor** reads PDFs and scans directly through Claude's Vision API (no local OCR) and returns validated line items, totals, dates and clauses, keeping the source text next to every extracted value.
3. **Critic** recalculates every total independently with fixed-precision arithmetic, compares it to the contract and marks a finding as OK only when there is evidence behind it.

Contract and invoice are the only required documents. A delivery note is used when available but is never mandatory.

## Stack
- Python 3.11+
- Anthropic API (Claude Vision) for document reading
- Pydantic models for every extracted fact
- `Decimal` for all monetary logic, never `float`
- Streamlit UI with in-memory session state
- pytest: **215 unit tests** covering IT/CH VAT rules, rounding edge cases, money handling and badge assignment

## Run it
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # then add your ANTHROPIC_API_KEY
streamlit run app.py
```

Tests:
```bash
pytest -m "not e2e" --ignore=tests/test_end_to_end.py
```
The end-to-end tests call the live API and need sample documents that are not included in this repository.

## Project structure
```
app.py                 Streamlit interface
audit_engine.py        audit logic and findings
src/core/              router, critic, orchestrator, money utilities
src/regions/           Italy and Switzerland extractors
src/fiscal_rules.py    IT/CH VAT and rounding rules
schemas/               Pydantic models
config/regions.py      regional configuration
tests/                 pytest suite
scripts/               generator for synthetic test invoices
Dati Test/             test scenario matrix (synthetic)
index.html             case study page (GitHub Pages)
```

## Data
No real client documents are included. Test scenarios and invoices are synthetic, modelled on public facility-management tender specifications.

## About
Designed and built solo by Vincenzo Micucci as product owner, using Claude Code as an AI pair programmer.
[LinkedIn](https://www.linkedin.com/in/vincenzo-micucci-7919802b0/)
