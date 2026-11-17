"""
Layer 2 — Switzerland Regional Extractor

Specialised for Swiss facility-management documents that may be written in
Italian (Ticino), German, or French.  Key regional assumptions baked into
prompts and validation:

  Currency  : CHF (also written Fr., SFr., sFr., franchi svizzeri)
  VAT label : MWST (Mehrwertsteuer) | TVA (taxe sur la valeur ajoutée) |
              IVA (imposta sul valore aggiunto, Ticino)
  VAT rate  : 8.1% standard rate (valid from 01.01.2024 — previous rate 7.7%)
  Common CH contract terms:
    de: Pauschalpreis, Stundensatz, Notfalleinsatz, Konventionalstrafe,
        Teuerungsanpassung, Leistungsverzeichnis
    fr: Prix forfaitaire, taux horaire, intervention urgente, pénalité,
        indexation des prix, cahier des charges
    it: Canone forfettario, tariffa oraria, intervento urgente, penale,
        indicizzazione, capitolato
"""

from __future__ import annotations

from src.core.router import DocumentContext
from src.regions.base_extractor import (
    BaseExtractor,
    RawExtraction,
    _build_contract_tool,
    _build_invoice_tool,
)

# ---------------------------------------------------------------------------
# Region-specific context injected into shared tool schemas
# ---------------------------------------------------------------------------

_REGION_CONTEXT = (
    "The document is Swiss. "
    "Currency is CHF (also written Fr., SFr., sFr., or 'franchi svizzeri'). "
    "VAT may appear as MWST (German), TVA (French), or IVA (Italian/Ticino) and the "
    "standard rate since 01.01.2024 is 8.1%%. "
    "Do NOT default to any rate — extract only what is written."
)

_CONTRACT_TOOL = _build_contract_tool(_REGION_CONTEXT)
_INVOICE_TOOL = _build_invoice_tool(_REGION_CONTEXT)

# ---------------------------------------------------------------------------
# System prompt — enforces Swiss-specific rules for the LLM
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior legal and financial analyst specialised in Swiss facility-management contracts.

CRITICAL RULES:
1. Currency is always CHF. If you see €, flag it as suspicious.
2. The Swiss VAT rate is 8.1% (since 01.01.2024). Previous contracts may cite 7.7% — extract
   exactly what is written, do not normalise.
3. The document may be in German, French, or Italian (or a mix for Ticino documents).
   Recognise equivalent terms across all three languages:
     base fee   → Pauschalpreis / prix forfaitaire / canone forfettario
     hourly rate→ Stundensatz / taux horaire / tariffa oraria
     penalty    → Konventionalstrafe / pénalité / penale
     indexation → Teuerungsanpassung / indexation / indicizzazione
     urgent fee → Notfalleinsatz / intervention urgente / intervento urgente
4. If a clause is absent from the document, return 0 for numeric fields and
   empty string for text fields. NEVER invent values.
5. SLA penalties often appear in Swiss contracts as a percentage of the monthly
   fee — convert to the absolute CHF amount per day when possible.
"""

# ---------------------------------------------------------------------------
# User prompts
# ---------------------------------------------------------------------------

_CONTRACT_USER_PREFIX = (
    "Extract all contract rules from the following Swiss facility-management contract. "
    "The document may be in German, French, or Italian. "
    "Pay close attention to tax clauses (MWST / TVA / IVA) and currency (CHF). "
    "Document text:"
)

_INVOICE_USER_PREFIX = (
    "Extract all billing data from the following Swiss facility-management invoice. "
    "The document may be in German, French, or Italian. "
    "Look for MWST / TVA / IVA labels for the tax rate. "
    "Identify any mention of Verzögerung / retard / ritardo (delays) "
    "or Konventionalstrafe / pénalité / penale (SLA penalties). "
    "Document text:"
)


# ---------------------------------------------------------------------------
# Switzerland extractor
# ---------------------------------------------------------------------------

class SwitzerlandExtractor(BaseExtractor):

    EXTRACTOR_ID = "switzerland_v1"

    def extract_contract(
        self, text: str, context: DocumentContext
    ) -> RawExtraction:
        warnings: list[str] = []

        raw = self._call_llm(
            text=text,
            tool=_CONTRACT_TOOL,
            system_prompt=_SYSTEM_PROMPT,
            user_prefix=_CONTRACT_USER_PREFIX,
        )

        # Regional sanity checks — non-fatal, recorded as warnings for the Critic
        if str(raw.get("currency", "")).upper() != "CHF":
            warnings.append(
                f"Expected currency CHF but extractor returned '{raw.get('currency')}'. "
                "Possible EUR/CHF confusion — Critic should verify."
            )

        vat = float(raw.get("vat_rate", 0.0))
        if vat not in (0.0, 7.7, 8.1):
            warnings.append(
                f"Extracted VAT rate {vat}% is outside known Swiss rates (7.7% or 8.1%). "
                "Critic should verify tax clause."
            )

        return RawExtraction(
            fields=raw,
            extractor_id=self.EXTRACTOR_ID,
            extraction_warnings=warnings,
        )

    def extract_invoice(
        self, text: str, context: DocumentContext
    ) -> RawExtraction:
        warnings: list[str] = []

        raw = self._call_llm(
            text=text,
            tool=_INVOICE_TOOL,
            system_prompt=_SYSTEM_PROMPT,
            user_prefix=_INVOICE_USER_PREFIX,
        )

        vat = float(raw.get("billed_vat_rate", 0.0))
        if vat not in (0.0, 7.7, 8.1):
            warnings.append(
                f"Billed VAT rate {vat}% is outside known Swiss rates (7.7% or 8.1%). "
                "Critic should verify."
            )

        return RawExtraction(
            fields=raw,
            extractor_id=self.EXTRACTOR_ID,
            extraction_warnings=warnings,
        )
