"""
Layer 2 — Italy Regional Extractor

Specialised for Italian facility-management documents written in Italian
or English.  Key regional assumptions baked into prompts and validation:

  Currency  : EUR (also written €, Euro)
  VAT label : IVA (Imposta sul Valore Aggiunto)
  VAT rate  : 22% standard rate (aliquota ordinaria)
  Common IT contract terms:
    Canone mensile, tariffa oraria, intervento urgente, penale giornaliera,
    indicizzazione ISTAT, capitolato tecnico, corrispettivo, oneri accessori
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
    "The document is Italian. "
    "Currency is EUR (also written €, 'Euro'). "
    "VAT is called IVA (Imposta sul Valore Aggiunto) and the standard rate is 22%%. "
    "Do NOT default to any rate — extract only what is explicitly written in the tax clause."
)

_CONTRACT_TOOL = _build_contract_tool(_REGION_CONTEXT)
_INVOICE_TOOL = _build_invoice_tool(_REGION_CONTEXT)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior legal and financial analyst specialised in Italian facility-management contracts.

CRITICAL RULES:
1. Currency is EUR. Amounts may be written with a comma as decimal separator (e.g. 1.500,00 €
   means one thousand five hundred euros). Parse Italian number formatting correctly.
2. VAT is IVA at 22% (aliquota ordinaria). Some contracts may cite reduced rates (10%, 4%)
   for specific services — extract exactly what is written, do not normalise.
3. Key Italian contract terms to recognise:
     base fee   → canone mensile / corrispettivo mensile / quota fissa
     hourly rate→ tariffa oraria / costo orario
     penalty    → penale / penale giornaliera / penalità
     indexation → indicizzazione ISTAT / adeguamento ISTAT / revisione prezzi
     urgent fee → intervento urgente / chiamata fuori orario / reperibilità
4. Indexation in Italian contracts is almost always tied to ISTAT indices and
   permitted only from a specific month. Extract that month carefully.
5. If a clause is absent, return 0 for numeric fields and empty string for text.
   NEVER invent values.
"""

# ---------------------------------------------------------------------------
# User prompts
# ---------------------------------------------------------------------------

_CONTRACT_USER_PREFIX = (
    "Estrai le regole contrattuali dal seguente contratto italiano di facility management. "
    "Presta attenzione alle clausole IVA, ai canoni mensili e alle penali SLA. "
    "Se il documento è in inglese, applica comunque le convenzioni italiane "
    "(valuta EUR, aliquota IVA 22%). "
    "Testo del documento:"
)

_INVOICE_USER_PREFIX = (
    "Estrai i dati di fatturazione dalla seguente fattura italiana di facility management. "
    "Cerca l'aliquota IVA (etichettata 'IVA'), eventuali ore extra, "
    "penali SLA e servizi non autorizzati. "
    "I numeri potrebbero usare il formato italiano (punto per migliaia, virgola per decimali). "
    "Testo del documento:"
)


# ---------------------------------------------------------------------------
# Italy extractor
# ---------------------------------------------------------------------------

class ItalyExtractor(BaseExtractor):

    EXTRACTOR_ID = "italy_v1"

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

        if str(raw.get("currency", "")).upper() != "EUR":
            warnings.append(
                f"Expected currency EUR but extractor returned '{raw.get('currency')}'. "
                "Critic should verify."
            )

        vat = float(raw.get("vat_rate", 0.0))
        if vat not in (0.0, 4.0, 10.0, 22.0):
            warnings.append(
                f"Extracted VAT rate {vat}% is outside known Italian IVA rates "
                "(4%, 10%, 22%). Critic should verify tax clause."
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
        if vat not in (0.0, 4.0, 10.0, 22.0):
            warnings.append(
                f"Billed VAT rate {vat}% is outside known Italian IVA rates. "
                "Critic should verify."
            )

        return RawExtraction(
            fields=raw,
            extractor_id=self.EXTRACTOR_ID,
            extraction_warnings=warnings,
        )
