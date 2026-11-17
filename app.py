"""
app.py — Adaptive Audit Engine IT/CH · Streamlit dashboard
"""
from __future__ import annotations

import os
from datetime import date
from typing import Optional

import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Audit Engine IT/CH",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

from audit_engine import AuditEngine
from src.extractor import UniversalExtractor
from src.models import AuditError, AuditResult, DocumentFact, Finding, Money

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_BG     = {"OK": "#d4edda", "Warning": "#fff3cd", "Anomaly": "#f8d7da"}
_FG     = {"OK": "#155724", "Warning": "#856404", "Anomaly": "#721c24"}
_BORDER = {"OK": "#28a745", "Warning": "#ffc107", "Anomaly": "#dc3545"}
_ICON   = {"OK": "🟢",      "Warning": "🟡",      "Anomaly": "🔴"}

_DOC_ICON  = {"CONTRACT": "📄", "INVOICE": "🧾", "DDT": "📦", "UNKNOWN": "❓"}
_DOC_COLOR = {
    "CONTRACT": ("#d4edda", "#155724"),
    "INVOICE":  ("#cce5ff", "#004085"),
    "DDT":      ("#fff3cd", "#856404"),
    "UNKNOWN":  ("#f8d7da", "#721c24"),
}

_EXCLUSION_KWS = [
    "esclu", "non include", "non comprend", "not cover",
    "not included", "ausgeschl", "nicht enthalten", "exclu",
]

_CHAT_SYSTEM = """\
Sei un assistente specializzato nell'analisi di documenti commerciali e fiscali.
Hai accesso ai dati strutturati estratti dai documenti caricati dall'utente.
Rispondi ESCLUSIVAMENTE in base ai dati forniti nel contesto.
Se l'informazione richiesta non è presente nei documenti, dichiaralo esplicitamente.
Non inventare dati, clausole o importi. Sii conciso e professionale.
Rispondi sempre nella stessa lingua in cui l'utente ha posto la domanda \
(italiano, inglese, tedesco o francese).
"""

# ─────────────────────────────────────────────────────────────────────────────
# Localisation
# ─────────────────────────────────────────────────────────────────────────────

_LANG_OPTIONS = ["English", "Italiano", "Français", "Deutsch"]
_LANG_CODE    = {"English": "en", "Italiano": "it", "Français": "fr", "Deutsch": "de"}

UI_LABELS: dict[str, dict[str, str]] = {
    "it": {
        "sidebar_title": "## 🔍 AI Audit Engine",
        "sidebar_caption": "Audit automatizzato contratti e fatture · IT / CH",
        "new_audit_btn": "🔄 Nuovo Audit",
        "legend_title": "### 📖 Legenda Badge",
        "legend_ok_desc": "Nessuna anomalia. Importi e clausole conformi al contratto.",
        "legend_warning_desc": "Deviazione minore o dato incerto. Revisione manuale consigliata.",
        "legend_anomaly_desc": "Discrepanza significativa. Probabile overbilling o clausola violata.",
        "sidebar_jurisdictions": "Giurisdizioni: 🇮🇹 Italia (EUR) · 🇨🇭 Svizzera (CHF)",
        "sidebar_threshold": "Soglia Warning/Anomalia prezzi: 2%",
        "sidebar_vat": "IVA IT: 22% · IVA CH: 8.1%",
        "welcome_heading": "AI Audit Engine IT/CH",
        "welcome_subtitle": "Carica i tuoi documenti PDF per un audit automatico immediato.<br>Il motore rileva anomalie di prezzo, quantità, clausole e IVA.",
        "card1_title": "Contratto",
        "card1_sub": "Termini, prezzi, clausole, SLA, date",
        "card2_title": "Fattura",
        "card2_sub": "Voci di spesa, IVA, totali, fornitore",
        "card3_title": "DDT",
        "card3_optional": "opzionale",
        "card3_sub": "Quantità consegnate, confronto con fattura",
        "welcome_info": "**Come iniziare:** trascina i file nel pannello di caricamento qui sopra oppure clicca **Browse files**. Sono supportati PDF (anche scannerizzati), JPEG, PNG e WebP. Il motore analizza ogni documento tramite Vision AI e avvia l'audit in pochi secondi.",
        "upload_label": "Carica i documenti (contratto, fattura, DDT) — PDF o immagine",
        "upload_help": "Supportati: PDF (anche scannerizzati), JPEG, PNG, WebP. Il motore classifica automaticamente ogni documento.",
        "upload_invalid": "File '{name}' non valido o vuoto — sarà ignorato.",
        "upload_n_single": "{n} file selezionato:",
        "upload_n_multi": "{n} file selezionati:",
        "run_audit_btn": "▶ Avvia Audit",
        "docs_title": "#### 📁 Documenti Analizzati",
        "vendor_label": "Fornitore",
        "client_label": "Cliente",
        "date_label": "Data",
        "jurisdiction_label": "Giurisdizione",
        "currency_label": "Valuta",
        "extraction_conf": "Confidenza estrazione",
        "key_points_title": "**Punti chiave**",
        "truncated_doc": "Documento esteso: analisi focalizzata sulle clausole economiche e tecniche",
        "kp_invoice_total": "💰 Totale fattura",
        "kp_vat": "🏛️ IVA",
        "kp_line_items": "📋 Voci di spesa",
        "kp_items_fmt": "{n} voci: {sample}{suffix}",
        "kp_more": "… +{n} altre",
        "kp_duration": "📅 Durata",
        "kp_start": "📅 Inizio",
        "kp_main_clause": "📜 Clausola principale",
        "kp_contract_prices": "💶 Prezzi contrattuali",
        "kp_price_list": "{n} voci di listino",
        "kp_qty_delivered": "📦 Quantità consegnata",
        "kp_articles": "📋 Articoli",
        "kp_articles_fmt": "{n} voci: {sample}",
        "kp_delivery_date": "📅 Data consegna",
        "kp_detected_vendor": "🏢 Fornitore rilevato",
        "kp_detected_amount": "💰 Importo rilevato",
        "kp_unknown_type": "⚠️ Tipo non riconosciuto",
        "kp_manual_review": "Revisione manuale consigliata",
        "doc_CONTRACT": "Contratto",
        "doc_INVOICE": "Fattura",
        "doc_DDT": "Doc. di Trasporto",
        "doc_UNKNOWN": "Documento Sconosciuto",
        "type_QUANTITY": "📦 Quantità",
        "type_PRICING": "💶 Prezzi",
        "type_CONTRACTUAL": "📋 Contrattuale",
        "type_FISCAL": "🏛️ Fiscale",
        "type_SLA": "⏱️ SLA",
        "type_OTHER": "❓ Altro",
        "identikit_vendor": "🏢 Fornitore",
        "identikit_client": "👤 Cliente",
        "identikit_jurisdiction": "⚖️ Giurisdizione",
        "identikit_contract_status": "#### ⏱️ Stato del Contratto",
        "identikit_start_date": "📅 Data Inizio",
        "identikit_end_date": "📅 Data Fine",
        "identikit_days_remaining": "⏳ Giorni Rimanenti",
        "identikit_elapsed_delta": "{pct:.0%} trascorso",
        "identikit_progress_text": "Durata contratto: {pct:.0%} trascorsa — {days} giorni rimanenti",
        "identikit_no_dates": "Date contrattuali non rilevate.",
        "identikit_included": "#### ✅ Servizi Inclusi",
        "identikit_exclusions": "#### 🚫 Esclusioni / Clausole Critiche",
        "identikit_per_unit": "/ unità",
        "identikit_no_services": "Nessuna voce di servizio estratta dal contratto.",
        "identikit_no_exclusions": "Nessuna clausola di esclusione rilevata.",
        "identikit_all_clauses": "📜 Tutte le clausole ({n})",
        "identikit_no_contract": "Nessun contratto caricato. Carica un contratto per visualizzare l'Identikit.",
        "session_vendor": "**Fornitore**",
        "session_client": "**Cliente**",
        "session_invoice_date": "**Data Fattura**",
        "session_invoice_total": "**Totale Fattura**",
        "metric_savings": "💰 Risparmio Potenziale",
        "metric_savings_help_none": "Nessuna anomalia finanziaria rilevata",
        "metric_savings_help_found": "Somma degli impatti finanziari rilevati",
        "metric_anomalies": "🔴 Anomalie",
        "metric_warnings": "🟡 Avvisi",
        "metric_compliant": "🟢 Conformi",
        "audit_summary_title": "#### 📊 Riepilogo Audit",
        "findings_title": "#### 📋 Rilievi",
        "filter_label": "Filtra rilievi",
        "filter_all": "Tutti",
        "filter_anomaly": "🔴 Anomalia",
        "filter_review": "🟡 Revisione",
        "filter_ok": "🟢 Conforme",
        "findings_count": "{n} rilievi",
        "findings_filter_active": "  ·  filtro attivo: **{filter}**",
        "findings_click_hint": "  ·  Clicca una riga per dettaglio ed evidenza.",
        "findings_none": "✅ Nessun rilievo per il filtro selezionato.",
        "findings_select_hint": "👆 Seleziona una riga per vedere dettaglio, prove e testo di contestazione.",
        "findings_detail_title": "#### 🔎 Dettaglio — Rilievo #{n}",
        "col_type": "Tipo",
        "col_status": "Stato",
        "col_motivation": "Motivazione",
        "col_impact": "Impatto",
        "detail_financial_impact": "💰 Impatto Finanziario",
        "detail_rule_prefix": "📜 **Regola:**",
        "detail_no_rule": "Nessun riferimento normativo esplicito.",
        "detail_evidence_title": "##### 📎 Prove a Confronto",
        "detail_invoice_label": "🧾 Fattura",
        "detail_contract_label": "📄 Contratto",
        "detail_field_label": "Campo",
        "detail_page_label": "Pag.",
        "detail_confidence_label": "Confidenza",
        "detail_no_invoice_ev": "Nessuna evidenza dalla fattura.",
        "detail_no_contract_ev": "Nessuna evidenza dal contratto.",
        "detail_no_evidence": "⚠️ Nessuna evidenza documentale allegata.",
        "detail_contestation_title": "##### 📋 Testo Pronto per la Contestazione",
        "detail_contestation_caption": "Copia il testo sopra e invialo direttamente al fornitore per richiedere chiarimenti.",
        "banner_anomaly_one": "🔴 <b>1 anomalia rilevata</b>",
        "banner_anomaly_many": "🔴 <b>{n} anomalie rilevate</b>",
        "banner_anomaly_suffix": " &nbsp;—&nbsp; Richiesta revisione o contestazione immediata per i valori indicati.",
        "banner_impact": " &nbsp;·&nbsp; Impatto potenziale: <b>{amount}</b>",
        "banner_warning_one": "🟡 <b>1 voce da revisionare</b>",
        "banner_warning_many": "🟡 <b>{n} voci da revisionare</b>",
        "banner_warning_suffix": " &nbsp;—&nbsp; Alcune voci richiedono verifica manuale prima dell'approvazione.",
        "banner_ok": "✅ <b>Tutto conforme</b> &nbsp;—&nbsp; Nessuna irregolarità o discrepanza finanziaria rilevata.",
        "chat_title": "#### 💬 Domande sui Documenti",
        "chat_caption": "Fai domande sui documenti caricati — il motore risponde esclusivamente in base ai dati estratti.",
        "chat_context_label": "📂 Contesto disponibile:",
        "chat_placeholder": "Esempio: Qual è la scadenza del contratto? / Ci sono ore straordinarie?",
        "chat_spinner": "Elaborazione…",
        "main_title": "🔍 AI Audit Engine IT/CH",
        "main_caption": "Audit automatizzato · Contratti & Fatture · 🇮🇹 Italia (EUR) · 🇨🇭 Svizzera (CHF)",
        "main_docs_expander": "📁 Documenti analizzati e sorgenti",
        "main_no_invoice": "**Nessuna fattura rilevata.** Carica almeno una fattura per avviare l'analisi delle anomalie. Puoi già esplorare il contratto e interrogare i documenti nell'Assistente.",
        "progress_init": "Inizializzazione…",
        "progress_vision": "Analisi Vision AI: {name}…",
        "progress_complete": "Estrazione completata.",
        "progress_audit": "Esecuzione audit…",
        "toast_complete": "Audit completato!",
        "error_audit_failed": "**Audit non completato:** {exc}",
        "error_no_readable": "Nessun documento leggibile. Verifica che i file siano validi (PDF non protetti da password, o immagini JPEG/PNG/WebP leggibili).",
        "error_config": "**Configurazione non valida:** {exc}",
        "contestation_subject": "Oggetto: Contestazione anomalia — {type_label}",
        "contestation_greeting": "Gentile fornitore,",
        "contestation_body": "A seguito dell'audit documentale è stata rilevata la seguente anomalia{impact}:",
        "contestation_rule_prefix": "\n\nRiferimento contrattuale: {ref}",
        "contestation_closing": "Si richiede un chiarimento e, ove dovuto, l'emissione di nota di credito.",
        "contestation_salutation": "Cordiali saluti",
        "contestation_impact_suffix": " con un impatto stimato di {amount}",
        "tab_identikit": "📋 Identikit Contratto",
        "tab_findings":  "🔍 Rilievi Audit",
        "tab_chat":      "💬 Assistente",
    },
    "en": {
        "sidebar_title": "## 🔍 AI Audit Engine",
        "sidebar_caption": "Automated audit · Contracts & Invoices · IT / CH",
        "new_audit_btn": "🔄 New Audit",
        "legend_title": "### 📖 Badge Legend",
        "legend_ok_desc": "No anomaly. Amounts and clauses comply with the contract.",
        "legend_warning_desc": "Minor deviation or uncertain data. Manual review recommended.",
        "legend_anomaly_desc": "Significant discrepancy. Likely overbilling or violated clause.",
        "sidebar_jurisdictions": "Jurisdictions: 🇮🇹 Italy (EUR) · 🇨🇭 Switzerland (CHF)",
        "sidebar_threshold": "Warning/Anomaly price threshold: 2%",
        "sidebar_vat": "VAT IT: 22% · VAT CH: 8.1%",
        "welcome_heading": "AI Audit Engine IT/CH",
        "welcome_subtitle": "Upload your PDF documents for an immediate automated audit.<br>The engine detects price, quantity, clause and VAT anomalies.",
        "card1_title": "Contract",
        "card1_sub": "Terms, prices, clauses, SLA, dates",
        "card2_title": "Invoice",
        "card2_sub": "Line items, VAT, totals, vendor",
        "card3_title": "DDT",
        "card3_optional": "optional",
        "card3_sub": "Delivered quantities, comparison with invoice",
        "welcome_info": "**Getting started:** drag files into the upload panel above or click **Browse files**. Supported formats: PDF (incl. scanned), JPEG, PNG and WebP. The engine analyses each document via Vision AI and starts the audit within seconds.",
        "upload_label": "Upload documents (contract, invoice, DDT) — PDF or image",
        "upload_help": "Supported: PDF (incl. scanned), JPEG, PNG, WebP. The engine auto-classifies each document.",
        "upload_invalid": "File '{name}' is invalid or empty — it will be ignored.",
        "upload_n_single": "{n} file selected:",
        "upload_n_multi": "{n} files selected:",
        "run_audit_btn": "▶ Run Audit",
        "docs_title": "#### 📁 Analysed Documents",
        "vendor_label": "Vendor",
        "client_label": "Client",
        "date_label": "Date",
        "jurisdiction_label": "Jurisdiction",
        "currency_label": "Currency",
        "extraction_conf": "Extraction confidence",
        "key_points_title": "**Key Points**",
        "truncated_doc": "Extended document: analysis focused on economic and technical clauses",
        "kp_invoice_total": "💰 Invoice Total",
        "kp_vat": "🏛️ VAT",
        "kp_line_items": "📋 Line Items",
        "kp_items_fmt": "{n} items: {sample}{suffix}",
        "kp_more": "… +{n} more",
        "kp_duration": "📅 Duration",
        "kp_start": "📅 Start",
        "kp_main_clause": "📜 Main Clause",
        "kp_contract_prices": "💶 Contract Prices",
        "kp_price_list": "{n} price list items",
        "kp_qty_delivered": "📦 Quantity Delivered",
        "kp_articles": "📋 Articles",
        "kp_articles_fmt": "{n} items: {sample}",
        "kp_delivery_date": "📅 Delivery Date",
        "kp_detected_vendor": "🏢 Vendor Detected",
        "kp_detected_amount": "💰 Amount Detected",
        "kp_unknown_type": "⚠️ Type Not Recognised",
        "kp_manual_review": "Manual review recommended",
        "doc_CONTRACT": "Contract",
        "doc_INVOICE": "Invoice",
        "doc_DDT": "Delivery Note",
        "doc_UNKNOWN": "Unknown Document",
        "type_QUANTITY": "📦 Quantity",
        "type_PRICING": "💶 Pricing",
        "type_CONTRACTUAL": "📋 Contractual",
        "type_FISCAL": "🏛️ Fiscal",
        "type_SLA": "⏱️ SLA",
        "type_OTHER": "❓ Other",
        "identikit_vendor": "🏢 Vendor",
        "identikit_client": "👤 Client",
        "identikit_jurisdiction": "⚖️ Jurisdiction",
        "identikit_contract_status": "#### ⏱️ Contract Status",
        "identikit_start_date": "📅 Start Date",
        "identikit_end_date": "📅 End Date",
        "identikit_days_remaining": "⏳ Days Remaining",
        "identikit_elapsed_delta": "{pct:.0%} elapsed",
        "identikit_progress_text": "Contract duration: {pct:.0%} elapsed — {days} days remaining",
        "identikit_no_dates": "Contract dates not detected.",
        "identikit_included": "#### ✅ Included Services",
        "identikit_exclusions": "#### 🚫 Exclusions / Critical Clauses",
        "identikit_per_unit": "/ unit",
        "identikit_no_services": "No service items extracted from the contract.",
        "identikit_no_exclusions": "No exclusion clauses detected.",
        "identikit_all_clauses": "📜 All clauses ({n})",
        "identikit_no_contract": "No contract loaded. Upload a contract to display the Identikit.",
        "session_vendor": "**Vendor**",
        "session_client": "**Client**",
        "session_invoice_date": "**Invoice Date**",
        "session_invoice_total": "**Invoice Total**",
        "metric_savings": "💰 Potential Savings",
        "metric_savings_help_none": "No financial anomaly detected",
        "metric_savings_help_found": "Sum of detected financial impacts",
        "metric_anomalies": "🔴 Anomalies",
        "metric_warnings": "🟡 Warnings",
        "metric_compliant": "🟢 Compliant",
        "audit_summary_title": "#### 📊 Audit Summary",
        "findings_title": "#### 📋 Findings",
        "filter_label": "Filter findings",
        "filter_all": "All",
        "filter_anomaly": "🔴 Anomaly",
        "filter_review": "🟡 Review",
        "filter_ok": "🟢 Compliant",
        "findings_count": "{n} findings",
        "findings_filter_active": "  ·  active filter: **{filter}**",
        "findings_click_hint": "  ·  Click a row for detail and evidence.",
        "findings_none": "✅ No findings for the selected filter.",
        "findings_select_hint": "👆 Select a row to view detail, evidence and contestation text.",
        "findings_detail_title": "#### 🔎 Detail — Finding #{n}",
        "col_type": "Type",
        "col_status": "Status",
        "col_motivation": "Motivation",
        "col_impact": "Impact",
        "detail_financial_impact": "💰 Financial Impact",
        "detail_rule_prefix": "📜 **Rule:**",
        "detail_no_rule": "No explicit regulatory reference.",
        "detail_evidence_title": "##### 📎 Evidence Side-by-Side",
        "detail_invoice_label": "🧾 Invoice",
        "detail_contract_label": "📄 Contract",
        "detail_field_label": "Field",
        "detail_page_label": "Pag.",
        "detail_confidence_label": "Confidence",
        "detail_no_invoice_ev": "No evidence from invoice.",
        "detail_no_contract_ev": "No evidence from contract.",
        "detail_no_evidence": "⚠️ No documentary evidence attached.",
        "detail_contestation_title": "##### 📋 Ready-to-Send Contestation Text",
        "detail_contestation_caption": "Copy the text above and send it directly to the vendor to request clarification.",
        "banner_anomaly_one": "🔴 <b>1 anomaly detected</b>",
        "banner_anomaly_many": "🔴 <b>{n} anomalies detected</b>",
        "banner_anomaly_suffix": " &nbsp;—&nbsp; Immediate review or contestation required for the indicated values.",
        "banner_impact": " &nbsp;·&nbsp; Potential impact: <b>{amount}</b>",
        "banner_warning_one": "🟡 <b>1 item to review</b>",
        "banner_warning_many": "🟡 <b>{n} items to review</b>",
        "banner_warning_suffix": " &nbsp;—&nbsp; Some items require manual verification before approval.",
        "banner_ok": "✅ <b>All compliant</b> &nbsp;—&nbsp; No irregularity or financial discrepancy detected.",
        "chat_title": "#### 💬 Questions on Documents",
        "chat_caption": "Ask questions about the uploaded documents — the engine responds exclusively based on extracted data.",
        "chat_context_label": "📂 Available context:",
        "chat_placeholder": "Example: What is the contract expiry? / Are there overtime hours?",
        "chat_spinner": "Processing…",
        "main_title": "🔍 AI Audit Engine IT/CH",
        "main_caption": "Automated audit · Contracts & Invoices · 🇮🇹 Italy (EUR) · 🇨🇭 Switzerland (CHF)",
        "main_docs_expander": "📁 Analysed documents and sources",
        "main_no_invoice": "**No invoice detected.** Upload at least one invoice to start anomaly analysis. You can already explore the contract and query documents in the Assistant.",
        "progress_init": "Initialising…",
        "progress_vision": "Vision AI analysis: {name}…",
        "progress_complete": "Extraction complete.",
        "progress_audit": "Running audit…",
        "toast_complete": "Audit completed!",
        "error_audit_failed": "**Audit not completed:** {exc}",
        "error_no_readable": "No readable document. Check that files are valid (non-password-protected PDFs, or readable JPEG/PNG/WebP images).",
        "error_config": "**Invalid configuration:** {exc}",
        "contestation_subject": "Subject: Anomaly contestation — {type_label}",
        "contestation_greeting": "Dear vendor,",
        "contestation_body": "Following the document audit, the following anomaly was identified{impact}:",
        "contestation_rule_prefix": "\n\nContractual reference: {ref}",
        "contestation_closing": "We request a clarification and, where applicable, the issuance of a credit note.",
        "contestation_salutation": "Kind regards",
        "contestation_impact_suffix": " with an estimated impact of {amount}",
        "tab_identikit": "📋 Contract Identikit",
        "tab_findings":  "🔍 Audit Findings",
        "tab_chat":      "💬 AI Assistant",
    },
    "fr": {
        "sidebar_title": "## 🔍 AI Audit Engine",
        "sidebar_caption": "Audit automatisé · Contrats & Factures · IT / CH",
        "new_audit_btn": "🔄 Nouvel Audit",
        "legend_title": "### 📖 Légende des Badges",
        "legend_ok_desc": "Aucune anomalie. Montants et clauses conformes au contrat.",
        "legend_warning_desc": "Écart mineur ou donnée incertaine. Révision manuelle recommandée.",
        "legend_anomaly_desc": "Divergence significative. Probable surfacturation ou clause violée.",
        "sidebar_jurisdictions": "Juridictions : 🇮🇹 Italie (EUR) · 🇨🇭 Suisse (CHF)",
        "sidebar_threshold": "Seuil Avertissement/Anomalie prix : 2 %",
        "sidebar_vat": "TVA IT : 22 % · TVA CH : 8,1 %",
        "welcome_heading": "AI Audit Engine IT/CH",
        "welcome_subtitle": "Chargez vos documents PDF pour un audit automatique immédiat.<br>Le moteur détecte les anomalies de prix, de quantité, de clauses et de TVA.",
        "card1_title": "Contrat",
        "card1_sub": "Termes, prix, clauses, SLA, dates",
        "card2_title": "Facture",
        "card2_sub": "Lignes de dépenses, TVA, totaux, fournisseur",
        "card3_title": "BL",
        "card3_optional": "optionnel",
        "card3_sub": "Quantités livrées, comparaison avec la facture",
        "welcome_info": "**Pour commencer :** faites glisser les fichiers dans le panneau ci-dessus ou cliquez sur **Browse files**. Formats supportés : PDF (y compris numérisés), JPEG, PNG et WebP. Le moteur analyse chaque document via Vision AI et lance l'audit en quelques secondes.",
        "upload_label": "Charger les documents (contrat, facture, BL) — PDF ou image",
        "upload_help": "Supportés : PDF (y compris numérisés), JPEG, PNG, WebP. Le moteur classifie automatiquement chaque document.",
        "upload_invalid": "Fichier '{name}' invalide ou vide — il sera ignoré.",
        "upload_n_single": "{n} fichier sélectionné :",
        "upload_n_multi": "{n} fichiers sélectionnés :",
        "run_audit_btn": "▶ Lancer l'Audit",
        "docs_title": "#### 📁 Documents Analysés",
        "vendor_label": "Fournisseur",
        "client_label": "Client",
        "date_label": "Date",
        "jurisdiction_label": "Juridiction",
        "currency_label": "Devise",
        "extraction_conf": "Confiance d'extraction",
        "key_points_title": "**Points clés**",
        "truncated_doc": "Document étendu : analyse centrée sur les clauses économiques et techniques",
        "kp_invoice_total": "💰 Total facture",
        "kp_vat": "🏛️ TVA",
        "kp_line_items": "📋 Lignes de dépenses",
        "kp_items_fmt": "{n} lignes : {sample}{suffix}",
        "kp_more": "… +{n} autres",
        "kp_duration": "📅 Durée",
        "kp_start": "📅 Début",
        "kp_main_clause": "📜 Clause principale",
        "kp_contract_prices": "💶 Prix contractuels",
        "kp_price_list": "{n} postes de prix",
        "kp_qty_delivered": "📦 Quantité livrée",
        "kp_articles": "📋 Articles",
        "kp_articles_fmt": "{n} lignes : {sample}",
        "kp_delivery_date": "📅 Date de livraison",
        "kp_detected_vendor": "🏢 Fournisseur détecté",
        "kp_detected_amount": "💰 Montant détecté",
        "kp_unknown_type": "⚠️ Type non reconnu",
        "kp_manual_review": "Révision manuelle recommandée",
        "doc_CONTRACT": "Contrat",
        "doc_INVOICE": "Facture",
        "doc_DDT": "Bon de livraison",
        "doc_UNKNOWN": "Document inconnu",
        "type_QUANTITY": "📦 Quantité",
        "type_PRICING": "💶 Prix",
        "type_CONTRACTUAL": "📋 Contractuel",
        "type_FISCAL": "🏛️ Fiscal",
        "type_SLA": "⏱️ SLA",
        "type_OTHER": "❓ Autre",
        "identikit_vendor": "🏢 Fournisseur",
        "identikit_client": "👤 Client",
        "identikit_jurisdiction": "⚖️ Juridiction",
        "identikit_contract_status": "#### ⏱️ État du Contrat",
        "identikit_start_date": "📅 Date de début",
        "identikit_end_date": "📅 Date de fin",
        "identikit_days_remaining": "⏳ Jours restants",
        "identikit_elapsed_delta": "{pct:.0%} écoulé",
        "identikit_progress_text": "Durée du contrat : {pct:.0%} écoulée — {days} jours restants",
        "identikit_no_dates": "Dates contractuelles non détectées.",
        "identikit_included": "#### ✅ Services inclus",
        "identikit_exclusions": "#### 🚫 Exclusions / Clauses critiques",
        "identikit_per_unit": "/ unité",
        "identikit_no_services": "Aucune prestation extraite du contrat.",
        "identikit_no_exclusions": "Aucune clause d'exclusion détectée.",
        "identikit_all_clauses": "📜 Toutes les clauses ({n})",
        "identikit_no_contract": "Aucun contrat chargé. Chargez un contrat pour afficher l'Identikit.",
        "session_vendor": "**Fournisseur**",
        "session_client": "**Client**",
        "session_invoice_date": "**Date de facture**",
        "session_invoice_total": "**Total facture**",
        "metric_savings": "💰 Économies potentielles",
        "metric_savings_help_none": "Aucune anomalie financière détectée",
        "metric_savings_help_found": "Somme des impacts financiers détectés",
        "metric_anomalies": "🔴 Anomalies",
        "metric_warnings": "🟡 Avertissements",
        "metric_compliant": "🟢 Conformes",
        "audit_summary_title": "#### 📊 Résumé de l'Audit",
        "findings_title": "#### 📋 Constatations",
        "filter_label": "Filtrer les constatations",
        "filter_all": "Toutes",
        "filter_anomaly": "🔴 Anomalie",
        "filter_review": "🟡 À réviser",
        "filter_ok": "🟢 Conforme",
        "findings_count": "{n} constatations",
        "findings_filter_active": "  ·  filtre actif : **{filter}**",
        "findings_click_hint": "  ·  Cliquez sur une ligne pour les détails et les preuves.",
        "findings_none": "✅ Aucune constatation pour le filtre sélectionné.",
        "findings_select_hint": "👆 Sélectionnez une ligne pour voir les détails, les preuves et le texte de contestation.",
        "findings_detail_title": "#### 🔎 Détail — Constatation #{n}",
        "col_type": "Type",
        "col_status": "Statut",
        "col_motivation": "Motivation",
        "col_impact": "Impact",
        "detail_financial_impact": "💰 Impact financier",
        "detail_rule_prefix": "📜 **Règle :**",
        "detail_no_rule": "Aucune référence réglementaire explicite.",
        "detail_evidence_title": "##### 📎 Preuves comparatives",
        "detail_invoice_label": "🧾 Facture",
        "detail_contract_label": "📄 Contrat",
        "detail_field_label": "Champ",
        "detail_page_label": "Page",
        "detail_confidence_label": "Confiance",
        "detail_no_invoice_ev": "Aucune preuve de la facture.",
        "detail_no_contract_ev": "Aucune preuve du contrat.",
        "detail_no_evidence": "⚠️ Aucune preuve documentaire jointe.",
        "detail_contestation_title": "##### 📋 Texte de contestation prêt à envoyer",
        "detail_contestation_caption": "Copiez le texte ci-dessus et envoyez-le directement au fournisseur pour demander des éclaircissements.",
        "banner_anomaly_one": "🔴 <b>1 anomalie détectée</b>",
        "banner_anomaly_many": "🔴 <b>{n} anomalies détectées</b>",
        "banner_anomaly_suffix": " &nbsp;—&nbsp; Révision ou contestation immédiate requise pour les valeurs indiquées.",
        "banner_impact": " &nbsp;·&nbsp; Impact potentiel : <b>{amount}</b>",
        "banner_warning_one": "🟡 <b>1 élément à réviser</b>",
        "banner_warning_many": "🟡 <b>{n} éléments à réviser</b>",
        "banner_warning_suffix": " &nbsp;—&nbsp; Certains éléments nécessitent une vérification manuelle avant approbation.",
        "banner_ok": "✅ <b>Tout conforme</b> &nbsp;—&nbsp; Aucune irrégularité ou divergence financière détectée.",
        "chat_title": "#### 💬 Questions sur les documents",
        "chat_caption": "Posez des questions sur les documents chargés — le moteur répond exclusivement sur la base des données extraites.",
        "chat_context_label": "📂 Contexte disponible :",
        "chat_placeholder": "Exemple : Quelle est la date d'expiration du contrat ? / Y a-t-il des heures supplémentaires ?",
        "chat_spinner": "Traitement…",
        "main_title": "🔍 AI Audit Engine IT/CH",
        "main_caption": "Audit automatisé · Contrats & Factures · 🇮🇹 Italie (EUR) · 🇨🇭 Suisse (CHF)",
        "main_docs_expander": "📁 Documents analysés et sources",
        "main_no_invoice": "**Aucune facture détectée.** Chargez au moins une facture pour lancer l'analyse des anomalies. Vous pouvez déjà explorer le contrat et interroger les documents dans l'Assistant.",
        "progress_init": "Initialisation…",
        "progress_vision": "Analyse Vision AI : {name}…",
        "progress_complete": "Extraction terminée.",
        "progress_audit": "Exécution de l'audit…",
        "toast_complete": "Audit terminé !",
        "error_audit_failed": "**Audit non terminé :** {exc}",
        "error_no_readable": "Aucun document lisible. Vérifiez que les fichiers sont valides (PDF non protégés, ou images JPEG/PNG/WebP lisibles).",
        "error_config": "**Configuration invalide :** {exc}",
        "contestation_subject": "Objet : Contestation d'anomalie — {type_label}",
        "contestation_greeting": "Cher fournisseur,",
        "contestation_body": "Suite à l'audit documentaire, l'anomalie suivante a été identifiée{impact} :",
        "contestation_rule_prefix": "\n\nRéférence contractuelle : {ref}",
        "contestation_closing": "Nous demandons des éclaircissements et, le cas échéant, l'émission d'un avoir.",
        "contestation_salutation": "Cordialement",
        "contestation_impact_suffix": " avec un impact estimé de {amount}",
        "tab_identikit": "📋 Identikit Contrat",
        "tab_findings":  "🔍 Constatations",
        "tab_chat":      "💬 Assistant IA",
    },
    "de": {
        "sidebar_title": "## 🔍 AI Audit Engine",
        "sidebar_caption": "Automatisiertes Audit · Verträge & Rechnungen · IT / CH",
        "new_audit_btn": "🔄 Neues Audit",
        "legend_title": "### 📖 Badge-Legende",
        "legend_ok_desc": "Keine Anomalie. Beträge und Klauseln entsprechen dem Vertrag.",
        "legend_warning_desc": "Geringfügige Abweichung oder unsichere Daten. Manuelle Prüfung empfohlen.",
        "legend_anomaly_desc": "Erhebliche Abweichung. Wahrscheinliche Überberechnung oder verletzte Klausel.",
        "sidebar_jurisdictions": "Rechtsordnungen: 🇮🇹 Italien (EUR) · 🇨🇭 Schweiz (CHF)",
        "sidebar_threshold": "Warnung/Anomalie Preisschwelle: 2 %",
        "sidebar_vat": "MwSt. IT: 22 % · MwSt. CH: 8,1 %",
        "welcome_heading": "AI Audit Engine IT/CH",
        "welcome_subtitle": "Laden Sie Ihre PDF-Dokumente für ein sofortiges automatisiertes Audit hoch.<br>Das System erkennt Preis-, Mengen-, Klausel- und MwSt.-Anomalien.",
        "card1_title": "Vertrag",
        "card1_sub": "Bedingungen, Preise, Klauseln, SLA, Daten",
        "card2_title": "Rechnung",
        "card2_sub": "Positionen, MwSt., Summen, Lieferant",
        "card3_title": "Lieferschein",
        "card3_optional": "optional",
        "card3_sub": "Gelieferte Mengen, Vergleich mit Rechnung",
        "welcome_info": "**Erste Schritte:** Ziehen Sie Dateien in das Upload-Panel oben oder klicken Sie auf **Browse files**. Unterstützte Formate: PDF (auch gescannte), JPEG, PNG und WebP. Das System analysiert jedes Dokument über Vision AI und startet das Audit in Sekunden.",
        "upload_label": "Dokumente hochladen (Vertrag, Rechnung, Lieferschein) — PDF oder Bild",
        "upload_help": "Unterstützt: PDF (auch gescannte), JPEG, PNG, WebP. Das System klassifiziert jedes Dokument automatisch.",
        "upload_invalid": "Datei '{name}' ist ungültig oder leer — sie wird ignoriert.",
        "upload_n_single": "{n} Datei ausgewählt:",
        "upload_n_multi": "{n} Dateien ausgewählt:",
        "run_audit_btn": "▶ Audit starten",
        "docs_title": "#### 📁 Analysierte Dokumente",
        "vendor_label": "Lieferant",
        "client_label": "Kunde",
        "date_label": "Datum",
        "jurisdiction_label": "Rechtsordnung",
        "currency_label": "Währung",
        "extraction_conf": "Extraktionszuverlässigkeit",
        "key_points_title": "**Wichtige Punkte**",
        "truncated_doc": "Erweitertes Dokument: Analyse fokussiert auf wirtschaftliche und technische Klauseln",
        "kp_invoice_total": "💰 Rechnungsgesamt",
        "kp_vat": "🏛️ MwSt.",
        "kp_line_items": "📋 Positionen",
        "kp_items_fmt": "{n} Positionen: {sample}{suffix}",
        "kp_more": "… +{n} weitere",
        "kp_duration": "📅 Laufzeit",
        "kp_start": "📅 Beginn",
        "kp_main_clause": "📜 Hauptklausel",
        "kp_contract_prices": "💶 Vertragspreise",
        "kp_price_list": "{n} Preislistenpositionen",
        "kp_qty_delivered": "📦 Gelieferte Menge",
        "kp_articles": "📋 Artikel",
        "kp_articles_fmt": "{n} Positionen: {sample}",
        "kp_delivery_date": "📅 Lieferdatum",
        "kp_detected_vendor": "🏢 Lieferant erkannt",
        "kp_detected_amount": "💰 Betrag erkannt",
        "kp_unknown_type": "⚠️ Typ nicht erkannt",
        "kp_manual_review": "Manuelle Prüfung empfohlen",
        "doc_CONTRACT": "Vertrag",
        "doc_INVOICE": "Rechnung",
        "doc_DDT": "Lieferschein",
        "doc_UNKNOWN": "Unbekanntes Dokument",
        "type_QUANTITY": "📦 Menge",
        "type_PRICING": "💶 Preise",
        "type_CONTRACTUAL": "📋 Vertraglich",
        "type_FISCAL": "🏛️ Fiskalisch",
        "type_SLA": "⏱️ SLA",
        "type_OTHER": "❓ Sonstige",
        "identikit_vendor": "🏢 Lieferant",
        "identikit_client": "👤 Kunde",
        "identikit_jurisdiction": "⚖️ Rechtsordnung",
        "identikit_contract_status": "#### ⏱️ Vertragsstatus",
        "identikit_start_date": "📅 Startdatum",
        "identikit_end_date": "📅 Enddatum",
        "identikit_days_remaining": "⏳ Verbleibende Tage",
        "identikit_elapsed_delta": "{pct:.0%} verstrichen",
        "identikit_progress_text": "Vertragslaufzeit: {pct:.0%} verstrichen — {days} Tage verbleibend",
        "identikit_no_dates": "Vertragsdaten nicht erkannt.",
        "identikit_included": "#### ✅ Enthaltene Leistungen",
        "identikit_exclusions": "#### 🚫 Ausschlüsse / Kritische Klauseln",
        "identikit_per_unit": "/ Einheit",
        "identikit_no_services": "Keine Leistungspositionen aus dem Vertrag extrahiert.",
        "identikit_no_exclusions": "Keine Ausschlussklauseln erkannt.",
        "identikit_all_clauses": "📜 Alle Klauseln ({n})",
        "identikit_no_contract": "Kein Vertrag geladen. Laden Sie einen Vertrag, um den Identikit anzuzeigen.",
        "session_vendor": "**Lieferant**",
        "session_client": "**Kunde**",
        "session_invoice_date": "**Rechnungsdatum**",
        "session_invoice_total": "**Rechnungsgesamt**",
        "metric_savings": "💰 Potenzielle Einsparungen",
        "metric_savings_help_none": "Keine finanzielle Anomalie erkannt",
        "metric_savings_help_found": "Summe der erkannten finanziellen Auswirkungen",
        "metric_anomalies": "🔴 Anomalien",
        "metric_warnings": "🟡 Warnungen",
        "metric_compliant": "🟢 Konform",
        "audit_summary_title": "#### 📊 Audit-Zusammenfassung",
        "findings_title": "#### 📋 Befunde",
        "filter_label": "Befunde filtern",
        "filter_all": "Alle",
        "filter_anomaly": "🔴 Anomalie",
        "filter_review": "🟡 Prüfen",
        "filter_ok": "🟢 Konform",
        "findings_count": "{n} Befunde",
        "findings_filter_active": "  ·  aktiver Filter: **{filter}**",
        "findings_click_hint": "  ·  Klicken Sie auf eine Zeile für Details und Beweise.",
        "findings_none": "✅ Keine Befunde für den gewählten Filter.",
        "findings_select_hint": "👆 Wählen Sie eine Zeile, um Details, Beweise und den Anfechtungstext zu sehen.",
        "findings_detail_title": "#### 🔎 Detail — Befund #{n}",
        "col_type": "Typ",
        "col_status": "Status",
        "col_motivation": "Begründung",
        "col_impact": "Auswirkung",
        "detail_financial_impact": "💰 Finanzielle Auswirkung",
        "detail_rule_prefix": "📜 **Regel:**",
        "detail_no_rule": "Kein expliziter regulatorischer Bezug.",
        "detail_evidence_title": "##### 📎 Beweise im Vergleich",
        "detail_invoice_label": "🧾 Rechnung",
        "detail_contract_label": "📄 Vertrag",
        "detail_field_label": "Feld",
        "detail_page_label": "Seite",
        "detail_confidence_label": "Zuverlässigkeit",
        "detail_no_invoice_ev": "Keine Beweise aus der Rechnung.",
        "detail_no_contract_ev": "Keine Beweise aus dem Vertrag.",
        "detail_no_evidence": "⚠️ Keine Dokumentenbeweise beigefügt.",
        "detail_contestation_title": "##### 📋 Versandfertiger Anfechtungstext",
        "detail_contestation_caption": "Kopieren Sie den obigen Text und senden Sie ihn direkt an den Lieferanten, um Klärung zu verlangen.",
        "banner_anomaly_one": "🔴 <b>1 Anomalie erkannt</b>",
        "banner_anomaly_many": "🔴 <b>{n} Anomalien erkannt</b>",
        "banner_anomaly_suffix": " &nbsp;—&nbsp; Sofortige Prüfung oder Anfechtung für die angegebenen Werte erforderlich.",
        "banner_impact": " &nbsp;·&nbsp; Potenzielle Auswirkung: <b>{amount}</b>",
        "banner_warning_one": "🟡 <b>1 zu prüfender Posten</b>",
        "banner_warning_many": "🟡 <b>{n} zu prüfende Posten</b>",
        "banner_warning_suffix": " &nbsp;—&nbsp; Einige Posten erfordern eine manuelle Prüfung vor der Genehmigung.",
        "banner_ok": "✅ <b>Alles konform</b> &nbsp;—&nbsp; Keine Unregelmäßigkeiten oder finanzielle Abweichungen erkannt.",
        "chat_title": "#### 💬 Fragen zu den Dokumenten",
        "chat_caption": "Stellen Sie Fragen zu den hochgeladenen Dokumenten — das System antwortet ausschließlich auf der Grundlage der extrahierten Daten.",
        "chat_context_label": "📂 Verfügbarer Kontext:",
        "chat_placeholder": "Beispiel: Wann läuft der Vertrag ab? / Gibt es Überstunden?",
        "chat_spinner": "Verarbeitung…",
        "main_title": "🔍 AI Audit Engine IT/CH",
        "main_caption": "Automatisiertes Audit · Verträge & Rechnungen · 🇮🇹 Italien (EUR) · 🇨🇭 Schweiz (CHF)",
        "main_docs_expander": "📁 Analysierte Dokumente und Quellen",
        "main_no_invoice": "**Keine Rechnung erkannt.** Laden Sie mindestens eine Rechnung hoch, um die Anomalie-Analyse zu starten. Sie können bereits den Vertrag erkunden und Dokumente im Assistenten abfragen.",
        "progress_init": "Initialisierung…",
        "progress_vision": "Vision-KI-Analyse: {name}…",
        "progress_complete": "Extraktion abgeschlossen.",
        "progress_audit": "Audit wird ausgeführt…",
        "toast_complete": "Audit abgeschlossen!",
        "error_audit_failed": "**Audit nicht abgeschlossen:** {exc}",
        "error_no_readable": "Kein lesbares Dokument. Stellen Sie sicher, dass die Dateien gültig sind (nicht passwortgeschützte PDFs oder lesbare JPEG/PNG/WebP-Bilder).",
        "error_config": "**Ungültige Konfiguration:** {exc}",
        "contestation_subject": "Betreff: Anomalie-Anfechtung — {type_label}",
        "contestation_greeting": "Sehr geehrter Lieferant,",
        "contestation_body": "Im Rahmen der Dokumentenprüfung wurde folgende Anomalie festgestellt{impact}:",
        "contestation_rule_prefix": "\n\nVertraglicher Bezug: {ref}",
        "contestation_closing": "Wir bitten um Klärung und gegebenenfalls um die Ausstellung einer Gutschrift.",
        "contestation_salutation": "Mit freundlichen Grüßen",
        "contestation_impact_suffix": " mit einer geschätzten Auswirkung von {amount}",
        "tab_identikit": "📋 Vertrags-Identikit",
        "tab_findings":  "🔍 Audit-Befunde",
        "tab_chat":      "💬 KI-Assistent",
    },
}


def _get_lang() -> str:
    return st.session_state.get("ui_lang", "it")


def _t(key: str) -> str:
    lang = _get_lang()
    labels = UI_LABELS.get(lang, UI_LABELS["it"])
    return labels.get(key, UI_LABELS["it"].get(key, key))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_money(m: Optional[Money]) -> str:
    return f"{m.currency} {m.amount:,.2f}" if m else "—"


def _truncate(text: str, n: int = 115) -> str:
    return text if len(text) <= n else text[:n] + "…"


def _translate_text_if_needed(text: str, target_lang: str) -> str:
    """Translate dynamic text to target_lang via Haiku. Results cached per session."""
    if not text or not text.strip() or target_lang == "it":
        return text
    cache = st.session_state.setdefault("_tr_cache", {})
    cache_key = (hash(text), target_lang)
    if cache_key in cache:
        return cache[cache_key]
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return text
    lang_name = {"en": "English", "fr": "French", "de": "German"}.get(target_lang, "English")
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": (
                    f"Translate the following text to {lang_name}. "
                    f"If it is already in {lang_name}, return it unchanged. "
                    "Return ONLY the translated text, no explanations or preamble.\n\n"
                    f"{text}"
                ),
            }],
        )
        result = resp.content[0].text.strip()
    except Exception:
        result = text
    cache[cache_key] = result
    return result


def _doc_context(result: AuditResult) -> str:
    """Serialize all documents + findings to plain text for the chat LLM."""
    parts: list[str] = []
    for doc in result.documents:
        icon = _DOC_ICON.get(doc.doc_type, "📄")
        doc_label = UI_LABELS["it"].get(f"doc_{doc.doc_type}", doc.doc_type)
        parts.append(
            f"=== {icon} {doc_label}"
            f": {doc.meta.source_file or 'N/A'} ==="
        )
        parts.append(f"Fornitore: {doc.vendor.value} (confidenza {doc.vendor.confidence:.0%})")
        parts.append(f"Cliente: {doc.client.value}")
        parts.append(f"Giurisdizione: {doc.meta.jurisdiction}  Valuta: {doc.currency.value}")
        if doc.meta.issue_date:
            parts.append(f"Data documento: {doc.meta.issue_date.strftime('%d/%m/%Y')}")
        if doc.start_date:
            parts.append(f"Data inizio contratto: {doc.start_date.value}")
        if doc.end_date:
            parts.append(f"Data fine contratto: {doc.end_date.value}")
        if doc.subtotal:
            parts.append(f"Imponibile: {_fmt_money(doc.subtotal)}")
        if doc.vat_rate:
            parts.append(f"Aliquota IVA: {doc.vat_rate.value}%")
        if doc.vat_amount:
            parts.append(f"IVA: {_fmt_money(doc.vat_amount)}")
        if doc.total:
            parts.append(f"Totale: {_fmt_money(doc.total)}")
        for item in doc.line_items:
            parts.append(
                f"  Voce: {item.description} — "
                f"qty {item.quantity} × {_fmt_money(item.unit_price)} = {_fmt_money(item.total)}"
            )
        for clause in doc.clauses:
            parts.append(f"  Clausola [{clause.title}]: {clause.text}")
        parts.append("")

    if result.findings:
        parts.append("=== RISULTATI AUDIT ===")
        for f in result.findings:
            type_label = UI_LABELS["it"].get(f"type_{f.finding_type}", f.finding_type)
            impact = f" | Impatto: {_fmt_money(f.impact_value)}" if f.impact_value else ""
            parts.append(
                f"[{f.status}] {type_label}: "
                f"{f.motivation}{impact}"
            )
        parts.append("")

    return "\n".join(parts)


def _build_contestation_text(finding: Finding) -> str:
    lang = _get_lang()
    type_label = _t(f"type_{finding.finding_type}")

    impact = (
        _t("contestation_impact_suffix").format(amount=_fmt_money(finding.impact_value))
        if finding.impact_value else ""
    )
    rule = (
        _t("contestation_rule_prefix").format(ref=finding.rule_reference)
        if finding.rule_reference else ""
    )
    motivation = _translate_text_if_needed(finding.motivation, lang)

    return (
        f"{_t('contestation_subject').format(type_label=type_label)}\n\n"
        f"{_t('contestation_greeting')}\n\n"
        f"{_t('contestation_body').format(impact=impact)}\n\n"
        f"{motivation}{rule}\n\n"
        f"{_t('contestation_closing')}\n\n"
        f"{_t('contestation_salutation')}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        selected_lang = st.selectbox(
            "🌐 UI Language",
            _LANG_OPTIONS,
            index=_LANG_OPTIONS.index("Italiano"),
            key="ui_lang_select",
        )
        st.session_state["ui_lang"] = _LANG_CODE[selected_lang]

        st.markdown(_t("sidebar_title"))
        st.caption(_t("sidebar_caption"))
        st.divider()

        if "audit_result" in st.session_state:
            st.button(_t("new_audit_btn"), use_container_width=True, on_click=_reset_session)
            st.divider()

        _render_legend()
        st.divider()
        st.caption(_t("sidebar_jurisdictions"))
        st.caption(_t("sidebar_threshold"))
        st.caption(_t("sidebar_vat"))


def _reset_session() -> None:
    for key in ("audit_result", "chat_history", "_tr_cache"):
        st.session_state.pop(key, None)


def _render_legend() -> None:
    st.markdown(_t("legend_title"))
    descriptions = {
        "OK":      _t("legend_ok_desc"),
        "Warning": _t("legend_warning_desc"),
        "Anomaly": _t("legend_anomaly_desc"),
    }
    for status in ("OK", "Warning", "Anomaly"):
        st.markdown(
            f'<div style="background:{_BG[status]};color:{_FG[status]};'
            f'padding:8px 12px;border-radius:6px;margin-bottom:6px;'
            f'border-left:4px solid {_BORDER[status]}">'
            f"<b>{_ICON[status]} {status}</b><br>"
            f'<span style="font-size:0.85em">{descriptions[status]}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Welcome screen
# ─────────────────────────────────────────────────────────────────────────────

def render_welcome() -> None:
    st.markdown(
        f"""
        <div style="text-align:center;padding:32px 20px 12px">
            <div style="font-size:3.2em">🔍</div>
            <h2 style="margin-top:6px;margin-bottom:4px">{_t("welcome_heading")}</h2>
            <p style="color:#666;max-width:560px;margin:0 auto 24px;font-size:1.05em;line-height:1.6">
                {_t("welcome_subtitle")}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    optional_tag = f"<small style='color:#aaa'>({_t('card3_optional')})</small>"
    cards = [
        ("📄", _t("card1_title"), _t("card1_sub")),
        ("🧾", _t("card2_title"), _t("card2_sub")),
        ("📦", f"{_t('card3_title')} {optional_tag}", _t("card3_sub")),
    ]
    for col, (icon, title, sub) in zip((c1, c2, c3), cards):
        with col:
            st.markdown(
                f'<div style="background:#f8f9fa;border-radius:10px;padding:20px;'
                f'text-align:center;border:1px solid #dee2e6;height:110px">'
                f'<div style="font-size:2em">{icon}</div>'
                f'<b>{title}</b><br>'
                f'<span style="color:#666;font-size:0.88em">{sub}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(_t("welcome_info"), icon="💡")


# ─────────────────────────────────────────────────────────────────────────────
# Upload panel
# ─────────────────────────────────────────────────────────────────────────────

def render_upload_panel() -> None:
    uploaded_files = st.file_uploader(
        _t("upload_label"),
        type=["pdf", "jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="up_files",
        help=_t("upload_help"),
    )

    if not uploaded_files:
        render_welcome()
        return

    n = len(uploaded_files)
    count_key = "upload_n_single" if n == 1 else "upload_n_multi"
    st.caption(
        _t(count_key).format(n=n)
        + " "
        + ", ".join(f.name for f in uploaded_files)
    )

    valid_files = [f for f in uploaded_files if f.size > 0]
    invalid_files = [f for f in uploaded_files if f.size == 0]
    for bad in invalid_files:
        st.error(_t("upload_invalid").format(name=bad.name))

    if st.button(_t("run_audit_btn"), type="primary"):
        _run_audit(valid_files)


# ─────────────────────────────────────────────────────────────────────────────
# Extraction + audit pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _run_audit(uploaded_files) -> None:
    st.session_state.pop("audit_result", None)
    st.session_state.pop("chat_history", None)

    try:
        extractor = UniversalExtractor()
    except AuditError as exc:
        st.error(_t("error_config").format(exc=exc))
        return

    documents: list[DocumentFact] = []
    total = len(uploaded_files)
    progress = st.progress(0, text=_t("progress_init"))

    for i, f in enumerate(uploaded_files):
        progress.progress(i / total, text=_t("progress_vision").format(name=f.name))
        try:
            doc = extractor.extract(f.read(), f.name)
            documents.append(doc)
        except AuditError as exc:
            st.warning(f"⚠️ File ignorato — {f.name}: {exc}")

    progress.progress(1.0, text=_t("progress_complete"))

    if not documents:
        st.error(_t("error_no_readable"))
        return

    contracts = [d for d in documents if d.doc_type == "CONTRACT"]
    invoices  = [d for d in documents if d.doc_type == "INVOICE"]
    ddts      = [d for d in documents if d.doc_type == "DDT"]

    anchor = invoices[0] if invoices else documents[0]
    session = AuditResult(
        name=_session_name(anchor, contracts[0] if contracts else None)
    )
    for doc in documents:
        session.add_document(doc)

    if contracts and invoices:
        contract = contracts[0]
        ddt = ddts[0] if ddts else None
        try:
            progress.progress(0.85, text=_t("progress_audit"))
            for invoice in invoices:
                for finding in AuditEngine().cross_check(invoice, contract, ddt=ddt):
                    session.add_finding(finding)
        except AuditError as exc:
            st.error(_t("error_audit_failed").format(exc=exc))
            return

    session.status = "COMPLETE"
    st.session_state.audit_result = session
    st.toast(_t("toast_complete"), icon="✅")
    st.rerun()


def _session_name(primary: DocumentFact, contract: Optional[DocumentFact]) -> str:
    vendor = primary.vendor.value or (contract.vendor.value if contract else "N/A")
    ref = primary.meta.document_number or primary.meta.source_file or "—"
    return f"{vendor} — {ref}"


# ─────────────────────────────────────────────────────────────────────────────
# Document Intelligence Panel
# ─────────────────────────────────────────────────────────────────────────────

def _key_points(doc: DocumentFact) -> list[tuple[str, str]]:
    pts: list[tuple[str, str]] = []

    if doc.doc_type == "INVOICE":
        if doc.total:
            pts.append((_t("kp_invoice_total"), _fmt_money(doc.total)))
        if doc.vat_rate and doc.vat_amount:
            pts.append((_t("kp_vat"), f"{doc.vat_rate.value}% → {_fmt_money(doc.vat_amount)}"))
        if doc.line_items:
            n = len(doc.line_items)
            sample = ", ".join(li.description for li in doc.line_items[:2])
            suffix = _t("kp_more").format(n=n - 2) if n > 2 else ""
            pts.append((_t("kp_line_items"), _t("kp_items_fmt").format(n=n, sample=sample, suffix=suffix)))

    elif doc.doc_type == "CONTRACT":
        if doc.start_date and doc.end_date:
            pts.append((_t("kp_duration"), f"{doc.start_date.value} → {doc.end_date.value}"))
        elif doc.start_date:
            pts.append((_t("kp_start"), str(doc.start_date.value)))
        if doc.clauses:
            c = doc.clauses[0]
            pts.append((_t("kp_main_clause"), f"{c.title}: {_truncate(c.text, 75)}"))
        if doc.line_items:
            pts.append((_t("kp_contract_prices"), _t("kp_price_list").format(n=len(doc.line_items))))

    elif doc.doc_type == "DDT":
        if doc.line_items:
            total_qty = sum(li.quantity for li in doc.line_items)
            pts.append((_t("kp_qty_delivered"), str(total_qty)))
            sample = ", ".join(li.description for li in doc.line_items[:2])
            pts.append((_t("kp_articles"), _t("kp_articles_fmt").format(n=len(doc.line_items), sample=sample)))
        if doc.meta.issue_date:
            pts.append((_t("kp_delivery_date"), doc.meta.issue_date.strftime("%d/%m/%Y")))

    else:
        if doc.vendor.value:
            pts.append((_t("kp_detected_vendor"), doc.vendor.value))
        if doc.total:
            pts.append((_t("kp_detected_amount"), _fmt_money(doc.total)))
        pts.append((_t("kp_unknown_type"), _t("kp_manual_review")))

    return pts[:3]


def render_document_panel(result: AuditResult) -> None:
    if not result.documents:
        return

    st.markdown(_t("docs_title"))
    tab_labels = [
        f"{_DOC_ICON.get(d.doc_type, '📄')} {d.meta.source_file or d.doc_type}"
        for d in result.documents
    ]
    tabs = st.tabs(tab_labels)

    for tab, doc in zip(tabs, result.documents):
        with tab:
            col_meta, col_pts = st.columns([1, 1])

            with col_meta:
                bg, fg = _DOC_COLOR.get(doc.doc_type, ("#f8f9fa", "#333"))
                doc_label = _t(f"doc_{doc.doc_type}")
                st.markdown(
                    f'<span style="background:{bg};color:{fg};padding:4px 12px;'
                    f'border-radius:20px;font-size:0.85em;font-weight:600">'
                    f'{_DOC_ICON.get(doc.doc_type)} {doc_label}'
                    f'</span>',
                    unsafe_allow_html=True,
                )
                st.markdown("")
                if doc.vendor.value:
                    st.markdown(f"**{_t('vendor_label')}:** {doc.vendor.value}")
                if doc.client.value:
                    st.markdown(f"**{_t('client_label')}:** {doc.client.value}")
                if doc.meta.issue_date:
                    st.markdown(f"**{_t('date_label')}:** {doc.meta.issue_date.strftime('%d/%m/%Y')}")
                st.markdown(
                    f"**{_t('jurisdiction_label')}:** {doc.meta.jurisdiction} &nbsp; "
                    f"**{_t('currency_label')}:** {doc.currency.value}",
                    unsafe_allow_html=True,
                )
                conf = doc.meta.extraction_confidence
                conf_color = "#28a745" if conf >= 0.9 else "#ffc107" if conf >= 0.7 else "#dc3545"
                st.markdown(
                    f'<span style="font-size:0.8em;color:{conf_color}">'
                    f'{_t("extraction_conf")}: {conf:.0%}</span>',
                    unsafe_allow_html=True,
                )

            with col_pts:
                st.markdown(_t("key_points_title"))
                for label, value in _key_points(doc):
                    st.markdown(
                        f'<div style="background:#f8f9fa;border-radius:6px;'
                        f'padding:10px 14px;margin-bottom:8px;'
                        f'border-left:3px solid #dee2e6">'
                        f'<span style="font-size:0.78em;color:#666;font-weight:600">'
                        f'{label}</span><br>'
                        f'<span style="font-size:0.95em">{value}</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            if doc.is_truncated:
                st.warning(_t("truncated_doc"), icon="⚠️")


# ─────────────────────────────────────────────────────────────────────────────
# Contract Identikit
# ─────────────────────────────────────────────────────────────────────────────

def render_contract_identikit(result: AuditResult) -> None:
    lang = _get_lang()
    contracts = result.contracts
    if not contracts:
        st.info(_t("identikit_no_contract"), icon="ℹ️")
        return

    contract = contracts[0]

    col_v, col_c, col_j = st.columns(3)
    col_v.metric(_t("identikit_vendor"), contract.vendor.value or "—")
    col_c.metric(_t("identikit_client"), contract.client.value or "—")
    col_j.metric(_t("identikit_jurisdiction"), contract.meta.jurisdiction)

    st.markdown("---")
    if contract.start_date and contract.end_date:
        today = date.today()
        start = contract.start_date.value
        end = contract.end_date.value
        total_days = max((end - start).days, 1)
        elapsed_days = (today - start).days
        elapsed_pct = max(0.0, min(1.0, elapsed_days / total_days))
        days_remaining = max(0, (end - today).days)

        st.markdown(_t("identikit_contract_status"))
        col_s, col_e, col_r = st.columns(3)
        col_s.metric(_t("identikit_start_date"), start.strftime("%d/%m/%Y"))
        col_e.metric(_t("identikit_end_date"), end.strftime("%d/%m/%Y"))
        col_r.metric(
            _t("identikit_days_remaining"),
            days_remaining,
            delta=_t("identikit_elapsed_delta").format(pct=elapsed_pct),
            delta_color="inverse",
        )
        st.progress(
            elapsed_pct,
            text=_t("identikit_progress_text").format(pct=elapsed_pct, days=days_remaining),
        )
    else:
        st.caption(_t("identikit_no_dates"))

    st.markdown("---")
    col_in, col_ex = st.columns(2)

    with col_in:
        st.markdown(_t("identikit_included"))
        if contract.line_items:
            for item in contract.line_items:
                desc = _translate_text_if_needed(item.description, lang)
                price_str = (
                    f" &nbsp;·&nbsp; {_fmt_money(item.unit_price)} {_t('identikit_per_unit')}"
                    if item.unit_price and item.unit_price.amount else ""
                )
                st.markdown(
                    f'<div style="background:#d4edda;color:#155724;padding:8px 14px;'
                    f'border-radius:8px;margin-bottom:6px;border-left:4px solid #28a745">'
                    f'<b>{desc}</b>'
                    f'<span style="font-size:0.83em;opacity:0.85">{price_str}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption(_t("identikit_no_services"))

    with col_ex:
        st.markdown(_t("identikit_exclusions"))
        exclusions = [
            c for c in contract.clauses
            if any(kw in (c.title + " " + c.text).lower() for kw in _EXCLUSION_KWS)
        ]
        if exclusions:
            for clause in exclusions:
                title = _translate_text_if_needed(clause.title, lang)
                text  = _translate_text_if_needed(_truncate(clause.text, 100), lang)
                st.markdown(
                    f'<div style="background:#f8d7da;color:#721c24;padding:8px 14px;'
                    f'border-radius:8px;margin-bottom:6px;border-left:4px solid #dc3545">'
                    f'<b>{title}</b><br>'
                    f'<span style="font-size:0.83em">{text}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption(_t("identikit_no_exclusions"))

    if contract.clauses:
        st.markdown("---")
        with st.expander(_t("identikit_all_clauses").format(n=len(contract.clauses))):
            for clause in contract.clauses:
                title = _translate_text_if_needed(clause.title, lang)
                text  = _translate_text_if_needed(clause.text, lang)
                st.markdown(f"**{title}**")
                st.markdown(text)
                st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Audit results
# ─────────────────────────────────────────────────────────────────────────────

def _render_session_header(session: AuditResult) -> None:
    contract = session.contracts[0] if session.contracts else None
    invoice  = session.invoices[0]  if session.invoices  else None
    st.markdown(f"### 📁 {session.name}")
    if contract or invoice:
        c1, c2, c3, c4 = st.columns(4)
        src = contract or invoice
        c1.markdown(f"{_t('session_vendor')}  \n{src.vendor.value}")
        c2.markdown(f"{_t('session_client')}  \n{src.client.value}")
        if invoice and invoice.meta.issue_date:
            c3.markdown(f"{_t('session_invoice_date')}  \n{invoice.meta.issue_date.strftime('%d/%m/%Y')}")
        if invoice:
            c4.markdown(f"{_t('session_invoice_total')}  \n{_fmt_money(invoice.total)}")


def _render_metrics(session: AuditResult) -> None:
    currency = "EUR"
    if session.invoices and session.invoices[0].subtotal:
        currency = session.invoices[0].subtotal.currency

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        _t("metric_savings"),
        f"{currency} {session.total_impact:,.2f}" if session.total_impact else f"{currency} 0,00",
        help=(
            _t("metric_savings_help_none")
            if not session.total_impact
            else _t("metric_savings_help_found")
        ),
    )
    c2.metric(_t("metric_anomalies"), len(session.anomalies))
    c3.metric(_t("metric_warnings"),  len(session.reviews_needed))
    c4.metric(_t("metric_compliant"), len(session.confirmed_ok))


def _render_findings_table(findings: list[Finding]) -> Optional[int]:
    lang = _get_lang()
    if not findings:
        st.success(_t("findings_none"))
        return None

    rows = [
        {
            "#":                     i + 1,
            _t("col_type"):          _t(f"type_{f.finding_type}"),
            _t("col_status"):        f"{_ICON[f.status]} {f.status}",
            _t("col_motivation"):    _truncate(_translate_text_if_needed(f.motivation, lang), 115),
            _t("col_impact"):        _fmt_money(f.impact_value),
        }
        for i, f in enumerate(findings)
    ]
    df = pd.DataFrame(rows)

    col_type      = _t("col_type")
    col_status    = _t("col_status")
    col_motivation = _t("col_motivation")
    col_impact    = _t("col_impact")

    def _style_row(row: pd.Series) -> list[str]:
        s = findings[int(row.name)].status
        return [f"background-color:{_BG[s]};color:{_FG[s]}"] * len(row)

    event = st.dataframe(
        df.style.apply(_style_row, axis=1),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "#":             st.column_config.NumberColumn("#",              width="small"),
            col_type:        st.column_config.TextColumn(col_type,           width="medium"),
            col_status:      st.column_config.TextColumn(col_status,         width="small"),
            col_motivation:  st.column_config.TextColumn(col_motivation,     width="large"),
            col_impact:      st.column_config.TextColumn(col_impact,         width="small"),
        },
    )
    sel = event.selection.rows if event and event.selection else []
    return sel[0] if sel else None


def _render_finding_detail(finding: Finding) -> None:
    lang = _get_lang()
    bg, fg, border = _BG[finding.status], _FG[finding.status], _BORDER[finding.status]
    type_label = _t(f"type_{finding.finding_type}")
    motivation = _translate_text_if_needed(finding.motivation, lang)

    st.markdown(
        f'<div style="background:{bg};color:{fg};border-radius:8px;'
        f'padding:18px 22px;margin-bottom:14px;border-left:5px solid {border}">'
        f'<div style="font-size:0.82em;font-weight:600;margin-bottom:6px;opacity:0.75">'
        f"{_ICON[finding.status]} {finding.status} &nbsp;·&nbsp; {type_label}</div>"
        f'<div style="font-size:1.05em;line-height:1.65">{motivation}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    col_imp, col_rule = st.columns(2)
    with col_imp:
        st.metric(_t("detail_financial_impact"), _fmt_money(finding.impact_value))
    with col_rule:
        if finding.rule_reference:
            st.info(f"{_t('detail_rule_prefix')} {finding.rule_reference}")
        else:
            st.caption(_t("detail_no_rule"))

    if finding.evidence:
        st.markdown(_t("detail_evidence_title"))
        inv_ev   = [e for e in finding.evidence if e.source_document == "INVOICE"]
        cnt_ev   = [e for e in finding.evidence if e.source_document == "CONTRACT"]
        other_ev = [e for e in finding.evidence if e.source_document not in ("INVOICE", "CONTRACT")]

        col_inv, col_cnt = st.columns(2)

        with col_inv:
            st.markdown(
                f'<div style="background:#cce5ff;color:#004085;padding:7px 14px;'
                f'border-radius:6px;font-weight:600;margin-bottom:10px">'
                f'{_t("detail_invoice_label")}</div>',
                unsafe_allow_html=True,
            )
            if inv_ev:
                for ev in inv_ev:
                    if ev.extracted_text:
                        st.code(_translate_text_if_needed(ev.extracted_text, lang), language=None)
                    meta_parts = []
                    if ev.invoice_field:
                        meta_parts.append(f"{_t('detail_field_label')}: `{ev.invoice_field}`")
                    if ev.page_number:
                        meta_parts.append(f"{_t('detail_page_label')} {ev.page_number}")
                    meta_parts.append(f"{_t('detail_confidence_label')} {ev.confidence:.0%}")
                    st.caption("  ·  ".join(meta_parts))
            else:
                st.caption(_t("detail_no_invoice_ev"))

        with col_cnt:
            st.markdown(
                f'<div style="background:#d4edda;color:#155724;padding:7px 14px;'
                f'border-radius:6px;font-weight:600;margin-bottom:10px">'
                f'{_t("detail_contract_label")}</div>',
                unsafe_allow_html=True,
            )
            if cnt_ev:
                for ev in cnt_ev:
                    if ev.extracted_text:
                        st.code(_translate_text_if_needed(ev.extracted_text, lang), language=None)
                    if ev.matched_clause:
                        clause_tr = _translate_text_if_needed(ev.matched_clause, lang)
                        st.caption(f"Clausola: {clause_tr}")
                    meta_parts = []
                    if ev.page_number:
                        meta_parts.append(f"{_t('detail_page_label')} {ev.page_number}")
                    meta_parts.append(f"{_t('detail_confidence_label')} {ev.confidence:.0%}")
                    st.caption("  ·  ".join(meta_parts))
            else:
                st.caption(_t("detail_no_contract_ev"))

        for ev in other_ev:
            with st.expander(f"📦 {ev.source_document}"):
                if ev.extracted_text:
                    st.code(_translate_text_if_needed(ev.extracted_text, lang), language=None)
                if ev.page_number:
                    st.caption(f"{_t('detail_page_label')} {ev.page_number} · {_t('detail_confidence_label')} {ev.confidence:.0%}")
    else:
        st.caption(_t("detail_no_evidence"))

    st.markdown("---")
    st.markdown(_t("detail_contestation_title"))
    st.code(_build_contestation_text(finding), language=None)
    st.caption(_t("detail_contestation_caption"))


def _render_impact_banner(session: AuditResult) -> None:
    if session.anomalies:
        n = len(session.anomalies)
        anomaly_text = _t("banner_anomaly_one") if n == 1 else _t("banner_anomaly_many").format(n=n)
        currency = session.invoices[0].subtotal.currency if session.invoices and session.invoices[0].subtotal else "EUR"
        impact_str = (
            _t("banner_impact").format(amount=f"{currency} {session.total_impact:,.2f}")
            if session.total_impact else ""
        )
        st.markdown(
            f'<div style="background:#f8d7da;color:#721c24;border-radius:8px;'
            f'padding:14px 20px;margin-bottom:4px;border-left:6px solid #dc3545;font-size:1.0em">'
            f'{anomaly_text}{impact_str}{_t("banner_anomaly_suffix")}'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif session.reviews_needed:
        n = len(session.reviews_needed)
        warning_text = _t("banner_warning_one") if n == 1 else _t("banner_warning_many").format(n=n)
        st.markdown(
            f'<div style="background:#fff3cd;color:#856404;border-radius:8px;'
            f'padding:14px 20px;margin-bottom:4px;border-left:6px solid #ffc107;font-size:1.0em">'
            f'{warning_text}{_t("banner_warning_suffix")}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="background:#d4edda;color:#155724;border-radius:8px;'
            f'padding:14px 20px;margin-bottom:4px;border-left:6px solid #28a745;font-size:1.0em">'
            f'{_t("banner_ok")}'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_audit_results(session: AuditResult) -> None:
    _render_session_header(session)
    st.markdown("---")
    st.markdown(_t("audit_summary_title"))
    _render_metrics(session)
    st.markdown("")
    _render_impact_banner(session)
    st.markdown("---")
    st.markdown(_t("findings_title"))

    filter_all     = _t("filter_all")
    filter_anomaly = _t("filter_anomaly")
    filter_review  = _t("filter_review")
    filter_ok      = _t("filter_ok")
    filter_options = [filter_all, filter_anomaly, filter_review, filter_ok]
    default_filter = filter_anomaly if any(f.status == "Anomaly" for f in session.findings) else filter_all

    filter_val = st.pills(
        _t("filter_label"),
        filter_options,
        default=default_filter,
        key=f"findings_filter_{_get_lang()}",
    )
    if filter_val is None:
        filter_val = filter_all

    status_map = {filter_anomaly: "Anomaly", filter_review: "Warning", filter_ok: "OK"}
    filtered = (
        session.findings
        if filter_val == filter_all
        else [f for f in session.findings if f.status == status_map.get(filter_val, "")]
    )

    filter_note = _t("findings_filter_active").format(filter=filter_val) if filter_val != filter_all else ""
    st.caption(
        _t("findings_count").format(n=len(filtered))
        + filter_note
        + (_t("findings_click_hint") if filtered else "")
    )

    selected = _render_findings_table(filtered)
    if selected is not None:
        st.markdown("---")
        st.markdown(_t("findings_detail_title").format(n=selected + 1))
        _render_finding_detail(filtered[selected])
    else:
        if filtered:
            st.caption(_t("findings_select_hint"))


# ─────────────────────────────────────────────────────────────────────────────
# Chat Contextual Q&A
# ─────────────────────────────────────────────────────────────────────────────

def _chat_respond(question: str, result: AuditResult) -> str:
    """Ask Sonnet a question, providing only the already-extracted document data."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "⚠️ Chiave API Anthropic non configurata."

    context = _doc_context(result)
    history = st.session_state.get("chat_history", [])

    messages: list[dict] = []
    for i, msg in enumerate(history):
        if msg["role"] == "user" and i == 0:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"=== DATI ESTRATTI DAI DOCUMENTI ===\n\n{context}",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": msg["content"]},
                ],
            })
        else:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"=== DATI ESTRATTI DAI DOCUMENTI ===\n\n{context}",
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": question},
        ],
    })

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=[{
                "type": "text",
                "text": _CHAT_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=messages,
        )
        return response.content[0].text
    except anthropic.APIError as exc:
        return f"⚠️ Errore API: {exc}"


def render_chat(result: AuditResult) -> None:
    with st.container(border=True):
        st.markdown(_t("chat_title"))
        st.caption(_t("chat_caption"))

        doc_names = " ".join(
            f"[{_DOC_ICON.get(d.doc_type, '📄')} {d.meta.source_file or d.doc_type}]"
            for d in result.documents
        )
        st.caption(f"{_t('chat_context_label')} {doc_names}")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input(_t("chat_placeholder")):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner(_t("chat_spinner")):
                    answer = _chat_respond(prompt, result)
                st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    render_sidebar()
    st.title(_t("main_title"))
    st.caption(_t("main_caption"))
    st.divider()

    if "audit_result" in st.session_state:
        result: AuditResult = st.session_state.audit_result

        with st.expander(_t("main_docs_expander"), expanded=False):
            render_document_panel(result)
        st.markdown("---")

        if result.invoices:
            tab_identikit, tab_rilievi, tab_chat = st.tabs([
                _t("tab_identikit"),
                _t("tab_findings"),
                _t("tab_chat"),
            ])
            with tab_identikit:
                render_contract_identikit(result)
            with tab_rilievi:
                render_audit_results(result)
            with tab_chat:
                render_chat(result)
        else:
            tab_identikit, tab_chat = st.tabs([
                _t("tab_identikit"),
                _t("tab_chat"),
            ])
            with tab_identikit:
                render_contract_identikit(result)
            with tab_chat:
                render_chat(result)
            st.info(_t("main_no_invoice"), icon="ℹ️")
    else:
        render_upload_panel()


if __name__ == "__main__":
    main()
