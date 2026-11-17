"""
scripts/generate_comprehensive_tests.py
Generate 8 realistic PDF invoices for the Adaptive Audit Engine IT/CH test suite.

Scenario A – Capitolato Generale FM (Università di Siena, CIG 8821345AF2)
Scenario B – Capitolato Tecnico Opere Civili e Verde (Università di Siena, CIG 7734891BC1)

Each PDF is self-contained: it carries the anomaly in the numbers/text so the
audit engine can detect it without the actual capitolato PDF being present.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Output directory ──────────────────────────────────────────────────────────

OUT_DIR = Path(__file__).parent.parent / "test_docs"
OUT_DIR.mkdir(exist_ok=True)

# ── Styles ────────────────────────────────────────────────────────────────────

_ss = getSampleStyleSheet()

S_NORMAL = _ss["Normal"]
S_TITLE = ParagraphStyle(
    "InvTitle", parent=_ss["Heading1"],
    fontSize=15, spaceAfter=4,
    textColor=colors.HexColor("#1a1a2e"),
)
S_HEADER = ParagraphStyle(
    "InvHeader", parent=_ss["Heading2"],
    fontSize=11, spaceBefore=6, spaceAfter=3,
    textColor=colors.HexColor("#1a1a2e"),
)
S_BODY = ParagraphStyle(
    "InvBody", parent=_ss["Normal"],
    fontSize=9, leading=13,
)
S_SMALL = ParagraphStyle(
    "InvSmall", parent=_ss["Normal"],
    fontSize=8, textColor=colors.HexColor("#444444"),
)
S_NOTE = ParagraphStyle(
    "InvNote", parent=_ss["Normal"],
    fontSize=7.5, textColor=colors.HexColor("#666666"),
    leftIndent=8, spaceBefore=1,
)
S_SUM = ParagraphStyle(
    "InvSum", parent=_ss["Normal"],
    fontSize=8.5, alignment=2,
)

_HDR_BG  = colors.HexColor("#1a1a2e")
_ALT_ROW = colors.HexColor("#f4f6f8")
_BORDER  = colors.HexColor("#c0c0c0")

_TABLE_STYLE = TableStyle([
    # Header row
    ("BACKGROUND",    (0, 0), (-1, 0), _HDR_BG),
    ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
    ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE",      (0, 0), (-1, 0), 8.5),
    ("ALIGN",         (0, 0), (-1, 0), "CENTER"),
    # Data rows
    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, _ALT_ROW]),
    ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE",      (0, 1), (-1, -1), 8.2),
    ("ALIGN",         (3, 1), (-1, -1), "RIGHT"),
    # Grid
    ("GRID",          (0, 0), (-1, -1), 0.4, _BORDER),
    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING",    (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING",   (0, 0), (-1, -1), 5),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
])


# ── Money helpers ─────────────────────────────────────────────────────────────

def q(v: str | float) -> Decimal:
    return Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def eur(v: str | float | Decimal) -> str:
    """Italian-locale formatted euro string."""
    d = q(v)
    # format with dot thousands separator, comma decimal
    formatted = f"{float(d):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"€ {formatted}"


# ── Document builder helpers ──────────────────────────────────────────────────

def _new_doc(filename: str) -> SimpleDocTemplate:
    return SimpleDocTemplate(
        str(OUT_DIR / filename),
        pagesize=A4,
        topMargin=1.8 * cm, bottomMargin=2 * cm,
        leftMargin=2 * cm,  rightMargin=2 * cm,
    )


def _header(
    vendor_name: str, vendor_addr: str, vendor_vat: str,
    client_name: str, client_addr: str,
    inv_num: str, inv_date: str, contract_ref: str,
) -> list:
    """Two-column invoice header block."""
    data = [
        [Paragraph(f"<b>{vendor_name}</b>", S_BODY),
         Paragraph(f"<b>FATTURA N. {inv_num}</b>", S_HEADER)],
        [Paragraph(vendor_addr, S_SMALL),
         Paragraph(f"Data emissione: {inv_date}", S_SMALL)],
        [Paragraph(f"P.IVA: {vendor_vat}", S_SMALL),
         Paragraph(f"Rif. contratto: {contract_ref}", S_SMALL)],
        [Paragraph("", S_SMALL), Paragraph("", S_SMALL)],
        [Paragraph("<b>Spett.le:</b>", S_SMALL), Paragraph("", S_SMALL)],
        [Paragraph(client_name, S_SMALL), Paragraph("", S_SMALL)],
        [Paragraph(client_addr, S_SMALL), Paragraph("", S_SMALL)],
    ]
    t = Table(data, colWidths=[9.5 * cm, 7.5 * cm])
    t.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))
    return [
        t,
        Spacer(1, 0.4 * cm),
        HRFlowable(width="100%", thickness=1.2, color=_HDR_BG),
        Spacer(1, 0.25 * cm),
    ]


def _totals(subtotal: Decimal, vat_rate: Decimal,
            vat_amt: Decimal, total: Decimal) -> Table:
    data = [
        ["", "Imponibile:", eur(subtotal)],
        ["", f"IVA {vat_rate}%:", eur(vat_amt)],
        ["", "TOTALE FATTURA:", eur(total)],
    ]
    t = Table(data, colWidths=[9.5 * cm, 4.5 * cm, 3 * cm])
    t.setStyle(TableStyle([
        ("ALIGN",     (1, 0), (-1, -1), "RIGHT"),
        ("FONTNAME",  (1, 2), (-1, 2),  "Helvetica-Bold"),
        ("FONTSIZE",  (0, 0), (-1, -1), 9),
        ("LINEABOVE", (1, 2), (-1, 2),  0.8, colors.black),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _notes(lines: list[str]) -> list:
    out: list = [
        Spacer(1, 0.35 * cm),
        HRFlowable(width="100%", thickness=0.5, color=_BORDER),
        Spacer(1, 0.15 * cm),
        Paragraph("<b>Note e riferimenti contrattuali:</b>", S_SMALL),
    ]
    for line in lines:
        out.append(Paragraph(f"• {line}", S_NOTE))
    return out


def _items_table(rows: list[list[str]], col_widths: list[float]) -> Table:
    t = Table(rows, colWidths=[w * cm for w in col_widths])
    t.setStyle(_TABLE_STYLE)
    return t


# ── Shared parties ────────────────────────────────────────────────────────────

_CLIENT      = "Università degli Studi di Siena"
_CLIENT_ADDR = "Banchi di Sotto 55, 53100 Siena — C.F. 80002070524"

_VENDOR_GEN      = "GlobalFacility S.r.l."
_VENDOR_GEN_ADDR = "Via della Tecnica 42, 53100 Siena (SI)"
_VENDOR_GEN_VAT  = "IT 04812345678"
_CONTRACT_GEN    = "CIG 8821345AF2 – Cap. Generale FM Ateneo Siena"

_VENDOR_TECH      = "EcoEdil Servizi S.r.l."
_VENDOR_TECH_ADDR = "Via Massetana Romana 18, 53100 Siena (SI)"
_VENDOR_TECH_VAT  = "IT 05567891234"
_CONTRACT_TECH    = "CIG 7734891BC1 – Cap. Tecnico Opere Civili e Verde Ateneo"


# ─────────────────────────────────────────────────────────────────────────────
# Scenario A – Capitolato Generale FM
# ─────────────────────────────────────────────────────────────────────────────

def gen_prezzo_extra() -> None:
    """A1 – Canone mensile gonfiato: €21.200 vs base d'asta €18.333,33/mese."""
    doc = _new_doc("siena_gen_prezzo_extra.pdf")
    el = _header(
        _VENDOR_GEN, _VENDOR_GEN_ADDR, _VENDOR_GEN_VAT,
        _CLIENT, _CLIENT_ADDR,
        "2024/GF-0341", "15 Gennaio 2024", _CONTRACT_GEN,
    )
    el += [
        Paragraph("<b>Oggetto:</b> Servizi Facility Management – Canone Mensile Gennaio 2024", S_BODY),
        Spacer(1, 0.3 * cm),
        _items_table([
            ["Descrizione", "U.M.", "Qtà", "Prezzo unit.", "Totale"],
            ["Canone mensile Facility Management\n(servizi integrati edifici universitari)",
             "mese", "1", eur("21200.00"), eur("21200.00")],
            ["Presidio operativo e coordinamento\n(responsabile di commessa e back-office)",
             "mese", "1", eur("1450.00"),  eur("1450.00")],
            ["Smaltimento rifiuti speciali – canone fisso",
             "mese", "1", eur("480.00"),   eur("480.00")],
        ], [7.8, 1.4, 1.0, 2.8, 3.0]),
        Spacer(1, 0.4 * cm),
    ]
    subtotal = q("23130.00")
    vat_r    = q("22")
    vat_a    = q("5088.60")     # 23130 * 0.22
    total    = q("28218.60")
    el.append(_totals(subtotal, vat_r, vat_a, total))
    el += _notes([
        "Capitolato Generale FM – Art. 5: importo a base d'asta annuo € 220.000,00 → mensile: € 18.333,33.",
        "Canone fatturato: € 21.200,00/mese — eccedenza rispetto alla base contrattuale: € 2.866,67/mese.",
        "La reperibilità H24 e il presidio operativo sono inclusi nel canone base (Art. 7 – Servizi inclusi).",
        "Overbilling stimato mensile: € 2.866,67 + IVA 22% = € 3.497,34.",
    ])
    doc.build(el)
    print("  ✓  siena_gen_prezzo_extra.pdf")


def gen_ritardo_penale() -> None:
    """A2 – Intervento urgente con SLA 60 min violato (165 min). Penale non detratta."""
    doc = _new_doc("siena_gen_ritardo_penale.pdf")
    el = _header(
        _VENDOR_GEN, _VENDOR_GEN_ADDR, _VENDOR_GEN_VAT,
        _CLIENT, _CLIENT_ADDR,
        "2024/GF-0358", "22 Gennaio 2024", _CONTRACT_GEN,
    )
    el += [
        Paragraph("<b>Oggetto:</b> Intervento Urgente – Guasto Impianto Idraulico – Edificio S. Miniato", S_BODY),
        Spacer(1, 0.2 * cm),
        Paragraph(
            "Chiamata di emergenza ricevuta: <b>18/01/2024 ore 09:00</b><br/>"
            "Arrivo tecnico in loco: <b>18/01/2024 ore 11:45</b><br/>"
            "Tempo intercorso dalla chiamata all'arrivo: <b>165 minuti</b>.",
            S_BODY,
        ),
        Spacer(1, 0.3 * cm),
        _items_table([
            ["Descrizione", "U.M.", "Qtà", "Prezzo unit.", "Totale"],
            ["Intervento urgente idraulico – manodopera\n(ripristino tubazione edificio S. Miniato)",
             "ore", "4", eur("48.00"), eur("192.00")],
            ["Indennità trasferta intervento urgente",
             "forfait", "1", eur("85.00"), eur("85.00")],
            ["Materiali: raccorderia, guarnizioni e nastro PTFE",
             "forfait", "1", eur("62.00"), eur("62.00")],
            ["Maggiorazione urgenza H24",
             "forfait", "1", eur("120.00"), eur("120.00")],
        ], [7.8, 1.6, 1.0, 2.6, 3.0]),
        Spacer(1, 0.4 * cm),
    ]
    subtotal = q("459.00")
    vat_r    = q("22")
    vat_a    = q("100.98")
    total    = q("559.98")
    el.append(_totals(subtotal, vat_r, vat_a, total))
    el += _notes([
        "Capitolato Generale FM – Art. 12 SLA Urgenze: tempo massimo di intervento = 60 minuti dalla chiamata.",
        "Intervento 18/01/2024: chiamata ore 09:00 – arrivo ore 11:45 → 165 minuti (SLA superato di 105 min).",
        "Art. 18 Penali: penale pari a € 150,00 per ogni ora o frazione oltre SLA.",
        "Penale applicabile per 2 ore di ritardo: € 300,00 — non detratta dalla presente fattura.",
    ])
    doc.build(el)
    print("  ✓  siena_gen_ritardo_penale.pdf")


def gen_iva_error() -> None:
    """A3 – IVA indicata al 22% ma calcolata al 24% (delta +€88,00)."""
    doc = _new_doc("siena_gen_iva_error.pdf")
    el = _header(
        _VENDOR_GEN, _VENDOR_GEN_ADDR, _VENDOR_GEN_VAT,
        _CLIENT, _CLIENT_ADDR,
        "2024/GF-0372", "31 Gennaio 2024", _CONTRACT_GEN,
    )
    el += [
        Paragraph("<b>Oggetto:</b> Manutenzione Impianti Termici – Gennaio 2024", S_BODY),
        Spacer(1, 0.3 * cm),
        _items_table([
            ["Descrizione", "U.M.", "Qtà", "Prezzo unit.", "Totale"],
            ["Manutenzione programmata caldaie – 12 unità\n(campus universitario, revisione annuale)",
             "interv.", "12", eur("185.00"), eur("2220.00")],
            ["Sostituzione filtri aria UTA",
             "pz", "24", eur("42.00"), eur("1008.00")],
            ["Verifica e taratura valvole termostatiche",
             "pz", "86", eur("8.50"), eur("731.00")],
            ["Controllo e ricarica refrigerante – split (3,5 kg)",
             "forfait", "1", eur("297.50"), eur("297.50")],
            ["Redazione verbali di conformità impianti",
             "forfait", "1", eur("143.50"), eur("143.50")],
        ], [7.8, 1.4, 1.0, 2.8, 3.0]),
        Spacer(1, 0.4 * cm),
    ]
    subtotal       = q("4400.00")
    vat_stated     = q("22")
    vat_amt_wrong  = q("1056.00")   # errore: corrisponde al 24%, non al 22%
    total_wrong    = q("5456.00")
    el.append(_totals(subtotal, vat_stated, vat_amt_wrong, total_wrong))
    el += _notes([
        "Aliquota IVA indicata in fattura: 22% (corretta per servizi di manutenzione ordinaria).",
        "IVA applicata: € 1.056,00 — corrispondente al 24% di € 4.400,00, non al 22%.",
        "IVA corretta al 22%: € 4.400,00 × 22% = € 968,00.",
        "Differenza IVA a sfavore dell'Università: € 88,00.",
        "Totale corretto: € 4.400,00 + € 968,00 = € 5.368,00 (fatturato: € 5.456,00).",
    ])
    doc.build(el)
    print("  ✓  siena_gen_iva_error.pdf")


def gen_doppia_voce() -> None:
    """A4 – Reperibilità H24 inclusa nel canone (Art. 7) fatturata separatamente: +€1.200."""
    doc = _new_doc("siena_gen_doppia_voce.pdf")
    el = _header(
        _VENDOR_GEN, _VENDOR_GEN_ADDR, _VENDOR_GEN_VAT,
        _CLIENT, _CLIENT_ADDR,
        "2024/GF-0389", "15 Febbraio 2024", _CONTRACT_GEN,
    )
    el += [
        Paragraph("<b>Oggetto:</b> Canone Mensile FM + Servizi Supplementari – Febbraio 2024", S_BODY),
        Spacer(1, 0.3 * cm),
        _items_table([
            ["Descrizione", "U.M.", "Qtà", "Prezzo unit.", "Totale"],
            ["Canone mensile Facility Management\n(servizi integrati – Art. 5 capitolato)",
             "mese", "1", eur("18333.33"), eur("18333.33")],
            ["Gestione Reperibilità H24\n(presidio tecnico operativo notturno e festivo)",
             "mese", "1", eur("1200.00"),  eur("1200.00")],
            ["Coordinamento operativo e reporting mensile\n(report KPI e verbali interventi)",
             "mese", "1", eur("350.00"),   eur("350.00")],
            ["Interventi programmati extra-canone – n. 3\n(verbali allegati: 2024/02/V-01, V-02, V-03)",
             "interv.", "3", eur("210.00"),   eur("630.00")],
        ], [7.8, 1.4, 1.0, 2.8, 3.0]),
        Spacer(1, 0.4 * cm),
    ]
    subtotal = q("20513.33")
    vat_r    = q("22")
    vat_a    = q("4512.93")
    total    = q("25026.26")
    el.append(_totals(subtotal, vat_r, vat_a, total))
    el += _notes([
        "Capitolato Generale FM – Art. 7 'Servizi inclusi nel canone': la reperibilità H24,",
        "il presidio tecnico operativo notturno e festivo sono già compresi nel canone mensile.",
        "La voce 'Gestione Reperibilità H24' (€ 1.200,00) costituisce doppia fatturazione.",
        "Importo non dovuto: € 1.200,00 + IVA 22% = € 1.464,00.",
    ])
    doc.build(el)
    print("  ✓  siena_gen_doppia_voce.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Scenario B – Capitolato Tecnico Opere Civili e Verde
# ─────────────────────────────────────────────────────────────────────────────

def tech_verde_fuori_listino() -> None:
    """B1 – Sfalcio erba a €0,14/mq vs listino tabellare €0,08/mq (+75%)."""
    doc = _new_doc("siena_tech_verde_fuori_listino.pdf")
    el = _header(
        _VENDOR_TECH, _VENDOR_TECH_ADDR, _VENDOR_TECH_VAT,
        _CLIENT, _CLIENT_ADDR,
        "2024/EE-0118", "05 Marzo 2024", _CONTRACT_TECH,
    )
    el += [
        Paragraph("<b>Oggetto:</b> Manutenzione Aree Verdi – Ciclo Primaverile – Marzo 2024", S_BODY),
        Spacer(1, 0.3 * cm),
        _items_table([
            ["Descrizione", "U.M.", "Qtà", "Prezzo unit.", "Totale"],
            ["Sfalcio erba e raccolta – campus principale\n(S. Miniato, Prato S. Agostino, viali interni)",
             "mq", "12.500", eur("0.14"), eur("1750.00")],
            ["Sfalcio erba – aree periferiche\n(Rettorato, Polo didattico Magistero)",
             "mq",  "4.200", eur("0.14"), eur("588.00")],
            ["Potatura siepi perimetrali",
             "m.l.", "380",  eur("2.80"),  eur("1064.00")],
            ["Rimozione e smaltimento vegetazione di risulta",
             "mc",    "8",   eur("45.00"),  eur("360.00")],
            ["Trattamento fitosanitario preventivo",
             "interv.", "1", eur("420.00"),  eur("420.00")],
        ], [7.8, 1.4, 1.2, 2.6, 3.0]),
        Spacer(1, 0.4 * cm),
    ]
    subtotal = q("4182.00")
    vat_r    = q("22")
    vat_a    = q("920.04")
    total    = q("5102.04")
    el.append(_totals(subtotal, vat_r, vat_a, total))
    el += _notes([
        "Capitolato Tecnico – Allegato B 'Prezzi Unitari Verde': sfalcio erba = € 0,08/mq (tariffa tabellare).",
        "Prezzo applicato in fattura: € 0,14/mq → maggiorazione del 75% rispetto al listino contrattuale.",
        "Sfalcio totale: 16.700 mq. Costo fatturato: € 2.338,00. Costo contrattuale: € 1.336,00.",
        "Overbilling sfalcio: € 1.002,00 + IVA 22% = € 1.222,44.",
    ])
    doc.build(el)
    print("  ✓  siena_tech_verde_fuori_listino.pdf")


def tech_materiali_duplicati() -> None:
    """B2 – Materiali di consumo addebitati, ma a carico del fornitore per Art. 9."""
    doc = _new_doc("siena_tech_materiali_duplicati.pdf")
    el = _header(
        _VENDOR_TECH, _VENDOR_TECH_ADDR, _VENDOR_TECH_VAT,
        _CLIENT, _CLIENT_ADDR,
        "2024/EE-0135", "10 Marzo 2024", _CONTRACT_TECH,
    )
    el += [
        Paragraph("<b>Oggetto:</b> Manutenzione Ordinaria Opere Civili – Polo Scientifico – Marzo 2024", S_BODY),
        Spacer(1, 0.3 * cm),
        _items_table([
            ["Descrizione", "U.M.", "Qtà", "Prezzo unit.", "Totale"],
            ["Ripristino intonaci interni – aule 101-105\n(rasatura e tinteggiatura, Polo Scientifico)",
             "mq", "95", eur("18.50"), eur("1757.50")],
            ["Rifacimento giunti pavimentazione esterna\n(cortile principale, Polo Scientifico)",
             "m.l.", "120", eur("12.00"), eur("1440.00")],
            ["Materiali di consumo\n(pittura, stucco, sigillanti – lotti 2024/03/A-B-C)",
             "forfait", "1", eur("380.00"), eur("380.00")],
            ["Nolo attrezzatura e ponteggi",
             "giorno", "3", eur("65.00"), eur("195.00")],
            ["Smaltimento macerie e materiale di risulta",
             "mc", "2,5", eur("38.00"), eur("95.00")],
        ], [7.8, 1.6, 1.0, 2.6, 3.0]),
        Spacer(1, 0.4 * cm),
    ]
    subtotal = q("3867.50")
    vat_r    = q("22")
    vat_a    = q("850.85")
    total    = q("4718.35")
    el.append(_totals(subtotal, vat_r, vat_a, total))
    el += _notes([
        "Capitolato Tecnico – Art. 9 'Oneri a carico dell'Appaltatore': i materiali di consumo",
        "(pitture, stucchi, sigillanti, adesivi, guarnizioni) sono a totale carico dell'Appaltatore.",
        "La voce 'Materiali di consumo' (€ 380,00) non è addebitabile all'Ateneo.",
        "Importo non dovuto: € 380,00 + IVA 22% = € 463,60.",
    ])
    doc.build(el)
    print("  ✓  siena_tech_materiali_duplicati.pdf")


def tech_intervento_domenica() -> None:
    """B3 – Maggiorazione festiva 30% applicata a intervento programmato (non urgenza)."""
    doc = _new_doc("siena_tech_intervento_domenica.pdf")
    el = _header(
        _VENDOR_TECH, _VENDOR_TECH_ADDR, _VENDOR_TECH_VAT,
        _CLIENT, _CLIENT_ADDR,
        "2024/EE-0152", "18 Marzo 2024", _CONTRACT_TECH,
    )
    el += [
        Paragraph("<b>Oggetto:</b> Ispezione Programmata Coperture – Edificio Rettorato – 17/03/2024", S_BODY),
        Spacer(1, 0.2 * cm),
        Paragraph(
            "Data intervento: <b>17 Marzo 2024 (domenica)</b>.<br/>"
            "Tipologia: <b>Ispezione semestrale programmata</b> – inclusa nel Piano Manutenzioni 2024"
            " approvato dall'Ateneo in data 12/12/2023.",
            S_BODY,
        ),
        Spacer(1, 0.3 * cm),
        _items_table([
            ["Descrizione", "U.M.", "Qtà", "Prezzo unit.", "Totale"],
            ["Ispezione copertura piana – Rettorato\n(verifica impermeabilizzazione e scarichi)",
             "ore", "5", eur("42.00"), eur("210.00")],
            ["Maggiorazione festiva 30%\n(intervento in giornata domenicale 17/03/2024)",
             "forfait", "1", eur("63.00"), eur("63.00")],
            ["Redazione rapporto tecnico ispezione",
             "forfait", "1", eur("75.00"), eur("75.00")],
            ["Verifica gronde e pluviali – prospetto Nord/Est",
             "ore", "2", eur("42.00"), eur("84.00")],
        ], [7.8, 1.6, 1.0, 2.6, 3.0]),
        Spacer(1, 0.4 * cm),
    ]
    subtotal = q("432.00")
    vat_r    = q("22")
    vat_a    = q("95.04")
    total    = q("527.04")
    el.append(_totals(subtotal, vat_r, vat_a, total))
    el += _notes([
        "Capitolato Tecnico – Art. 14 'Maggiorazioni': la maggiorazione festiva (30%) si applica",
        "ESCLUSIVAMENTE agli interventi d'urgenza non pianificati richiesti dall'Ateneo con carattere d'emergenza.",
        "L'ispezione del 17/03/2024 è un intervento programmato (Piano Manutenzioni 2024).",
        "La maggiorazione di € 63,00 non è contrattualmente dovuta.",
        "Importo contestabile: € 63,00 + IVA 22% = € 76,86.",
    ])
    doc.build(el)
    print("  ✓  siena_tech_intervento_domenica.pdf")


def tech_errore_somma() -> None:
    """B4 – Totale imponibile (€11.212) non corrisponde alla somma righe (€10.572): delta +€640."""
    doc = _new_doc("siena_tech_errore_somma.pdf")
    el = _header(
        _VENDOR_TECH, _VENDOR_TECH_ADDR, _VENDOR_TECH_VAT,
        _CLIENT, _CLIENT_ADDR,
        "2024/EE-0168", "28 Marzo 2024", _CONTRACT_TECH,
    )
    el += [
        Paragraph("<b>Oggetto:</b> Manutenzione Straordinaria – Interventi Edili Misti – Marzo 2024", S_BODY),
        Spacer(1, 0.3 * cm),
        _items_table([
            ["Descrizione", "U.M.", "Qtà", "Prezzo unit.", "Totale"],
            ["Rifacimento pavimentazione esterna – lotto A\n(piazzale ingresso principale)",
             "mq", "85", eur("38.00"), eur("3230.00")],
            ["Ripristino cornicioni e fasce marcapiano\n(facciata Sud – Edificio Lettere)",
             "m.l.", "42", eur("28.50"), eur("1197.00")],
            ["Impermeabilizzazione terrazza – Polo Scienze\n(membrana bituminosa armata)",
             "mq", "65", eur("32.00"), eur("2080.00")],
            ["Sostituzione serramenti deteriorati – n. 6 unità\n(finestre PVC doppio vetro con cassonetto)",
             "pz", "6", eur("380.00"), eur("2280.00")],
            ["Tinteggiatura scale interne – Edificio Economia\n(intonaco e vernice traspirante)",
             "mq", "210", eur("8.50"), eur("1785.00")],
        ], [7.8, 1.4, 1.0, 2.8, 3.0]),
        Spacer(1, 0.15 * cm),
    ]
    # Print the real row sum for transparency
    real_sum = q("10572.00")   # 3230+1197+2080+2280+1785
    el.append(Paragraph(f"Somma voci di riga: {eur(real_sum)}", S_SUM))
    el.append(Spacer(1, 0.3 * cm))

    # Wrong totals: subtotal inflated by 640
    subtotal_wrong = q("11212.00")
    vat_r          = q("22")
    vat_a_wrong    = q("2466.64")   # 11212 × 22%
    total_wrong    = q("13678.64")
    el.append(_totals(subtotal_wrong, vat_r, vat_a_wrong, total_wrong))
    el += _notes([
        f"Somma effettiva delle voci di riga: € 10.572,00.",
        f"Imponibile dichiarato: € 11.212,00 — differenza non giustificata: € 640,00.",
        f"IVA corretta (22% su € 10.572,00): € 2.325,84.",
        f"Totale corretto: € 12.897,84 — fatturato: € 13.678,64.",
        f"Overbilling totale (IVA inclusa): € 780,80.",
    ])
    doc.build(el)
    print("  ✓  siena_tech_errore_somma.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nGenerazione PDF -> {OUT_DIR.resolve()}\n")

    print("Scenario A – Capitolato Generale FM:")
    gen_prezzo_extra()
    gen_ritardo_penale()
    gen_iva_error()
    gen_doppia_voce()

    print("\nScenario B – Capitolato Tecnico Opere Civili e Verde:")
    tech_verde_fuori_listino()
    tech_materiali_duplicati()
    tech_intervento_domenica()
    tech_errore_somma()

    pdfs = sorted(OUT_DIR.glob("*.pdf"))
    print(f"\n{'─'*60}")
    print(f"Totale: {len(pdfs)} PDF generati in {OUT_DIR.resolve()}")
    print(f"{'─'*60}")
    for p in pdfs:
        print(f"  {p.name:<50}  {p.stat().st_size // 1024:>3} KB")
