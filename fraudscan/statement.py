"""Estados de cuenta en PDF: formato propio de FraudScan, generación y lectura.

El formato es deliberadamente simple para que el servicio pueda leerlo sin ambigüedad:

  - Un bloque de encabezado con líneas "Etiqueta: valor" (tarjeta, tipo, límite, residencia...).
  - Una tabla con cuadrícula, una transacción por fila:
        N.º | Fecha | Hora | Categoría | ID de comercio | Canal | Ubicación | Incidencia | Monto (USD)

El banco es ficticio y el documento lo dice en su encabezado: es material de demostración.
"""
from __future__ import annotations

import io
import re
from datetime import date, datetime, timedelta

from . import schema as S
from .online import Transaction

BANK_NAME = "Banco Demo FraudScan"
DISCLAIMER = "Documento de demostración. Banco ficticio y datos sintéticos; no es un estado de cuenta real."
COLUMNS = ["N.º", "Fecha", "Hora", "Categoría", "ID de comercio", "Canal", "Ubicación", "Incidencia", "Monto (USD)"]
ONLINE_LABEL = "En línea"
CARD_TYPE_LABELS = {"Credit": "Crédito", "Debit": "Débito", "Debit (Prepaid)": "Prepago"}
HEADER_FIELDS = {                       # etiqueta en el PDF -> clave interna
    "Estado de cuenta N.º": "statement_id",
    "Tarjeta": "card_id",
    "Marca": "brand",
    "Tipo de tarjeta": "card_type",
    "Límite de crédito (USD)": "credit_limit",
    "Chip": "has_chip",
    "Residencia del titular": "home",
    "Periodo": "period",
}
_EPOCH = date(1970, 1, 1)


class StatementError(ValueError):
    """El PDF no tiene el formato de estado de cuenta de FraudScan."""


# ------------------------------------------------------------------ utilidades de fecha y texto
def ts_to_date_time(ts: int) -> tuple[str, str]:
    d = datetime(1970, 1, 1) + timedelta(minutes=int(ts))
    return d.strftime("%Y-%m-%d"), d.strftime("%H:%M")


def date_time_to_ts(day: str, hour: str) -> int:
    d = datetime.strptime(f"{day} {hour}", "%Y-%m-%d %H:%M")
    return (d.date() - _EPOCH).days * 1440 + d.hour * 60 + d.minute


def money(value: float) -> str:
    return ("-" if value < 0 else "") + f"${abs(value):,.2f}"


def parse_money(text: str) -> float:
    t = text.replace("$", "").replace(",", "").replace("−", "-").replace(" ", "")
    return float(t)


# ------------------------------------------------------------------ generación
def build_pdf(path, header: dict, txs: list[Transaction]) -> None:
    """Escribe un estado de cuenta. `header` usa las claves internas de HEADER_FIELDS, con
    `home_state`, `home_city`, `period_start` y `period_end` en lugar de `home` y `period`."""
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    navy, soft = colors.HexColor("#0b2f78"), colors.HexColor("#eef3fb")
    title = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=navy, spaceAfter=3)
    sub = ParagraphStyle("s", fontName="Helvetica", fontSize=9, textColor=colors.HexColor("#555555"), spaceAfter=10)
    line = ParagraphStyle("l", fontName="Helvetica", fontSize=10, leading=14)
    note = ParagraphStyle("n", fontName="Helvetica", fontSize=8.5, textColor=colors.HexColor("#555555"), leading=11)

    values = {
        "statement_id": header["statement_id"], "card_id": header["card_id"], "brand": header.get("brand", "-"),
        "card_type": CARD_TYPE_LABELS.get(header["card_type"], header["card_type"]),
        "credit_limit": f"{float(header['credit_limit']):,.2f}",
        "has_chip": "Sí" if header.get("has_chip") else "No",
        "home": f"{header.get('home_city', '-')}, {header['home_state']}",
        "period": f"{header['period_start']} a {header['period_end']}",
    }
    story = [Paragraph("Estado de cuenta", title), Paragraph(f"{BANK_NAME} · {DISCLAIMER}", sub)]
    labels = list(HEADER_FIELDS)
    half = (len(labels) + 1) // 2
    cells = [[Paragraph(f"<b>{lab}:</b> {values[HEADER_FIELDS[lab]]}", line) for lab in pair if lab]
             for pair in zip(labels[:half], labels[half:] + [None] * (2 * half - len(labels)))]
    info = Table(cells, colWidths=[4.6 * inch, 4.6 * inch])
    info.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                              ("TOPPADDING", (0, 0), (-1, -1), 1)]))
    story += [info, Spacer(1, 10)]

    cell = ParagraphStyle("c", fontName="Helvetica", fontSize=8, leading=9.5)
    wrap = lambda text: Paragraph(escape(text), cell)            # noqa: E731  (texto largo: se parte dentro de la celda)
    data = [COLUMNS]
    for i, tx in enumerate(txs, 1):
        day, hour = ts_to_date_time(tx.ts)
        data.append([
            str(i), day, hour, wrap(f"{int(tx.mcc)} {S.mcc_name(tx.mcc)}"), str(tx.merchant),
            S.CHANNEL_LABELS[tx.channel], wrap(ONLINE_LABEL if tx.loc == S.ONLINE_LOC else tx.loc),
            wrap("; ".join(S.ERROR_LABELS[e] for e in tx.errors)), money(tx.amount)])
    table = Table(data, repeatRows=1,
                  colWidths=[0.4 * inch, 0.85 * inch, 0.5 * inch, 2.1 * inch, 1.65 * inch, 1.0 * inch,
                             1.2 * inch, 1.6 * inch, 0.95 * inch])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("BACKGROUND", (0, 0), (-1, 0), navy),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, soft]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#8a9bbd")), ("ALIGN", (-1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    charges = sum(t.amount for t in txs if t.amount > 0)
    refunds = -sum(t.amount for t in txs if t.amount < 0)
    story += [table, Spacer(1, 10),
              Paragraph(f"<b>Transacciones:</b> {len(txs)} &nbsp;&nbsp; <b>Total de cargos:</b> {money(charges)} "
                        f"&nbsp;&nbsp; <b>Total de devoluciones:</b> {money(refunds)}", line),
              Spacer(1, 6),
              Paragraph("Canal: Chip, Banda magnética o En línea. Ubicación: estado de EE. UU. (2 letras), país, "
                        "o En línea. Incidencia: error reportado por la terminal al procesar la compra.", note)]
    SimpleDocTemplate(str(path), pagesize=landscape(letter), leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                      topMargin=0.55 * inch, bottomMargin=0.55 * inch,
                      title=f"Estado de cuenta {header['statement_id']}", author=BANK_NAME).build(story)


# ------------------------------------------------------------------ lectura
def _clean(cell) -> str:
    return re.sub(r"\s+", " ", (cell or "").replace("\n", " ")).strip()


def parse_pdf(data: bytes) -> tuple[dict, list[Transaction]]:
    """Lee un estado de cuenta y devuelve (encabezado, transacciones en orden cronológico)."""
    import pdfplumber

    channel_by_label = {v: k for k, v in S.CHANNEL_LABELS.items()}
    error_by_label = {v: k for k, v in S.ERROR_LABELS.items()}
    type_by_label = {v: k for k, v in CARD_TYPE_LABELS.items()}
    try:
        pdf = pdfplumber.open(io.BytesIO(data))
    except Exception as exc:                                   # archivo dañado o que no es PDF
        raise StatementError("El archivo no es un PDF válido.") from exc

    header: dict = {}
    txs: list[Transaction] = []
    with pdf:
        if len(pdf.pages) > 40:
            raise StatementError("El estado de cuenta tiene demasiadas páginas (máximo 40).")
        first_text = pdf.pages[0].extract_text() or ""
        for label, key in HEADER_FIELDS.items():
            # Las etiquetas van de dos en dos por renglón: el valor termina donde empieza otra etiqueta.
            others = "|".join(re.escape(x) + ":" for x in HEADER_FIELDS if x != label)
            m = re.search(re.escape(label) + r":\s*(.+?)\s*(?=" + others + r"|$)", first_text, flags=re.M)
            if m:
                header[key] = m.group(1).strip()
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    cells = [_clean(c) for c in row]
                    if len(cells) != len(COLUMNS) or not cells[0].isdigit():
                        continue                                # encabezado de la tabla u otra tabla
                    n, day, hour, cat, merchant, chan, loc, inc, amount = cells
                    try:
                        channel = channel_by_label[chan]
                        errors = [error_by_label[e.strip()] for e in inc.split(";") if e.strip()]
                        txs.append(Transaction(
                            ts=date_time_to_ts(day, hour), amount=parse_money(amount), channel=channel,
                            merchant=merchant.replace(" ", ""), mcc=int(cat.split()[0]),
                            loc=S.ONLINE_LOC if loc == ONLINE_LABEL else loc, errors=errors))
                    except (KeyError, ValueError, IndexError) as exc:
                        raise StatementError(f"No se pudo leer la fila {n} del estado de cuenta: {cells}") from exc

    missing = [lab for lab, key in HEADER_FIELDS.items() if key not in header]
    if missing or not txs:
        raise StatementError("El PDF no tiene el formato de estado de cuenta de FraudScan "
                             "(falta el encabezado o la tabla de transacciones). Usa uno de los ejemplos como plantilla.")
    try:
        header["credit_limit"] = float(header["credit_limit"].replace(",", ""))
    except ValueError as exc:
        raise StatementError("El límite de crédito no es un número.") from exc
    header["card_type"] = type_by_label.get(header["card_type"], header["card_type"])
    header["has_chip"] = header["has_chip"].lower().startswith("s")
    city, _, state = header.pop("home").rpartition(",")
    header["home_city"], header["home_state"] = city.strip(), state.strip()
    start, _, end = header.pop("period").partition(" a ")
    header["period_start"], header["period_end"] = start.strip(), end.strip()
    txs.sort(key=lambda t: t.ts)                                 # estable: respeta el orden en empates
    return header, txs
