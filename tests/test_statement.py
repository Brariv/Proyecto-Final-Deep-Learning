"""El formato de estado de cuenta se puede escribir y volver a leer sin perder información."""
import pytest

from fraudscan import statement
from fraudscan.online import Transaction

pytest.importorskip("reportlab")
pytest.importorskip("pdfplumber")

HEADER = {"statement_id": "EC-TEST-1-202001", "card_id": "9999-1", "brand": "Visa", "card_type": "Credit",
          "credit_limit": 12500.0, "has_chip": True, "home_state": "CA", "home_city": "San José",
          "period_start": "2020-01-01", "period_end": "2020-01-31"}


def _txs():
    base = statement.date_time_to_ts("2020-01-03", "09:15")
    return [
        Transaction(base, 42.5, "Chip Transaction", "-727612092139916043", 5411, "CA"),
        Transaction(base + 90, 1234.56, "Online Transaction", "4060646732831064559", 5732, "ONLINE", ["Bad CVV"]),
        Transaction(base + 91, -78.0, "Swipe Transaction", "190253443608377572", 3359, "Dominican Republic"),
        Transaction(base + 3000, 9.99, "Online Transaction", "88", 4899, "ONLINE", ["Bad PIN", "Insufficient Balance"]),
    ]


def test_roundtrip(tmp_path):
    path = tmp_path / "estado.pdf"
    txs = _txs()
    statement.build_pdf(path, HEADER, txs)
    header, parsed = statement.parse_pdf(path.read_bytes())
    assert header == HEADER
    assert parsed == txs


def test_many_rows_span_pages(tmp_path):
    path = tmp_path / "largo.pdf"
    base = statement.date_time_to_ts("2020-01-01", "00:00")
    txs = [Transaction(base + 37 * i, 5.0 + i, "Chip Transaction", str(1000 + i), 5411, "TX") for i in range(120)]
    statement.build_pdf(path, HEADER, txs)
    _, parsed = statement.parse_pdf(path.read_bytes())
    assert parsed == txs


def test_rejects_other_files(tmp_path):
    with pytest.raises(statement.StatementError):
        statement.parse_pdf(b"esto no es un pdf")
    from reportlab.pdfgen import canvas
    other = tmp_path / "otro.pdf"
    c = canvas.Canvas(str(other))
    c.drawString(72, 720, "Un documento cualquiera")
    c.save()
    with pytest.raises(statement.StatementError):
        statement.parse_pdf(other.read_bytes())
