"""Prueba de punta a punta del servicio. Requiere los artefactos (scripts 05 y 06); si no, se omite."""
from pathlib import Path

import pytest

ART = Path(__file__).resolve().parent.parent / "artifacts"
if not (ART / "serving.json").exists() or not (ART / "demo.json").exists():
    pytest.skip("No hay artefactos del servicio", allow_module_level=True)

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _card():
    meta = client.get("/api/meta").json()
    return meta, meta["cards"][0]


def test_health_and_meta():
    assert client.get("/health").json()["status"] == "ok"
    meta, card = _card()
    assert meta["models"] and meta["locations"] and meta["mccs"] and card["id"]


def test_score_normal_vs_suspicious():
    client.post("/api/reset")
    meta, card = _card()
    model = "B" if "B" in meta["models"] else "A"
    normal = client.post("/api/score", json={
        "card_id": card["id"], "amount": card["usual_amount"] or 30, "channel": "Swipe Transaction",
        "mcc": 5411, "loc": card["home_state"], "minutes_after_last": 600, "model": model, "commit": False}).json()
    risky = client.post("/api/score", json={
        "card_id": card["id"], "amount": 950, "channel": "Swipe Transaction", "mcc": 5311, "loc": "Italy",
        "minutes_after_last": 5, "model": model, "commit": False}).json()
    assert 0 <= normal["score"] <= 1 and 0 <= risky["score"] <= 1
    assert risky["score"] > normal["score"]
    assert risky["decision"] >= normal["decision"]
    assert normal["latency_ms"] > 0


def test_validation_errors():
    _, card = _card()
    r = client.post("/api/score", json={"card_id": "no-existe", "amount": 10, "loc": "CA"})
    assert r.status_code == 404
    r = client.post("/api/score", json={"card_id": card["id"], "amount": 10, "channel": "Online Transaction", "loc": "CA"})
    assert r.status_code == 422
    r = client.post("/api/score", json={"card_id": card["id"], "amount": 10, "loc": "CA", "errors": ["Nope"]})
    assert r.status_code == 422


def test_blocked_purchase_does_not_enter_history():
    client.post("/api/reset")
    meta, card = _card()
    model = "B" if "B" in meta["models"] else "A"
    before = card["prior_transactions"]
    r = client.post("/api/score", json={"card_id": card["id"], "amount": 950, "channel": "Swipe Transaction",
                                        "mcc": 5311, "loc": "Italy", "minutes_after_last": 5, "model": model}).json()
    assert r["card"]["prior_transactions"] == before + (1 if r["committed"] else 0)
    assert r["committed"] == (r["decision"] != 2)
    client.post("/api/reset")


def test_statement_samples_are_analyzed():
    samples = client.get("/api/statement/samples").json()
    if not samples:
        pytest.skip("No hay estados de cuenta de ejemplo")
    static = Path(__file__).resolve().parent.parent / "app" / "static" / "samples"
    by_file = {}
    for s in samples:
        r = client.post("/api/statement?model=B", content=(static / s["file"]).read_bytes(),
                        headers={"Content-Type": "application/pdf"})
        assert r.status_code == 200, r.text
        out = r.json()
        assert out["header"]["statement_id"] == s["statement_id"]
        assert out["summary"]["transactions"] == len(out["rows"]) > 0
        assert out["truth"]["frauds"] == sum(row["is_fraud"] for row in out["rows"])
        by_file[s["file"]] = out
    clean, fraud = by_file["estado_sin_fraude.pdf"], by_file["estado_fraude_en_linea.pdf"]
    assert clean["verdict"]["level"] == "BAJO" and clean["truth"]["frauds"] == 0
    assert fraud["verdict"]["level"] == "ALTO" and fraud["truth"]["frauds_flagged"] >= 1


def test_statement_rejects_bad_uploads():
    assert client.post("/api/statement", content=b"").status_code == 422
    assert client.post("/api/statement", content=b"no soy un pdf").status_code == 422
    assert client.post("/api/statement?model=Z", content=b"x").status_code == 422
    assert client.post("/api/statement", content=b"0" * (5 * 1024 * 1024 + 1)).status_code == 413


def test_stream_accumulates_value_and_resets():
    client.post("/api/reset")
    meta, _ = _card()
    model = "B" if "B" in meta["models"] else "A"
    out = client.post("/api/stream/next", json={"n": 500, "model": model}).json()
    assert out["position"] == out["total"] == meta["stream"]["total"]
    assert out["totals"]["none"]["value_usd"] == 0
    frauds = sum(r["is_fraud"] for r in out["rows"])
    assert frauds == meta["stream"]["frauds"]
    t = out["totals"][model]
    assert t["frauds_caught"] + t["frauds_missed"] == frauds
    client.post("/api/reset")
    assert client.get("/api/meta").json()["stream"]["position"] == 0
