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
