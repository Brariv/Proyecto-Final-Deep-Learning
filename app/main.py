"""API de FraudScan (FastAPI). Ejecutar desde la raíz del repo:

    uvicorn app.main:app --reload

Abre http://127.0.0.1:8000 para el demo. Documentación interactiva en /docs.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fraudscan import schema as S
from fraudscan import statement as ST
from fraudscan.online import CardState, Transaction
from fraudscan.service import Scorer

ARTIFACTS = Path(os.environ.get("FRAUDSCAN_ARTIFACTS", ROOT / "artifacts"))
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="FraudScan", version="0.1.0",
              description="Autorización de transacciones con deep learning, con umbrales elegidos por valor.")
scorer = Scorer(ARTIFACTS)


class Demo:
    """Estado del demo: tarjetas con su historial y el flujo de transacciones no vistas."""

    def __init__(self):
        with open(ARTIFACTS / "demo.json", encoding="utf-8") as f:
            self.raw = json.load(f)
        self.reset()

    def reset(self):
        self.cards = {cid: CardState.from_dict(d) for cid, d in self.raw["cards"].items()}
        self.stream = self.raw["stream"]
        self.pos = 0
        self.totals = {k: self._zero() for k in list(scorer.models) + ["none"]}

    @staticmethod
    def _zero():
        return {"value_usd": 0.0, "fraud_stopped_usd": 0.0, "fraud_missed_usd": 0.0,
                "verified": 0, "blocked": 0, "legit_bothered": 0, "frauds_caught": 0, "frauds_missed": 0}


demo = Demo()

# Estados de cuenta de ejemplo: historial de la tarjeta al inicio del periodo y etiquetas reales.
try:
    with open(ARTIFACTS / "statements.json", encoding="utf-8") as _f:
        STATEMENTS = json.load(_f)
except FileNotFoundError:
    STATEMENTS = {}
MAX_PDF_BYTES = 5 * 1024 * 1024


def fmt_ts(ts: int) -> str:
    return (datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=int(ts))).strftime("%Y-%m-%d %H:%M")


def card_summary(cid: str, st: CardState) -> dict:
    return {"id": cid, "brand": st.info.get("brand"), "card_type": st.card_type, "home_state": st.home_state,
            "home_city": st.info.get("home_city"), "credit_limit": st.credit_limit, "has_chip": st.has_chip,
            "prior_transactions": st.n, "usual_amount": round(st.sum_abs / st.n, 2) if st.n else None,
            "last_ts": fmt_ts(st.recent[-1][0]) if st.recent else None}


class ScoreRequest(BaseModel):
    card_id: str
    amount: float = Field(..., description="Monto en USD; negativo = devolución")
    channel: str = Field("Chip Transaction", description="Swipe Transaction | Chip Transaction | Online Transaction")
    mcc: int = Field(5411, description="Categoría del comercio (MCC)")
    loc: str = Field(..., description='Estado de EE. UU. (2 letras), país, u "ONLINE"')
    merchant: str | None = Field(None, description="Identificador del comercio (opcional)")
    errors: list[str] = Field(default_factory=list)
    minutes_after_last: int = Field(60, ge=0, description="Minutos desde la última compra de la tarjeta")
    model: str = Field("B", description="A = MLP, B = LSTM")
    commit: bool = Field(True, description="Agregar la transacción al historial si no se bloquea")


class StreamRequest(BaseModel):
    n: int = Field(10, ge=1, le=500)
    model: str = "B"


@app.get("/health")
def health():
    return {"status": "ok", "models": list(scorer.models)}


@app.get("/api/meta")
def meta():
    locs = [{"value": name, "label": "En línea" if name == S.ONLINE_LOC else name,
             "foreign": S.is_foreign_loc(name)} for name in scorer.vocab.loc_vocab[1:]]
    locs.sort(key=lambda x: (x["foreign"], x["label"]))
    mccs = sorted(({"code": m, "name": S.mcc_name(m)} for m in scorer.vocab.mcc_vocab[1:]),
                  key=lambda x: (x["name"], x["code"]))
    return {
        "cards": [card_summary(cid, st) for cid, st in demo.cards.items()],
        "channels": [{"value": c, "label": S.CHANNEL_LABELS[c]} for c in S.CHANNELS],
        "locations": locs, "mccs": mccs,
        "errors": [{"value": k, "label": v} for k, v in S.ERROR_LABELS.items()],
        "models": scorer.serving["models"], "costs": scorer.serving["costs"],
        "iterations": scorer.serving.get("iterations", []),
        "comparison_B_vs_A": scorer.serving.get("comparison_B_vs_A"),
        "profile": scorer.serving.get("profile"),
        "stream": {"total": len(demo.stream), "position": demo.pos, "year": demo.raw.get("year"),
                   "frauds": sum(s["is_fraud"] for s in demo.stream)},
    }


@app.post("/api/score")
def score(req: ScoreRequest):
    """Autoriza una transacción: devuelve score, decisión, razones y latencia."""
    state = demo.cards.get(req.card_id)
    if state is None:
        raise HTTPException(404, f"Tarjeta desconocida: {req.card_id}")
    if req.model not in scorer.models:
        raise HTTPException(422, f"Modelo desconocido: {req.model}. Opciones: {list(scorer.models)}")
    if req.channel not in S.CHANNELS:
        raise HTTPException(422, f"Canal inválido. Opciones: {S.CHANNELS}")
    bad = [e for e in req.errors if e not in S.ERROR_FLAGS]
    if bad:
        raise HTTPException(422, f"Errores inválidos: {bad}")
    online = req.channel == "Online Transaction"
    if online != (req.loc == S.ONLINE_LOC):
        raise HTTPException(422, 'Las compras en línea usan la ubicación "ONLINE", y solo ellas.')
    last = state.recent[-1][0] if state.recent else 26_000_000
    tx = Transaction(ts=last + req.minutes_after_last, amount=req.amount, channel=req.channel,
                     merchant=req.merchant or f"demo-{req.mcc}-{req.loc}", mcc=req.mcc, loc=req.loc,
                     errors=req.errors)
    out = scorer.score(state, tx, model=req.model, commit=False)
    # Una compra bloqueada no llega a ocurrir, así que no entra al historial de la tarjeta.
    out["committed"] = bool(req.commit and out["decision"] != S.BLOCK)
    if out["committed"]:
        state.commit(tx, *state.features(tx, scorer.vocab))
    out["timestamp"] = fmt_ts(tx.ts)
    out["card"] = card_summary(req.card_id, state)
    return out


@app.post("/api/stream/next")
def stream_next(req: StreamRequest):
    """Avanza el flujo de transacciones de prueba y acumula el valor real de cada política."""
    if req.model not in scorer.models:
        raise HTTPException(422, f"Modelo desconocido: {req.model}")
    rows = []
    for raw in demo.stream[demo.pos:demo.pos + req.n]:
        state = demo.cards[raw["card_id"]]
        tx = Transaction(ts=raw["ts"], amount=raw["amount"], channel=raw["channel"], merchant=raw["merchant"],
                         mcc=raw["mcc"], loc=raw["loc"], errors=raw["errors"])
        out = scorer.score(state, tx, model=req.model, commit=True)
        decisions = {req.model: out["decision"], **{k: v["decision"] for k, v in out["other_models"].items()},
                     "none": S.APPROVE}
        is_fraud = bool(raw["is_fraud"])
        loss = max(raw["amount"], 0.0) + scorer.costs.chargeback_fee
        for key, d in decisions.items():
            t = demo.totals[key]
            t["value_usd"] += scorer.realized_value(d, is_fraud, raw["amount"])
            t["verified"] += d == S.VERIFY
            t["blocked"] += d == S.BLOCK
            if is_fraud:
                stopped = loss if d == S.BLOCK else scorer.costs.otp_effectiveness * loss if d == S.VERIFY else 0.0
                t["fraud_stopped_usd"] += stopped
                t["fraud_missed_usd"] += loss - stopped
                t["frauds_caught"] += d != S.APPROVE
                t["frauds_missed"] += d == S.APPROVE
            else:
                t["legit_bothered"] += d != S.APPROVE
        rows.append({
            "card_id": raw["card_id"], "timestamp": fmt_ts(raw["ts"]), "amount": raw["amount"],
            "channel": S.CHANNEL_LABELS[raw["channel"]], "mcc_name": S.mcc_name(raw["mcc"]),
            "loc": "En línea" if raw["loc"] == S.ONLINE_LOC else raw["loc"], "is_fraud": is_fraud,
            "model": req.model, "score": out["score"], "decision": out["decision"],
            "decision_label": out["decision_label"], "reasons": out["reasons"], "latency_ms": out["latency_ms"],
            "decisions": {k: S.DECISION_LABELS[v] for k, v in decisions.items()},
        })
    demo.pos += len(rows)
    return {"rows": rows, "position": demo.pos, "total": len(demo.stream), "totals": demo.totals,
            "stats": scorer.stats()}


@app.get("/api/statement/samples")
def statement_samples():
    """Estados de cuenta de ejemplo que se pueden descargar y volver a subir."""
    return [{"statement_id": sid, "title": s["title"], "description": s["description"],
             "url": f"/static/samples/{s['file']}", "file": s["file"]} for sid, s in STATEMENTS.items()]


@app.post("/api/statement")
async def analyze_statement(request: Request, model: str = "B"):
    """Recibe un estado de cuenta en PDF (cuerpo de la petición) y analiza si contiene fraude."""
    if model not in scorer.models:
        raise HTTPException(422, f"Modelo desconocido: {model}. Opciones: {list(scorer.models)}")
    data = await request.body()
    if not data:
        raise HTTPException(422, "No se recibió ningún archivo.")
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(413, "El PDF pesa más de 5 MB.")
    try:
        header, txs = ST.parse_pdf(data)
    except ST.StatementError as exc:
        raise HTTPException(422, str(exc)) from exc

    known = STATEMENTS.get(header["statement_id"])
    warnings = []
    if known and len(known["truth"]) == len(txs):
        state = CardState.from_dict(known["state"])              # copia nueva: no altera el demo
        truth = known["truth"]
    else:
        # Tarjeta desconocida: se parte sin historial y se usan los datos del encabezado.
        state = CardState(header["credit_limit"], header["card_type"], header["has_chip"], header["home_state"],
                          seq_len=int(scorer.vocab_meta["seq_len"]))
        truth = None
        warnings.append("No hay historial previo de esta tarjeta: las primeras compras del periodo se evalúan "
                        "con menos contexto y el resultado es menos confiable.")
    out = scorer.analyze_statement(state, txs, model=model)
    for i, r in enumerate(out["rows"]):
        day, hour = ST.ts_to_date_time(r.pop("ts"))
        r.update(n=i + 1, date=day, time=hour, channel=S.CHANNEL_LABELS[r["channel"]], mcc_name=S.mcc_name(r["mcc"]),
                 loc=ST.ONLINE_LABEL if r["loc"] == S.ONLINE_LOC else r["loc"],
                 errors=[S.ERROR_LABELS[e] for e in r["errors"]])
        if truth is not None:
            r["is_fraud"] = bool(truth[i])
    out["header"] = header
    out["warnings"] = warnings
    if truth is not None:                                        # solo para los ejemplos: comparación con la realidad
        flagged = [r["decision"] != S.APPROVE for r in out["rows"]]
        out["truth"] = {
            "frauds": int(sum(truth)),
            "frauds_flagged": int(sum(1 for f, t in zip(flagged, truth) if f and t)),
            "false_alarms": int(sum(1 for f, t in zip(flagged, truth) if f and not t)),
        }
    return out


@app.post("/api/reset")
def reset():
    demo.reset()
    scorer.latencies_ms.clear()
    scorer.decisions = {k: 0 for k in scorer.decisions}
    return {"status": "ok"}


@app.get("/api/stats")
def stats():
    """Métricas operativas del servicio: latencia y conteo de decisiones."""
    return scorer.stats()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
