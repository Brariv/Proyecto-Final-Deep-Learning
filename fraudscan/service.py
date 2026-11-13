"""Núcleo del servicio de autorización: carga los modelos, puntúa una transacción, decide y explica.

Flujo de punta a punta:  transacción -> variables (online.py) -> modelo A o B -> umbrales elegidos
por valor -> APROBAR / VERIFICAR / BLOQUEAR -> razones legibles.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import schema as S
from .models import load_model
from .online import CardState, Transaction, Vocab
from .value import Costs, gains

# Grupos de variables que se "apagan" para explicar una decisión (oclusión).
# (etiqueta, variables numéricas, variables categóricas)
REASON_GROUPS = [
    ("Monto inusual para esta tarjeta", ["log_amt", "log_amt_ratio", "amt_over_limit", "is_refund"], []),
    ("Ubicación de la compra", ["is_foreign", "away_from_home", "state_changed", "loc_new",
                                "loc_share_prior"], ["loc"]),
    ("Canal de pago", [], ["channel"]),
    ("Comercio o categoría poco habitual", ["merchant_new", "mcc_new"], ["mcc"]),
    ("Ritmo de compras en las últimas horas", ["log_dt_prev", "log_dt_5", "log_cnt_24h", "log_sum_24h"], []),
    ("Hora y día de la compra", ["hour_sin", "hour_cos", "dow_sin", "dow_cos"], []),
    ("Errores al ingresar los datos", ["err_bad_pin", "err_bad_cvv", "err_bad_card", "err_bad_exp",
                                      "err_bad_zip", "err_insufficient", "err_glitch"], []),
]
HISTORY_REASON = "Comportamiento reciente de la tarjeta"


class Scorer:
    """Carga los artefactos una vez y atiende transacciones una por una."""

    def __init__(self, artifacts_dir: str | Path):
        art = Path(artifacts_dir)
        with open(art / "serving.json", encoding="utf-8") as f:
            self.serving = json.load(f)
        with open(art / "vocab.json", encoding="utf-8") as f:
            self.vocab_meta = json.load(f)
        self.vocab = Vocab(self.vocab_meta)
        self.costs = Costs(**self.serving["costs"])
        self.models = {key: load_model(art / m["file"]) for key, m in self.serving["models"].items()}
        self.thresholds = {key: m["thresholds"] for key, m in self.serving["models"].items()}
        torch.set_num_threads(1)                      # latencia estable para una transacción a la vez
        self._num_base = np.array(self.vocab_meta["num_baseline"], dtype="float32")
        self._cat_base = self.vocab_meta["cat_baseline"]
        self.latencies_ms: list[float] = []
        self.decisions = {S.APPROVE: 0, S.VERIFY: 0, S.BLOCK: 0}
        self._warm_up()

    @torch.no_grad()
    def _warm_up(self) -> None:
        """Una pasada en vacío por cada modelo, para que la primera transacción real no pague el arranque."""
        k = int(self.vocab_meta["seq_len"])
        num = torch.zeros(1, k, S.N_NUM)
        cat = torch.zeros(1, k, len(S.CAT_FEATURES), dtype=torch.long)
        for model in self.models.values():
            for _ in range(5):
                if model.kind == "mlp":
                    model(num[:, -1], cat[:, -1])
                else:
                    model(num, cat, torch.ones(1, k))

    # ------------------------------------------------------------------ puntuación
    @torch.no_grad()
    def _score_variants(self, key: str, state: CardState, x_num: np.ndarray, x_cat: np.ndarray,
                        explain: bool) -> tuple[float, list[dict]]:
        """Score de la transacción y, si se pide, cuánto baja al neutralizar cada grupo de variables."""
        model = self.models[key]
        nums, cats, labels = [x_num], [x_cat], []
        if explain:
            home_loc = self.vocab.loc(state.home_state)
            for label, num_names, cat_names in REASON_GROUPS:
                n, c = x_num.copy(), x_cat.copy()
                for name in num_names:
                    n[S.NUM_INDEX[name]] = self._num_base[S.NUM_INDEX[name]]
                for name in cat_names:
                    j = S.CAT_FEATURES.index(name)
                    c[j] = home_loc if name == "loc" else self._cat_base[j]
                nums.append(n)
                cats.append(c)
                labels.append(label)

        if model.kind == "mlp":
            logits = model(torch.from_numpy(np.stack(nums)), torch.from_numpy(np.stack(cats)))
        else:
            seqs = [state.sequence(n, c) for n, c in zip(nums, cats)]
            if explain:                                # variante extra: sin historial previo
                n, c, m = state.sequence(x_num, x_cat)
                m = m.copy()
                m[:-1] = 0.0
                seqs.append((n * m[:, None], c * m[:, None].astype("int64"), m))
                labels.append(HISTORY_REASON)
            logits = model(*[torch.from_numpy(np.stack(a)) for a in zip(*seqs)])
        scores = torch.sigmoid(logits).numpy().astype("float64")
        reasons = [{"label": lab, "points": round(100 * float(scores[0] - s), 1)}
                   for lab, s in zip(labels, scores[1:])]
        reasons = sorted([r for r in reasons if r["points"] >= 1.0], key=lambda r: -r["points"])[:3]
        return float(scores[0]), reasons

    def decide(self, key: str, score: float) -> int:
        th = self.thresholds[key]
        if score >= th["t_block"]:
            return S.BLOCK
        if score >= th["t_verify"]:
            return S.VERIFY
        return S.APPROVE

    def score(self, state: CardState, tx: Transaction, model: str = "B", commit: bool = True,
              explain: bool = True) -> dict:
        """Puntúa `tx` con el modelo elegido (y con el otro, para comparar) y actualiza el historial."""
        if model not in self.models:
            raise KeyError(f"Modelo desconocido: {model}")
        t0 = time.perf_counter()
        x_num, x_cat = state.features(tx, self.vocab)
        score, reasons = self._score_variants(model, state, x_num, x_cat, explain=False)
        decision = self.decide(model, score)
        latency_ms = (time.perf_counter() - t0) * 1000   # lo que tarda la decisión (sin explicación)

        if explain and decision != S.APPROVE:
            _, reasons = self._score_variants(model, state, x_num, x_cat, explain=True)
        others = {}
        for key in self.models:
            if key != model:
                s, _ = self._score_variants(key, state, x_num, x_cat, explain=False)
                others[key] = {"score": s, "decision": self.decide(key, s)}

        c = S.NUM_INDEX
        context = {
            "amount_vs_usual": round(float(np.exp(x_num[c["log_amt_ratio"]])), 2),
            "tx_last_24h": int(round(float(np.expm1(x_num[c["log_cnt_24h"]])))),
            "minutes_since_previous": None if state.n == 0 else int(round(float(np.expm1(x_num[c["log_dt_prev"]])))),
            "new_merchant": bool(x_num[c["merchant_new"]]),
            "new_location": bool(x_num[c["loc_new"]]),
            "away_from_home": bool(x_num[c["away_from_home"]]),
            "foreign": bool(x_num[c["is_foreign"]]),
            "prior_transactions": state.n,
        }
        if commit:
            state.commit(tx, x_num, x_cat)
        self.latencies_ms.append(latency_ms)
        self.decisions[decision] += 1
        th = self.thresholds[model]
        return {
            "model": model, "score": score, "decision": decision,
            "decision_label": S.DECISION_LABELS[decision],
            "thresholds": th, "reasons": reasons, "latency_ms": round(latency_ms, 3),
            "context": context, "other_models": others,
        }

    # ------------------------------------------------------------------ estados de cuenta
    def analyze_statement(self, state: CardState, txs: list[Transaction], model: str = "B") -> dict:
        """Puntúa en orden cronológico todas las transacciones de un estado de cuenta. `state` es el
        historial de la tarjeta al inicio del periodo y se va actualizando (pasa una copia)."""
        rows = []
        for tx in txs:
            out = self.score(state, tx, model=model, commit=True)
            rows.append({"ts": tx.ts, "amount": tx.amount, "channel": tx.channel, "mcc": tx.mcc, "loc": tx.loc,
                         "errors": tx.errors, "score": out["score"], "decision": out["decision"],
                         "decision_label": out["decision_label"], "reasons": out["reasons"],
                         "latency_ms": out["latency_ms"], "context": out["context"]})
        blocked = [r for r in rows if r["decision"] == S.BLOCK]
        verify = [r for r in rows if r["decision"] == S.VERIFY]
        if blocked:
            verdict = {"level": "ALTO", "title": "Fraude probable",
                       "text": "Hay transacciones que el modelo bloquearía. Conviene bloquear la tarjeta y contactar al titular."}
        elif verify:
            verdict = {"level": "MEDIO", "title": "Revisar",
                       "text": "Hay transacciones dudosas. Conviene confirmarlas con el titular antes de tomar otra acción."}
        else:
            verdict = {"level": "BAJO", "title": "Sin señales de fraude",
                       "text": "Ninguna transacción supera el umbral de verificación."}
        charge = lambda rs: float(sum(max(r["amount"], 0.0) for r in rs))   # noqa: E731
        summary = {
            "transactions": len(rows), "total_usd": charge(rows),
            "blocked": len(blocked), "blocked_usd": charge(blocked),
            "verify": len(verify), "verify_usd": charge(verify),
            "max_score": max((r["score"] for r in rows), default=0.0),
            "latency_total_ms": float(sum(r["latency_ms"] for r in rows)),
        }
        return {"model": model, "rows": rows, "summary": summary, "verdict": verdict,
                "thresholds": self.thresholds[model]}

    # ------------------------------------------------------------------ valor y operación
    def realized_value(self, decision: int, is_fraud: bool, amount: float) -> float:
        """Valor en USD de una decisión cuando ya se conoce la etiqueta real (solo para el demo)."""
        g_block, g_verify = gains(np.array([int(is_fraud)]), np.array([amount]), self.costs)
        return float(g_block[0] if decision == S.BLOCK else g_verify[0] if decision == S.VERIFY else 0.0)

    def stats(self) -> dict:
        lat = np.array(self.latencies_ms) if self.latencies_ms else np.array([0.0])
        return {"scored": len(self.latencies_ms), "latency_p50_ms": float(np.percentile(lat, 50)),
                "latency_p95_ms": float(np.percentile(lat, 95)),
                "decisions": {S.DECISION_LABELS[k]: v for k, v in self.decisions.items()}}
