"""Modelo de valor: traduce las decisiones del modelo en dólares.

El servicio toma una de tres decisiones por transacción según su score:

    score <  t_verify            -> APROBAR
    t_verify <= score < t_block  -> VERIFICAR (se pide un código al titular)
    score >= t_block             -> BLOQUEAR

El valor se mide contra "no tener modelo" (aprobar todo):

    valor =   Σ_{fraudes bloqueados}   (monto_i + c_contracargo)
            + Σ_{fraudes verificados}  p_otp · (monto_i + c_contracargo)
            − N_verificadas · c_verificación
            − FP_verificados · c_fricción_verificar
            − FP_bloqueados  · c_fricción_bloquear
            − c_operación

Un fraude aprobado no suma nada: su monto es la pérdida residual.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .schema import APPROVE, BLOCK, VERIFY


@dataclass
class Costs:
    chargeback_fee: float = 15.0
    verification_cost: float = 0.25
    friction_cost_verify: float = 1.0
    friction_cost_block: float = 10.0
    otp_effectiveness: float = 0.90

    @classmethod
    def from_config(cls, cfg: dict) -> "Costs":
        return cls(**{k: float(cfg["value"][k]) for k in cls.__dataclass_fields__})

    def scaled(self, name: str, factor: float) -> "Costs":
        """Copia con un parámetro multiplicado por `factor` (para el análisis de sensibilidad)."""
        d = asdict(self)
        d[name] = d[name] * factor
        if name == "otp_effectiveness":
            d[name] = min(d[name], 1.0)
        return Costs(**d)


def fraud_loss(amt: np.ndarray, costs: Costs) -> np.ndarray:
    """Pérdida si un fraude pasa: el monto (las devoluciones cuentan 0) más el costo del contracargo."""
    return np.maximum(amt.astype("float64"), 0.0) + costs.chargeback_fee


def gains(y: np.ndarray, amt: np.ndarray, costs: Costs) -> tuple[np.ndarray, np.ndarray]:
    """Ganancia por transacción (USD) si se BLOQUEA y si se VERIFICA, frente a aprobarla."""
    is_fraud = y.astype(bool)
    loss = fraud_loss(amt, costs)
    g_block = np.where(is_fraud, loss, -costs.friction_cost_block)
    g_verify = np.where(is_fraud, costs.otp_effectiveness * loss, -costs.friction_cost_verify) \
        - costs.verification_cost
    return g_block, g_verify


def decide(score: np.ndarray, t_verify: float, t_block: float) -> np.ndarray:
    d = np.full(len(score), APPROVE, dtype="int8")
    d[score >= t_verify] = VERIFY
    d[score >= t_block] = BLOCK
    return d


def value_per_tx(decision: np.ndarray, y: np.ndarray, amt: np.ndarray, costs: Costs,
                 weight: np.ndarray | None = None) -> np.ndarray:
    """Valor en USD que aporta cada transacción dada su decisión."""
    g_block, g_verify = gains(y, amt, costs)
    v = np.where(decision == BLOCK, g_block, np.where(decision == VERIFY, g_verify, 0.0))
    return v if weight is None else v * weight


def value_of_decisions(decision, y, amt, costs: Costs, weight=None) -> float:
    return float(value_per_tx(decision, y, amt, costs, weight).sum())


def oracle_value(y, amt, costs: Costs, weight=None) -> float:
    """Valor máximo posible: bloquear exactamente los fraudes."""
    v = np.where(y.astype(bool), fraud_loss(amt, costs), 0.0)
    return float((v if weight is None else v * weight).sum())


def best_thresholds(score, y, amt, costs: Costs, weight=None) -> dict:
    """Umbrales (t_verify, t_block) que maximizan el valor. Búsqueda exacta en O(n log n).

    Con las transacciones ordenadas de mayor a menor score, bloquear las primeras kb y verificar
    las siguientes hasta kv da:  valor(kb, kv) = B[kb] + V[kv] − V[kb],  donde B y V son las sumas
    acumuladas de la ganancia por bloquear y por verificar.
    """
    score = np.asarray(score, dtype="float64")
    order = np.argsort(-score, kind="stable")
    s = score[order]
    g_block, g_verify = gains(y[order], amt[order], costs)
    if weight is not None:
        g_block, g_verify = g_block * weight[order], g_verify * weight[order]
    B = np.r_[0.0, np.cumsum(g_block)]
    V = np.r_[0.0, np.cumsum(g_verify)]
    # Solo se puede cortar entre scores distintos.
    cut = np.r_[True, s[1:] != s[:-1], True]
    Vm = np.where(cut, V, -np.inf)
    # Mejor kv >= kb para cada kb (máximo por sufijo).
    rev = Vm[::-1]
    run = np.maximum.accumulate(rev)
    is_new = np.r_[True, rev[1:] > run[:-1]]
    idx_rev = np.maximum.accumulate(np.where(is_new, np.arange(len(rev)), 0))
    suffix_arg = (len(Vm) - 1 - idx_rev)[::-1]
    total = np.where(cut, B - V + V[suffix_arg], -np.inf)
    kb = int(np.argmax(total))
    kv = int(suffix_arg[kb])

    def threshold(k: int) -> float:
        if k == 0:
            return float(np.nextafter(s[0], np.inf)) if len(s) else 1.0
        if k >= len(s):
            return float(s[-1])
        return float((s[k - 1] + s[k]) / 2.0)

    return {"t_verify": threshold(kv), "t_block": threshold(kb), "value": float(total[kb])}


def value_curve(score, y, amt, costs: Costs, weight=None, points: int = 200) -> dict:
    """Curva de valor de una política de un solo umbral (alertar = verificar) al variar el umbral."""
    order = np.argsort(-score, kind="stable")
    s = score[order]
    yy = y[order].astype("float64")
    w = np.ones(len(s)) if weight is None else weight[order]
    _, g_verify = gains(y[order], amt[order], costs)
    V = np.cumsum(g_verify * w)
    tp = np.cumsum(yy * w)
    alerts = np.cumsum(w)
    ks = np.unique(np.geomspace(1, len(s), points).astype("int64")) - 1
    return {
        "threshold": s[ks].tolist(),
        "alert_rate": (alerts[ks] / alerts[-1]).tolist(),
        "recall": (tp[ks] / max(tp[-1], 1e-12)).tolist(),
        "precision": (tp[ks] / alerts[ks]).tolist(),
        "value": V[ks].tolist(),
    }


def f1_threshold(score, y, weight=None) -> dict:
    """Umbral que maximiza F1 (punto de operación "técnico", para contrastarlo con el de valor)."""
    order = np.argsort(-score, kind="stable")
    s = score[order]
    yy = y[order].astype("float64")
    w = np.ones(len(s)) if weight is None else weight[order]
    tp = np.cumsum(yy * w)
    pred = np.cumsum(w)
    precision = tp / pred
    recall = tp / max(tp[-1], 1e-12)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    cut = np.r_[s[1:] != s[:-1], True]
    k = int(np.argmax(np.where(cut, f1, -1.0)))
    return {"threshold": float(s[k]), "f1": float(f1[k]), "precision": float(precision[k]),
            "recall": float(recall[k])}


def summarize(decision, y, amt, costs: Costs) -> dict:
    """Conteos y dinero de una política ya decidida (para la tabla de resultados)."""
    is_fraud = y.astype(bool)
    alert = decision != APPROVE
    loss = fraud_loss(amt, costs)
    tp, fp, fn = int((alert & is_fraud).sum()), int((alert & ~is_fraud).sum()), int((~alert & is_fraud).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "value_usd": value_of_decisions(decision, y, amt, costs),
        "oracle_usd": oracle_value(y, amt, costs),
        "fraud_total_usd": float(loss[is_fraud].sum()),
        "fraud_missed_usd": float(loss[is_fraud & ~alert].sum()),
        "n": int(len(y)), "n_fraud": int(is_fraud.sum()),
        "n_verify": int((decision == VERIFY).sum()), "n_block": int((decision == BLOCK).sum()),
        "tp": tp, "fp": fp, "fn": fn,
        "fp_verify": int(((decision == VERIFY) & ~is_fraud).sum()),
        "fp_block": int(((decision == BLOCK) & ~is_fraud).sum()),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "alert_rate": float(alert.mean()),
    }


def bootstrap_ci(per_tx_value: np.ndarray, groups: np.ndarray, n_boot: int = 500, seed: int = 0) -> dict:
    """Intervalo de confianza del 95 % remuestreando CLIENTES (el fraude llega en ráfagas por tarjeta,
    así que remuestrear transacciones sueltas subestimaría la incertidumbre)."""
    _, inv = np.unique(groups, return_inverse=True)
    per_group = np.bincount(inv, weights=per_tx_value)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_group), size=(n_boot, len(per_group)))
    totals = per_group[idx].sum(axis=1)
    return {"low": float(np.percentile(totals, 2.5)), "high": float(np.percentile(totals, 97.5)),
            "std": float(totals.std())}


def sensitivity(score_val, y_val, amt_val, score_test, y_test, amt_test, costs: Costs,
                factors: list[float], fixed_thresholds: dict) -> list[dict]:
    """¿Qué pasa si un supuesto está equivocado en ±20 % o ±50 %?

    Para cada parámetro y factor reporta el valor en prueba (a) si se mantienen los umbrales
    elegidos con los supuestos originales y (b) si se vuelven a optimizar con el supuesto corregido.
    """
    rows = []
    for name in Costs.__dataclass_fields__:
        for f in factors:
            c = costs.scaled(name, 1.0 + f)
            fixed = value_of_decisions(decide(score_test, fixed_thresholds["t_verify"], fixed_thresholds["t_block"]),
                                       y_test, amt_test, c)
            th = best_thresholds(score_val, y_val, amt_val, c)
            reopt = value_of_decisions(decide(score_test, th["t_verify"], th["t_block"]), y_test, amt_test, c)
            rows.append({"param": name, "change": f, "value_fixed_thresholds": fixed,
                         "value_reoptimized": reopt, "t_verify": th["t_verify"], "t_block": th["t_block"]})
    return rows
