"""Evaluación: métricas técnicas, valor de negocio, incertidumbre, sensibilidad, latencia y figuras."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from . import runs, schema as S
from .data import Processed
from .value import (Costs, best_thresholds, bootstrap_ci, decide, f1_threshold, sensitivity, summarize,
                    value_curve, value_of_decisions, value_per_tx)

# Iteraciones en orden. (nombre interno, etiqueta, descripción del cambio)
ITERATIONS = [
    ("rules", "0 · Reglas fijas", "Proceso actual: reglas escritas a mano, sin modelo"),
    ("hgb", "1 · Gradient boosting", "Alternativa clásica con las mismas variables y pesos que el modelo A"),
    ("mlp_loc", "2 · Modelo A0 (MLP)", "Red densa, pérdida estándar (BCE), con embedding del país o estado del comercio"),
    ("mlp_plain", "3 · Modelo A1 (MLP)", "Se quita la identidad de la ubicación; quedan las señales genéricas"),
    ("mlp_cost", "4 · Modelo A (MLP)", "Igual que A1, con pérdida ponderada por el costo del error"),
    ("lstm_cost", "5 · Modelo B (LSTM)", "LSTM sobre las últimas 16 transacciones de la tarjeta, pérdida por costo"),
]
SERVING = {"A": "mlp_cost", "B": "lstm_cost"}
PARAM_LABELS = {
    "chargeback_fee": "Costo del contracargo",
    "verification_cost": "Costo de verificar",
    "friction_cost_verify": "Fricción al verificar a un legítimo",
    "friction_cost_block": "Fricción al bloquear a un legítimo",
    "otp_effectiveness": "Efectividad de la verificación",
}


# ------------------------------------------------------------------ latencia
@torch.no_grad()
def measure_latency(model, data: Processed, reps: int = 300) -> dict:
    """Latencia de UNA transacción en CPU (como en el servicio) y throughput por lotes."""
    model = model.to("cpu").eval()
    idx = data.split_idx("test")[:4096]
    threads = torch.get_num_threads()
    torch.set_num_threads(1)                         # igual que el servicio: un hilo por transacción

    def batch(ix):
        if model.kind == "mlp":
            return [torch.from_numpy(a) for a in data.tabular(ix)]
        return [torch.from_numpy(a) for a in data.sequences(ix)]

    one = batch(idx[:1])
    for _ in range(20):
        model(*one)
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        model(*one)
        times.append(time.perf_counter() - t)
    big = batch(idx)
    t = time.perf_counter()
    model(*big)
    dt = time.perf_counter() - t
    torch.set_num_threads(threads)
    times = np.array(times) * 1000
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
            "throughput_tx_s": float(len(idx) / dt)}


def measure_latency_sklearn(model, data: Processed, reps: int = 100) -> dict:
    from .baselines import _matrix
    idx = data.split_idx("test")[:4096]
    X = _matrix(data, idx, getattr(model, "fraudscan_cat_cols", None))
    times = []
    for _ in range(reps):
        t = time.perf_counter()
        model.predict_proba(X[:1])
        times.append(time.perf_counter() - t)
    t = time.perf_counter()
    model.predict_proba(X)
    dt = time.perf_counter() - t
    times = np.array(times) * 1000
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
            "throughput_tx_s": float(len(idx) / dt)}


# ------------------------------------------------------------------ evaluación de una iteración
def evaluate_run(cfg: dict, name: str, data: Processed, costs: Costs, latency: dict | None) -> dict:
    """Elige el punto de operación en VALIDACIÓN y reporta todo en PRUEBA."""
    val_idx, test_idx = data.split_idx("val"), data.split_idx("test")
    y_val, amt_val = data.y[val_idx], data.amt[val_idx]
    y_test, amt_test = data.y[test_idx], data.amt[test_idx]
    s_val, s_test = runs.load_scores(cfg, name, "val"), runs.load_scores(cfg, name, "test")
    months = 12 * len(cfg["split"]["test_years"])
    out: dict = {"name": name}

    if name == "rules":                       # las reglas ya son decisiones, no scores
        d_val, d_test = s_val, s_test
    else:
        th = best_thresholds(s_val, y_val, amt_val, costs)
        f1 = f1_threshold(s_val, y_val)
        d_val = decide(s_val, th["t_verify"], th["t_block"])
        d_test = decide(s_test, th["t_verify"], th["t_block"])
        d_f1 = decide(s_test, f1["threshold"], np.inf)          # alertar (verificar) con el umbral de máximo F1
        out.update({
            "thresholds": {"t_verify": th["t_verify"], "t_block": th["t_block"]},
            "f1_threshold": f1["threshold"],
            "ap_val": float(average_precision_score(y_val, s_val)),
            "ap_test": float(average_precision_score(y_test, s_test)),
            "auc_val": float(roc_auc_score(y_val, s_val)),
            "auc_test": float(roc_auc_score(y_test, s_test)),
            "at_f1_threshold": summarize(d_f1, y_test, amt_test, costs),
        })

    out["val"] = summarize(d_val, y_val, amt_val, costs)
    out["test"] = summarize(d_test, y_test, amt_test, costs)
    # Desglose por año: muestra si el modelo aguanta cuando cambia el patrón de fraude (deriva).
    out["by_year"] = {}
    for idx, d, y_, a_ in ((val_idx, d_val, y_val, amt_val), (test_idx, d_test, y_test, amt_test)):
        for year in np.unique(data.year[idx]):
            m = data.year[idx] == year
            s = summarize(d[m], y_[m], a_[m], costs)
            out["by_year"][int(year)] = {k: s[k] for k in ("n_fraud", "recall", "precision", "value_usd", "oracle_usd")}
    per_tx = value_per_tx(d_test, y_test, amt_test, costs)
    out["test"]["value_ci95"] = bootstrap_ci(per_tx, data.user[test_idx], cfg["value"]["bootstrap_samples"])

    # Costo de operación: latencia por transacción x precio del cómputo.
    lat = latency or {"p50_ms": 0.0, "p95_ms": 0.0, "throughput_tx_s": float("inf")}
    cost_per_pred = lat["p50_ms"] / 1000 * cfg["value"]["vcpu_price_per_second"]
    op_cost = cost_per_pred * len(y_test)
    out["latency"] = lat
    out["cost_per_1k_predictions_usd"] = cost_per_pred * 1000
    out["test"]["operation_cost_usd"] = op_cost
    out["test"]["net_value_usd"] = out["test"]["value_usd"] - op_cost
    out["test"]["net_value_per_month_usd"] = out["test"]["net_value_usd"] / months
    out["test"]["net_value_per_million_tx_usd"] = out["test"]["net_value_usd"] / len(y_test) * 1e6
    out["test"]["pct_of_oracle"] = out["test"]["value_usd"] / out["test"]["oracle_usd"]
    out["val"]["pct_of_oracle"] = out["val"]["value_usd"] / out["val"]["oracle_usd"]
    return out


def train_metrics(model_kind: str, model, data: Processed, cfg: dict) -> dict:
    """AP en la muestra de entrenamiento (para ver sobreajuste frente a validación y prueba)."""
    from . import baselines, train
    idx = data.train_sample(cfg["train"]["neg_per_pos"], cfg["train"]["seed"])
    if len(idx) > 400_000:
        idx = np.sort(np.random.default_rng(0).choice(idx, 400_000, replace=False))
    s = baselines.predict_hgb(model, data, idx) if model_kind == "hgb" else train.predict(model, data, idx, "cpu")
    return {"ap_train_sample": float(average_precision_score(data.y[idx], s))}


# ------------------------------------------------------------------ tabla de iteraciones
def iterations_table(results: list[dict], run_infos: dict) -> tuple[list[dict], str]:
    rows = []
    for name, label, change in ITERATIONS:
        r = next((x for x in results if x["name"] == name), None)
        if r is None:
            continue
        info = run_infos.get(name) or {}
        t = r["test"]
        rows.append({
            "iteracion": label, "cambio": change,
            "ap_train": r.get("ap_train_sample"), "ap_val": r.get("ap_val"), "ap_test": r.get("ap_test"),
            "recall_test": t["recall"], "precision_test": t["precision"], "f1_test": t["f1"],
            "valor_test_usd": t["value_usd"], "valor_ic95_bajo": t["value_ci95"]["low"],
            "valor_ic95_alto": t["value_ci95"]["high"], "pct_maximo": t["pct_of_oracle"],
            "valor_neto_mes_usd": t["net_value_per_month_usd"],
            "fraude_no_detectado_usd": t["fraud_missed_usd"],
            "verificaciones": t["n_verify"], "bloqueos": t["n_block"],
            "legitimas_molestadas": t["fp"],
            "entrenamiento_s": info.get("train_seconds"), "dispositivo": info.get("device"),
            "latencia_p50_ms": r["latency"]["p50_ms"], "costo_1k_pred_usd": r["cost_per_1k_predictions_usd"],
        })

    def f(v, fmt):
        return "–" if v is None else format(v, fmt)

    head = ("| Iteración | Cambio | AP train | AP val | AP test | Recall | Precisión | Valor en prueba (USD) | "
            "IC 95 % | % del máximo | Valor neto / mes | Entrenamiento | Latencia p50 |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = [
        f"| {r['iteracion']} | {r['cambio']} | {f(r['ap_train'], '.3f')} | {f(r['ap_val'], '.3f')} | "
        f"{f(r['ap_test'], '.3f')} | {r['recall_test']:.3f} | {r['precision_test']:.3f} | "
        f"{r['valor_test_usd']:,.0f} | {r['valor_ic95_bajo']:,.0f} a {r['valor_ic95_alto']:,.0f} | "
        f"{100 * r['pct_maximo']:.1f} % | {r['valor_neto_mes_usd']:,.0f} | "
        f"{f(r['entrenamiento_s'], '.0f')} s ({r['dispositivo'] or '–'}) | {r['latencia_p50_ms']:.2f} ms |"
        for r in rows]
    return rows, head + "\n".join(lines) + "\n"


# ------------------------------------------------------------------ figuras
def make_figures(cfg: dict, data: Processed, costs: Costs, results: list[dict], run_infos: dict,
                 sens: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    test_idx = data.split_idx("test")
    y, amt = data.y[test_idx], data.amt[test_idx]
    by_name = {r["name"]: r for r in results}
    labels = {n: lab for n, lab, _ in ITERATIONS}
    colors = {"hgb": "#8a8a8a", "mlp_loc": "#c6dbef", "mlp_plain": "#9ecae1", "mlp_cost": "#3182bd",
              "lstm_cost": "#d95f0e", "rules": "#444444"}

    # 1) Curva de valor: valor de alertar según la fracción de transacciones alertadas.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name in ("hgb", "mlp_cost", "lstm_cost"):
        if name not in by_name:
            continue
        s = runs.load_scores(cfg, name, "test")
        c = value_curve(s, y, amt, costs, points=400)
        keep = np.array(c["alert_rate"]) <= 0.08          # más allá el valor solo cae (demasiadas falsas alarmas)
        ax.plot(np.array(c["alert_rate"])[keep] * 100, np.array(c["value"])[keep] / 1000, label=labels[name],
                color=colors[name])
        d_f1 = decide(s, by_name[name]["f1_threshold"], np.inf)
        ax.scatter([100 * (d_f1 != S.APPROVE).mean()], [value_of_decisions(d_f1, y, amt, costs) / 1000],
                   marker="x", color=colors[name], zorder=3)
        k = int(np.argmax(c["value"]))
        ax.scatter([c["alert_rate"][k] * 100], [c["value"][k] / 1000], marker="o", color=colors[name], zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Transacciones alertadas (%) - escala logarítmica")
    ax.set_ylabel("Valor en prueba (miles de USD)")
    ax.set_title("Curva de valor al mover el umbral de alerta (círculo: óptimo por valor; cruz: óptimo por F1)",
                 fontsize=10)
    ax.axhline(0, color="#bbbbbb", lw=0.8)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "curva_de_valor.png", dpi=150)
    plt.close(fig)

    # 2) Precisión-recall.
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    for name in ("hgb", "mlp_loc", "mlp_plain", "mlp_cost", "lstm_cost"):
        if name not in by_name:
            continue
        p, r, _ = precision_recall_curve(y, runs.load_scores(cfg, name, "test"))
        step = max(1, len(p) // 2000)
        ax.plot(r[::step], p[::step], label=f"{labels[name]} (AP {by_name[name]['ap_test']:.3f})", color=colors[name])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precisión")
    ax.set_title("Curva precisión-recall en prueba")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "precision_recall.png", dpi=150)
    plt.close(fig)

    # 3) Curvas de entrenamiento.
    dl = [n for n in ("mlp_loc", "mlp_plain", "mlp_cost", "lstm_cost") if (run_infos.get(n) or {}).get("history")]
    if dl:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for name in dl:
            h = run_infos[name]["history"]
            ep = [x["epoch"] for x in h]
            axes[0].plot(ep, [x["train_loss"] for x in h], "--", color=colors[name])
            axes[0].plot(ep, [x["val_loss"] for x in h], "-", color=colors[name], label=labels[name])
            axes[1].plot(ep, [100 * x["val_value_pct"] for x in h], "-o", color=colors[name], label=labels[name])
        axes[0].set_yscale("log")
        axes[0].set_title("Pérdida (línea punteada: train; continua: validación)")
        axes[0].set_xlabel("Época")
        axes[1].set_title("Valor en validación (% del máximo posible)")
        axes[1].set_xlabel("Época")
        for a in axes:
            a.grid(alpha=0.25)
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "curvas_de_entrenamiento.png", dpi=150)
        plt.close(fig)

    # 4) Valor por iteración con intervalo de confianza.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [n for n, _, _ in ITERATIONS if n in by_name]
    vals = np.array([by_name[n]["test"]["value_usd"] for n in names]) / 1000
    lo = np.array([by_name[n]["test"]["value_ci95"]["low"] for n in names]) / 1000
    hi = np.array([by_name[n]["test"]["value_ci95"]["high"] for n in names]) / 1000
    ax.bar(range(len(names)), vals, color=[colors[n] for n in names],
           yerr=[np.maximum(vals - lo, 0), np.maximum(hi - vals, 0)], capsize=4)
    ax.axhline(by_name[names[0]]["test"]["oracle_usd"] / 1000, color="#2ca25f", ls="--", lw=1,
               label="Máximo posible (detector perfecto)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([labels[n] for n in names], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Valor en prueba (miles de USD)")
    ax.set_title("Valor de negocio por iteración (IC 95 % por clientes)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "valor_por_iteracion.png", dpi=150)
    plt.close(fig)

    # 5) Sensibilidad.
    for key, rows in sens.items():
        params = list(dict.fromkeys(r["param"] for r in rows))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        base = by_name[SERVING[key]]["test"]["value_usd"]
        for p in params:
            pr = sorted([r for r in rows if r["param"] == p], key=lambda r: r["change"])
            xs = [r["change"] * 100 for r in pr]
            xs.insert(len([x for x in xs if x < 0]), 0)
            ys = [r["value_reoptimized"] / 1000 for r in pr]
            ys.insert(len([x for x in xs if x < 0]), base / 1000)
            ax.plot(xs, ys, "-o", label=PARAM_LABELS.get(p, p))
        ax.set_xlabel("Error en el supuesto (%)")
        ax.set_ylabel("Valor en prueba (miles de USD)")
        ax.set_title(f"Sensibilidad del valor a los supuestos - modelo {key}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"sensibilidad_modelo_{key}.png", dpi=150)
        plt.close(fig)


def run_sensitivity(cfg: dict, data: Processed, costs: Costs, results: list[dict]) -> dict:
    val_idx, test_idx = data.split_idx("val"), data.split_idx("test")
    out = {}
    for key, name in SERVING.items():
        r = next((x for x in results if x["name"] == name), None)
        if r is None:
            continue
        out[key] = sensitivity(
            runs.load_scores(cfg, name, "val"), data.y[val_idx], data.amt[val_idx],
            runs.load_scores(cfg, name, "test"), data.y[test_idx], data.amt[test_idx],
            costs, cfg["value"]["sensitivity"], r["thresholds"])
    return out


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
