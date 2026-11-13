"""Paso 5: evalúa todas las iteraciones. Umbrales elegidos por valor en validación; resultados en prueba.

Genera reports/metrics.json, reports/iteraciones.md, reports/iteraciones.csv, las figuras y
artifacts/serving.json (umbrales y costos que usa el servicio)."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib

from fraudscan import evaluate as E
from fraudscan import runs
from fraudscan.config import base_parser, load_config
from fraudscan.data import Processed
from fraudscan.models import load_model
from fraudscan.value import Costs

if __name__ == "__main__":
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.profile)
    data, costs = Processed(cfg["data"]["processed_dir"]), Costs.from_config(cfg)
    art, rep = Path(cfg["data"]["artifacts_dir"]), Path(cfg["data"]["reports_dir"])

    results, run_infos = [], {}
    for name, label, _ in E.ITERATIONS:
        if not runs.scores_path(cfg, name, "test").exists():
            print(f"(sin resultados para {label}; se omite)")
            continue
        print(f"Evaluando {label}")
        run_infos[name] = runs.load_run(cfg, name)
        latency, extra = None, {}
        if name == "hgb" and (art / "hgb.joblib").exists():
            model = joblib.load(art / "hgb.joblib")
            latency = E.measure_latency_sklearn(model, data)
            extra = E.train_metrics("hgb", model, data, cfg)
        elif (art / f"{name}.pt").exists():
            model = load_model(art / f"{name}.pt")
            latency = E.measure_latency(model, data)
            extra = E.train_metrics(model.kind, model, data, cfg)
        r = E.evaluate_run(cfg, name, data, costs, latency)
        r.update(extra)
        results.append(r)
        t = r["test"]
        print(f"  valor en prueba: {t['value_usd']:,.0f} USD ({100 * t['pct_of_oracle']:.1f} % del máximo) | "
              f"recall {t['recall']:.3f} | precisión {t['precision']:.3f} | "
              f"verificaciones {t['n_verify']:,} | bloqueos {t['n_block']:,}")
        print("  por año: " + " | ".join(f"{y}: recall {v['recall']:.2f}, {100 * v['value_usd'] / max(v['oracle_usd'], 1):.0f} % del máx."
                                         for y, v in r["by_year"].items()))

    print("Análisis de sensibilidad")
    sens = E.run_sensitivity(cfg, data, costs, results)
    rows, table_md = E.iterations_table(results, run_infos)

    by_name = {r["name"]: r for r in results}
    comparison = None
    if "mlp_cost" in by_name and "lstm_cost" in by_name:
        a, b = by_name["mlp_cost"]["test"], by_name["lstm_cost"]["test"]
        months = 12 * len(cfg["split"]["test_years"])
        comparison = {
            "extra_value_per_month_usd": (b["value_usd"] - a["value_usd"]) / months,
            "extra_compute_per_month_usd": (b["operation_cost_usd"] - a["operation_cost_usd"]) / months,
            "extra_latency_ms": by_name["lstm_cost"]["latency"]["p50_ms"] - by_name["mlp_cost"]["latency"]["p50_ms"],
        }
        print(f"B frente a A: {comparison['extra_value_per_month_usd']:+,.0f} USD/mes de valor, "
              f"{comparison['extra_compute_per_month_usd']:+.4f} USD/mes de cómputo, "
              f"{comparison['extra_latency_ms']:+.2f} ms de latencia")

    E.write_json(rep / "metrics.json", {"profile": cfg["profile"], "costs": costs.__dict__, "split": cfg["split"],
                                        "counts": data.meta["counts"], "results": results,
                                        "sensitivity": sens, "comparison_B_vs_A": comparison})
    (rep / "iteraciones.md").write_text(table_md, encoding="utf-8")
    with open(rep / "iteraciones.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    E.make_figures(cfg, data, costs, results, run_infos, sens, rep / "figuras")

    # Lo que necesita el servicio para decidir.
    serving = {"costs": costs.__dict__, "profile": cfg["profile"], "models": {}, "iterations": rows,
               "comparison_B_vs_A": comparison}
    for key, name in E.SERVING.items():
        if name in by_name:
            r = by_name[name]
            serving["models"][key] = {
                "file": f"{name}.pt", "name": name, "thresholds": r["thresholds"],
                "test": {k: r["test"][k] for k in ("value_usd", "pct_of_oracle", "recall", "precision",
                                                   "net_value_per_month_usd")},
                "latency": r["latency"]}
    E.write_json(art / "serving.json", serving)
    print("\n" + table_md)
    print(f"Reportes en {rep} y umbrales en {art / 'serving.json'}")
