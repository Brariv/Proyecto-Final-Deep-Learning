"""Paso 8 (Laboratorio 10): explicabilidad del modelo B con Integrated Gradients.

Usa el modelo ya entrenado (artifacts/lstm_cost.pt). No reentrena nada.

Genera en reports/explicabilidad/:
  figuras/casos_locales.png      seis casos del conjunto de prueba, explicados uno por uno
  figuras/global_variables.png   importancia agregada por variable, fraude frente a legítimas
  figuras/global_pasos.png       cuánto aporta cada compra previa de la tarjeta
  figuras/validacion.png         IG frente a oclusión, y sensibilidad a la línea base
  casos_locales.csv/.md          los números de cada caso
  global_variables.csv           importancia media por variable
  resumen.json                   parámetros, costo, completitud y resultados de validación

Todo depende de una sola semilla (--seed, por defecto 42).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from fraudscan import runs
from fraudscan import schema as S
from fraudscan.config import base_parser, load_config
from fraudscan.data import Processed
from fraudscan.explain import Explainer, label, spearman
from fraudscan.models import load_model
from fraudscan.service import HISTORY_REASON, REASON_GROUPS
from fraudscan.value import Costs, decide

MODEL_FILE = "lstm_cost.pt"
N_STEPS = 256           # pasos de la integral para la vista global
N_STEPS_LOCAL = 512     # los casos con score cercano a 1 saturan la sigmoide y piden más pasos
N_GLOBAL = 2000         # transacciones por grupo (fraude y legítimas) para la vista global
N_COST = 256            # transacciones para medir el costo de una explicación
TOP_LOCAL = 8           # contribuciones que se muestran por caso
TOP_GLOBAL = 16         # variables que se muestran en la figura global

ROJO, AZUL, GRIS = "#c0392b", "#2d6cb3", "#8a8a8a"


# ------------------------------------------------------------------ selección de casos
def pick_cases(data, score, test, th, rng):
    """Seis casos del conjunto de prueba: un acierto claro, dos errores, un caso en el borde y
    dos de subgrupos donde ya sabíamos que el modelo se comporta distinto."""
    y, amt = data.y[test], data.amt[test]
    chan = data.X_cat[test, 0]
    online = chan == S.CHANNELS.index("Online Transaction")
    foreign = np.asarray(data.X_num[test, S.NUM_INDEX["is_foreign"]]) > 0.5
    fraud, legit = y == 1, y == 0
    flagged = score >= th["t_verify"]
    blocked = score >= th["t_block"]

    def one(mask, key=None, name=""):
        pool = np.flatnonzero(mask)
        if len(pool) == 0:
            raise SystemExit(f"No hay candidatos para el caso: {name}")
        if key is None:
            return int(rng.choice(pool))
        return int(pool[np.argmin(key[pool])])

    cases = [
        ("Acierto con alta confianza", "Fraude bloqueado con el score más alto del periodo",
         one(fraud & blocked, -score, "acierto")),
        ("Error: fraude no detectado", "Fraude que el modelo aprobó, el error que cuesta dinero",
         one(fraud & ~flagged, -amt, "falso negativo")),
        ("Error: compra legítima bloqueada", "Compra buena detenida, el error que molesta al cliente",
         one(legit & blocked, -amt, "falso positivo")),
        ("Caso en el borde del umbral", "Score pegado al umbral de verificación",
         one(fraud, np.abs(score - th["t_verify"]), "borde")),
        ("Subgrupo: compra en línea legítima", "Compra en línea buena que el modelo marcó",
         one(legit & online & flagged, -score, "en línea")),
        ("Subgrupo: fraude en el extranjero", "La campaña de fraude que domina 2018-2019",
         one(fraud & foreign & flagged, -amt, "extranjero")),
    ]
    return [(t, d, test[i], i) for t, d, i in cases]


# ------------------------------------------------------------------ figuras
def fig_local(ex, data, cases, res, th, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats = ex.by_feature(res["attr"], step=-1)
    hist = res["attr"][:, :-1].sum(axis=(1, 2))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
    for k, ((title, desc, row, _), ax) in enumerate(zip(cases, axes.ravel())):
        contrib = {label(n): float(v[k]) for n, v in feats.items()}
        contrib[HISTORY_REASON] = float(hist[k])
        top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:TOP_LOCAL][::-1]
        names = [n for n, _ in top]
        vals = np.array([v for _, v in top]) * 100          # en puntos de score
        ax.barh(range(len(top)), vals, color=[ROJO if v > 0 else AZUL for v in vals])
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(0, color="#444444", lw=0.8)
        s, real = res["score"][k], data.y[row]
        dec = ["aprobar", "verificar", "bloquear"][int(decide(np.array([s]), th["t_verify"], th["t_block"])[0])]
        ax.set_title(f"{title}\nscore {100 * s:.1f} · {dec} · real: {'fraude' if real else 'legítima'} · "
                     f"${abs(float(data.amt[row])):,.0f}", fontsize=9, loc="left")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Integrated Gradients por caso: aporte de cada señal al score (puntos sobre 100)",
                 fontsize=11)
    fig.supxlabel("Rojo: sube el score. Azul: lo baja.", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out / "casos_locales.png", dpi=150)
    plt.close(fig)


def mean_abs(feats: dict) -> np.ndarray:
    return np.array([np.abs(v).mean() for v in feats.values()]) * 100


def fig_global(groups, out):
    """Dos paneles con escala propia: qué mueve un bloqueo correcto y qué mueve uno equivocado."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("Fraudes detectados", "fraudes_detectados", ROJO),
              ("Compras legítimas bloqueadas por error", "falsos_positivos", AZUL)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, (title, key, color) in zip(axes, panels):
        feats = groups[key]
        names = list(feats)
        imp = mean_abs(feats)
        order = np.argsort(-imp)[:TOP_GLOBAL][::-1]
        ax.barh(np.arange(len(order)), imp[order], color=color)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([label(names[i]) for i in order], fontsize=8.5)
        ax.set_xlabel("Atribución media absoluta (puntos de score)")
        ax.set_title(f"{title} (n = {len(next(iter(feats.values()))):,})", fontsize=10)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Importancia global por variable en la compra que se autoriza", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "global_variables.png", dpi=150)
    plt.close(fig)


def fig_steps(step_fraud, step_legit, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = step_fraud.shape[1]
    pos = np.arange(k)
    labels = ["actual"] + [f"-{i}" for i in range(1, k)]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.bar(pos - 0.2, np.abs(step_fraud).mean(axis=0)[::-1] * 100, width=0.4, color=ROJO, label="Fraudes")
    ax.bar(pos + 0.2, np.abs(step_legit).mean(axis=0)[::-1] * 100, width=0.4, color=GRIS, label="Compras legítimas")
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Posición en la secuencia de la tarjeta (0 = compra que se está autorizando)")
    ax.set_ylabel("Atribución media\nabsoluta (puntos)")
    ax.set_yscale("log")
    ax.set_title("Cuánto pesa cada compra previa de la tarjeta", fontsize=11)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "global_pasos.png", dpi=150)
    plt.close(fig)


def fig_validation(group_ig, occ, rho_group, feats_a, feats_b, rho_base, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.scatter(occ.ravel() * 100, group_ig.ravel() * 100, s=6, alpha=0.3, color=AZUL, edgecolors="none")
    lim = max(np.abs(occ).max(), np.abs(group_ig).max()) * 100
    ax.plot([-lim, lim], [-lim, lim], color="#444444", lw=0.8, ls="--")
    ax.set_xlabel("Oclusión: caída del score (puntos)")
    ax.set_ylabel("Integrated Gradients (puntos)")
    ax.set_title(f"Los dos métodos coinciden\nSpearman por caso: {rho_group:.2f}", fontsize=10)
    ax.grid(alpha=0.25)

    ax = axes[1]
    names = list(feats_a)
    a = np.array([np.abs(feats_a[n]).mean() for n in names]) * 100
    b = np.array([np.abs(feats_b[n]).mean() for n in names]) * 100
    ax.scatter(a, b, s=18, color=ROJO, edgecolors="none")
    top = np.argsort(-b)[:5]
    spread = np.linspace(0.97, 0.45, len(top))          # separa las etiquetas en vertical
    for rank, i in enumerate(top):
        ax.annotate(label(names[i]), xy=(a[i], b[i]), xytext=(0.40, spread[rank]),
                    textcoords="axes fraction", fontsize=7, va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#999999"))
    lim = max(a.max(), b.max()) * 1.08
    ax.plot([0, lim], [0, lim], color="#444444", lw=0.8, ls="--")
    ax.set_xlabel("Línea base: secuencia neutra")
    ax.set_ylabel("Línea base: solo la compra actual")
    ax.set_title(f"La línea base cambia la escala, no el orden\nSpearman: {rho_base:.3f}", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "validacion.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ baseline interpretable
def interpretable_baselines(data, cfg, costs, score_lstm, test, th):
    """¿Un modelo interpretable por diseño alcanza al modelo B? Se entrena aparte, no toca al
    modelo final."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    from fraudscan.baselines import _matrix
    from fraudscan.train import cost_weights
    from fraudscan.value import best_thresholds, summarize

    tc = cfg["train"]
    use = [data.meta["cat_features"].index(n) for n in cfg["features"]["cat_use"]]
    tr = data.train_sample(tc["neg_per_pos"], tc["seed"])
    if len(tr) > 400_000:
        tr = np.sort(np.random.default_rng(tc["seed"]).choice(tr, 400_000, replace=False))
    val = data.split_idx("val")
    val = np.sort(np.random.default_rng(tc["seed"]).choice(val, 400_000, replace=False))
    Xtr, ytr = _matrix(data, tr, use), data.y[tr]
    w = cost_weights(ytr, data.amt[tr], costs, tuple(tc["weight_clip"]))
    sc = StandardScaler().fit(Xtr)
    out = {}
    for name, model in (("Regresión logística", LogisticRegression(max_iter=400, C=1.0)),
                        ("Árbol de decisión (prof. 3)", DecisionTreeClassifier(max_depth=3, random_state=tc["seed"]))):
        Xt = sc.transform(Xtr) if "logística" in name else Xtr
        model.fit(Xt, ytr, sample_weight=w)
        f = (lambda X: model.predict_proba(sc.transform(X))[:, 1]) if "logística" in name \
            else (lambda X: model.predict_proba(X)[:, 1])
        s_val = f(_matrix(data, val, use))
        t = best_thresholds(s_val, data.y[val], data.amt[val], costs)
        s_test = np.concatenate([f(_matrix(data, test[i:i + 400_000], use))
                                 for i in range(0, len(test), 400_000)])
        r = summarize(decide(s_test, t["t_verify"], t["t_block"]), data.y[test], data.amt[test], costs)
        out[name] = {"value_usd": r["value_usd"], "pct_of_oracle": r["value_usd"] / r["oracle_usd"],
                     "recall": r["recall"], "precision": r["precision"]}
    r = summarize(decide(score_lstm, th["t_verify"], th["t_block"]), data.y[test], data.amt[test], costs)
    out["Modelo B (LSTM)"] = {"value_usd": r["value_usd"], "pct_of_oracle": r["value_usd"] / r["oracle_usd"],
                              "recall": r["recall"], "precision": r["precision"]}
    return out


# ------------------------------------------------------------------ principal
if __name__ == "__main__":
    parser = base_parser(__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Omite el entrenamiento de los modelos interpretables de comparación")
    args = parser.parse_args()
    cfg = load_config(args.config, args.profile)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = Processed(cfg["data"]["processed_dir"])
    art = Path(cfg["data"]["artifacts_dir"])
    out = Path(cfg["data"]["reports_dir"]) / "explicabilidad"
    (out / "figuras").mkdir(parents=True, exist_ok=True)
    with open(art / "vocab.json", encoding="utf-8") as f:
        vocab_meta = json.load(f)
    with open(art / "serving.json", encoding="utf-8") as f:
        th = json.load(f)["models"]["B"]["thresholds"]
    costs = Costs.from_config(cfg)
    ex = Explainer(load_model(art / MODEL_FILE), data.meta, vocab_meta)
    test = data.split_idx("test")
    score = runs.load_scores(cfg, "lstm_cost", "test")
    print(f"Modelo B sobre {len(test):,} transacciones de prueba; umbrales "
          f"verificar {th['t_verify']:.4f}, bloquear {th['t_block']:.4f}")

    # --- 1. casos locales
    cases = pick_cases(data, score, test, th, rng)
    rows = np.array([r for _, _, r, _ in cases])
    err = ex.verify_split(data, rows)
    res_local = ex.attribute(data, rows, "neutral", n_steps=N_STEPS_LOCAL)
    print(f"Corte modelo/codificador exacto (error {err:.1e}); "
          f"completitud: error máximo {np.abs(res_local['delta']).max():.1e}")
    fig_local(ex, data, cases, res_local, th, out / "figuras")

    feats_local = ex.by_feature(res_local["attr"], step=-1)
    hist_local = res_local["attr"][:, :-1].sum(axis=(1, 2))
    local_table = []
    for k, (title, desc, row, _) in enumerate(cases):
        contrib = {label(n): float(v[k]) for n, v in feats_local.items()}
        contrib[HISTORY_REASON] = float(hist_local[k])
        top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:3]
        local_table.append({
            "caso": title, "descripcion": desc, "fila": int(row),
            "score": float(res_local["score"][k]), "real": "fraude" if data.y[row] else "legítima",
            "decision": ["aprobar", "verificar", "bloquear"][
                int(decide(res_local["score"][k:k + 1], th["t_verify"], th["t_block"])[0])],
            "monto_usd": float(data.amt[row]),
            "aporte_historial_pts": 100 * float(hist_local[k]),
            "top_senales": "; ".join(f"{n} ({100 * v:+.1f})" for n, v in top),
            "senales": [{"etiqueta": n, "puntos": 100 * float(v)} for n, v in top],
        })
        print(f"  {title}: score {100 * res_local['score'][k]:.1f} | {top[0][0]} ({100 * top[0][1]:+.1f} pts)")

    # --- 2. vista global, separando aciertos de errores
    y_test, flagged = data.y[test], score >= th["t_verify"]
    pools = {
        "fraudes_detectados": (y_test == 1) & flagged,
        "falsos_positivos": (y_test == 0) & flagged,
        "falsos_negativos": (y_test == 1) & ~flagged,
        "legitimas_aprobadas": (y_test == 0) & ~flagged,
    }
    groups, sel, secs = {}, {}, 0.0
    for key, mask in pools.items():
        pool = np.flatnonzero(mask)
        sel[key] = test[np.sort(rng.choice(pool, min(N_GLOBAL, len(pool)), replace=False))]
        r = ex.attribute(data, sel[key], "neutral", n_steps=N_STEPS)
        groups[key] = ex.by_feature(r["attr"], step=-1)
        secs += r["seconds"]
        if key == "fraudes_detectados":
            res_f, step_f = r, ex.by_step(r["attr"])
        if key == "legitimas_aprobadas":
            step_l = ex.by_step(r["attr"])
        print(f"  {key:22s} n={len(sel[key]):>5,}  score medio {100 * r['score'].mean():5.1f}")
    fig_global(groups, out / "figuras")
    fig_steps(step_f, step_l, out / "figuras")
    print(f"Vista global en {secs:.0f}s")
    names = list(groups["fraudes_detectados"])
    imp = {k: mean_abs(v) for k, v in groups.items()}
    top_global = sorted(zip(names, imp["fraudes_detectados"], imp["falsos_positivos"],
                            imp["falsos_negativos"]), key=lambda t: -t[1])[:TOP_GLOBAL]
    for n, a, b, c in top_global[:6]:
        print(f"    {label(n):34s} fraude {a:6.2f} | falso positivo {b:6.2f} | falso negativo {c:6.2f}")
    sel_f = sel["fraudes_detectados"]

    # --- 3. validación: IG frente a oclusión, y sensibilidad a la línea base
    # El modelo final no usa la identidad de la ubicación, así que ese grupo se reduce a sus
    # variables numéricas (extranjero, fuera del estado, ubicación nueva).
    occ_groups = [(n, nums, [c for c in cats if c in ex.spans]) for n, nums, cats in REASON_GROUPS]
    sel_v = test[np.sort(rng.choice(np.flatnonzero(score >= th["t_verify"]), 300, replace=False))]
    res_v = ex.attribute(data, sel_v, "neutral", n_steps=N_STEPS)
    occ = ex.occlusion_groups(data, sel_v, occ_groups)
    feats_v = ex.by_feature(res_v["attr"], step=-1)
    group_ig = np.stack([sum(feats_v[n] for n in nums + cats) for _, nums, cats in occ_groups], axis=1)
    rho_group = float(np.mean([spearman(group_ig[i], occ["drop"][i]) for i in range(len(sel_v))]))
    res_c = ex.attribute(data, sel_f[:600], "current", n_steps=N_STEPS)
    feats_a = {k: v[:600] for k, v in groups["fraudes_detectados"].items()}
    feats_b = ex.by_feature(res_c["attr"], step=-1)
    rho_base = spearman(np.array([np.abs(feats_a[n]).mean() for n in feats_a]),
                        np.array([np.abs(feats_b[n]).mean() for n in feats_a]))
    fig_validation(group_ig, occ["drop"], rho_group, feats_a, feats_b, rho_base, out / "figuras")
    print(f"Validación: Spearman IG vs oclusión {rho_group:.2f} | línea base neutra vs actual {rho_base:.2f}")

    # --- 4. costo
    sel_cost = test[np.sort(rng.choice(len(test), N_COST, replace=False))]
    t_batch = ex.attribute(data, sel_cost, "neutral", n_steps=N_STEPS, batch_size=64)["seconds"]
    t_one = ex.attribute(data, sel_cost[:32], "neutral", n_steps=N_STEPS, batch_size=1)["seconds"] / 32
    t_occ = ex.occlusion_groups(data, sel_cost[:32], occ_groups)["seconds"] / 32
    print(f"Costo: IG {1000 * t_one:.1f} ms por transacción suelta, "
          f"{1000 * t_batch / N_COST:.1f} ms en lote de 64; oclusión {1000 * t_occ:.1f} ms")

    # --- 5. modelos interpretables de comparación
    baselines = {}
    if args.skip_baselines and (out / "resumen.json").exists():
        with open(out / "resumen.json", encoding="utf-8") as f:   # conserva lo ya calculado
            baselines = json.load(f).get("baselines_interpretables", {})
    if not args.skip_baselines:
        print("Entrenando modelos interpretables de comparación")
        baselines = interpretable_baselines(data, cfg, costs, score, test, th)
        for n, v in baselines.items():
            print(f"  {n:28s} {v['value_usd']:>10,.0f} USD ({100 * v['pct_of_oracle']:.1f} % del máximo)")

    # --- 6. guardar
    import csv
    with open(out / "casos_locales.csv", "w", newline="", encoding="utf-8") as f:
        cols = [c for c in local_table[0] if c != "senales"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(local_table)
    md = ("| Caso | Score | Decisión | Real | Monto | Señales principales (puntos de score) |\n"
          "|---|---|---|---|---|---|\n" + "\n".join(
            f"| {r['caso']} | {100 * r['score']:.1f} | {r['decision']} | {r['real']} | "
            f"${abs(r['monto_usd']):,.0f} | {r['top_senales']} |" for r in local_table) + "\n")
    (out / "casos_locales.md").write_text(md, encoding="utf-8")
    with open(out / "global_variables.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variable", "etiqueta"] + [f"{k}_pts" for k in imp])
        order = np.argsort(-imp["fraudes_detectados"])
        for i in order:
            w.writerow([names[i], label(names[i])] + [round(float(imp[k][i]), 4) for k in imp])
    summary = {
        "seed": args.seed, "modelo": MODEL_FILE, "captum": __import__("captum").__version__, "n_steps": N_STEPS, "n_steps_local": N_STEPS_LOCAL,
        "linea_base": "transacción neutra (media de entrenamiento en lo continuo, 0 en lo binario, "
                      "categoría más frecuente en lo categórico), repetida en los 16 pasos",
        "umbrales": th,
        "score_linea_base_pts": 100 * float(res_local["base_score"].mean()),
        "muestras": {"casos_locales": len(cases), **{k: int(len(v)) for k, v in sel.items()},
                     "validacion": int(len(sel_v)), "costo": N_COST},
        "completitud": {"error_max_local": float(np.abs(res_local["delta"]).max()),
                        "error_max_global": float(np.abs(res_f["delta"]).max()),
                        "error_corte_modelo": err},
        "validacion": {"spearman_ig_vs_oclusion": rho_group, "spearman_linea_base": rho_base},
        "costo_ms": {"ig_transaccion_suelta": 1000 * t_one, "ig_en_lote_64": 1000 * t_batch / N_COST,
                     "oclusion_transaccion_suelta": 1000 * t_occ},
        "top_global": [{"variable": n, "etiqueta": label(n), "fraudes_detectados_pts": float(a),
                        "falsos_positivos_pts": float(b), "falsos_negativos_pts": float(c)}
                       for n, a, b, c in top_global],
        "atribucion_por_paso": {
            "fraudes_detectados": [float(v) for v in np.abs(step_f).mean(axis=0) * 100],
            "legitimas_aprobadas": [float(v) for v in np.abs(step_l).mean(axis=0) * 100]},
        "casos": local_table,
        "baselines_interpretables": baselines,
    }
    with open(out / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"\nListo. Figuras y tablas en {out}")
