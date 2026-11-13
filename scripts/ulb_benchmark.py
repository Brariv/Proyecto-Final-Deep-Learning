"""Verificación secundaria con datos REALES (dataset ULB, tarjetas europeas, septiembre de 2013).

El dataset principal (IBM) es sintético. Aquí se repite la comparación gradient boosting vs. MLP
con el mismo modelo de valor sobre transacciones reales. ULB no trae identificador de tarjeta
(sus variables V1..V28 son componentes PCA anónimos), así que no se puede usar el modelo B.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from fraudscan.config import base_parser, load_config
from fraudscan.evaluate import write_json
from fraudscan.models import pick_device
from fraudscan.train import cost_weights
from fraudscan.value import Costs, best_thresholds, bootstrap_ci, decide, summarize, value_per_tx


def train_mlp(Xtr, ytr, wtr, Xva, yva, amt_va, costs, device, epochs=40, seed=42):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 64), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    X, y, w = (torch.tensor(a, dtype=torch.float32, device=device) for a in (Xtr, ytr, wtr))
    Xv = torch.tensor(Xva, dtype=torch.float32, device=device)
    best, best_state = -np.inf, None
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(X), device=device).split(2048):
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(model(X[b]).squeeze(-1), y[b],
                                                                         reduction="none") * w[b]).sum() / w[b].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            s = torch.sigmoid(model(Xv).squeeze(-1)).cpu().numpy()
        v = best_thresholds(s, yva, amt_va, costs)["value"]
        if v > best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval()


if __name__ == "__main__":
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.profile)
    costs, device = Costs.from_config(cfg), pick_device()
    df = pd.read_csv(cfg["data"]["ulb_csv"]).sort_values("Time", kind="stable").reset_index(drop=True)
    n = len(df)
    i_tr, i_va = int(0.6 * n), int(0.8 * n)          # división temporal 60 / 20 / 20
    hour = (df["Time"].to_numpy() % 86400) / 3600
    X = np.column_stack([df[[f"V{i}" for i in range(1, 29)]].to_numpy(), np.log1p(df["Amount"].to_numpy()),
                         np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]).astype("float32")
    y, amt = df["Class"].to_numpy().astype("uint8"), df["Amount"].to_numpy().astype("float32")
    mean, std = X[:i_tr].mean(0), X[:i_tr].std(0) + 1e-6
    X = (X - mean) / std
    sl = {"train": slice(0, i_tr), "val": slice(i_tr, i_va), "test": slice(i_va, n)}
    print(f"ULB: {n:,} transacciones, {int(y.sum())} fraudes "
          f"(train {int(y[sl['train']].sum())}, val {int(y[sl['val']].sum())}, test {int(y[sl['test']].sum())})")
    w_cost = cost_weights(y[sl["train"]], amt[sl["train"]], costs, tuple(cfg["train"]["weight_clip"]))
    # Sin submuestreo: se compensa el desbalance subiendo el peso de la clase positiva.
    w_cost = w_cost * np.where(y[sl["train"]] == 1, 50.0, 1.0).astype("float32")

    scores, seconds = {}, {}
    t0 = time.time()
    hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True, random_state=42)
    hgb.fit(X[sl["train"]], y[sl["train"]], sample_weight=w_cost)
    seconds["Gradient boosting"] = time.time() - t0
    scores["Gradient boosting"] = {k: hgb.predict_proba(X[s])[:, 1] for k, s in sl.items() if k != "train"}
    t0 = time.time()
    mlp = train_mlp(X[sl["train"]], y[sl["train"]].astype("float32"), w_cost, X[sl["val"]], y[sl["val"]],
                    amt[sl["val"]], costs, device)
    seconds["MLP (pérdida por costo)"] = time.time() - t0
    with torch.no_grad():
        scores["MLP (pérdida por costo)"] = {
            k: torch.sigmoid(mlp(torch.tensor(X[s], device=device)).squeeze(-1)).cpu().numpy()
            for k, s in sl.items() if k != "train"}

    out = []
    for name, s in scores.items():
        th = best_thresholds(s["val"], y[sl["val"]], amt[sl["val"]], costs)
        d = decide(s["test"], th["t_verify"], th["t_block"])
        r = summarize(d, y[sl["test"]], amt[sl["test"]], costs)
        ci = bootstrap_ci(value_per_tx(d, y[sl["test"]], amt[sl["test"]], costs), np.arange(len(d)), 500)
        r.update(name=name, ap_test=float(average_precision_score(y[sl["test"]], s["test"])),
                 ci_low=ci["low"], ci_high=ci["high"], train_seconds=seconds[name])
        out.append(r)
        print(f"  {name}: AP {r['ap_test']:.3f} | recall {r['recall']:.3f} | precisión {r['precision']:.3f} | "
              f"valor {r['value_usd']:,.0f} USD (IC 95 % {ci['low']:,.0f} a {ci['high']:,.0f}; "
              f"{100 * r['value_usd'] / r['oracle_usd']:.1f} % del máximo) | {seconds[name]:.0f}s")
    write_json(Path(cfg["data"]["reports_dir"]) / "ulb_benchmark.json", out)
