"""Líneas base: reglas fijas (el proceso actual del emisor) y gradient boosting (alternativa clásica)."""
from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from . import schema as S
from .data import Processed
from .train import cost_weights
from .value import Costs


def rules_decision(x_num: np.ndarray, x_cat: np.ndarray, amt: np.ndarray) -> np.ndarray:
    """Reglas fijas típicas de un equipo de riesgo sin modelo. Representan "cómo se resuelve hoy".

    VERIFICAR si: compra en el extranjero de 50 USD o más; compra en línea de 200 USD o más que
    cuadruplica el gasto habitual de la tarjeta; 8 o más compras en 24 h; o compra en línea con
    datos de tarjeta incorrectos.  BLOQUEAR si: compra en el extranjero de 500 USD o más.
    """
    c = S.NUM_INDEX
    abs_amt = np.abs(amt)
    foreign = x_num[:, c["is_foreign"]] > 0.5
    online = x_cat[:, 0] == S.CHANNELS.index("Online Transaction")
    ratio = np.exp(x_num[:, c["log_amt_ratio"]])
    cnt_24h = np.expm1(x_num[:, c["log_cnt_24h"]])
    bad_data = (x_num[:, c["err_bad_cvv"]] + x_num[:, c["err_bad_card"]] + x_num[:, c["err_bad_exp"]]) > 0
    verify = (foreign & (abs_amt >= 50)) | (online & (ratio >= 4) & (abs_amt >= 200)) \
        | (cnt_24h >= 8 - 1e-6) | (online & bad_data)
    block = foreign & (abs_amt >= 500)
    d = np.full(len(amt), S.APPROVE, dtype="int8")
    d[verify] = S.VERIFY
    d[block] = S.BLOCK
    return d


def _matrix(data: Processed, idx: np.ndarray, cat_cols: list[int] | None = None) -> np.ndarray:
    x_num, x_cat = data.tabular(idx)
    if cat_cols is not None:
        x_cat = x_cat[:, cat_cols]
    return np.hstack([x_num, x_cat.astype("float32")])


def fit_hgb(data: Processed, cfg: dict, costs: Costs, log=print):
    """Gradient boosting con las mismas variables, la misma muestra y los mismos pesos que el modelo A."""
    tc = cfg["train"]
    idx = data.train_sample(tc["neg_per_pos"], tc["seed"])
    use = [data.meta["cat_features"].index(name) for name in cfg["features"]["cat_use"]]
    X, y = _matrix(data, idx, use), data.y[idx]
    w = cost_weights(y, data.amt[idx], costs, tuple(tc["weight_clip"]))
    n_num = len(data.meta["num_features"])
    cat_cols = [n_num + j for j, i in enumerate(use) if data.meta["cat_sizes"][i] <= 255]
    model = HistGradientBoostingClassifier(
        max_iter=tc["hgb_max_iter"], learning_rate=0.1, max_leaf_nodes=63, l2_regularization=1.0,
        categorical_features=cat_cols, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=tc["seed"])
    t0 = time.time()
    model.fit(X, y, sample_weight=w)
    info = {"kind": "hgb", "weighted_loss": True, "train_rows": int(len(idx)),
            "train_seconds": time.time() - t0, "device": "cpu", "trees": int(model.n_iter_), "cat_cols": use}
    model.fraudscan_cat_cols = use
    log(f"  gradient boosting: {model.n_iter_} árboles en {info['train_seconds']:.0f}s")
    return model, info


def predict_hgb(model, data: Processed, idx: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    out = np.empty(len(idx), dtype="float32")
    cols = getattr(model, "fraudscan_cat_cols", None)
    for i in range(0, len(idx), chunk):
        b = idx[i:i + chunk]
        out[i:i + len(b)] = model.predict_proba(_matrix(data, b, cols))[:, 1]
    return out
