"""Preprocesamiento por lotes: del CSV de IBM a arreglos listos para entrenar.

Todas las variables de historial usan SOLO información anterior a la transacción (sin fuga de
información). La misma lógica está implementada transacción por transacción en `online.py`, y
`tests/test_online_parity.py` verifica que ambas den el mismo resultado.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema as S

TX_FILE = "credit_card_transactions-ibm_v2.csv"
CARDS_FILE = "sd254_cards.csv"
USERS_FILE = "sd254_users.csv"


def _money(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.replace("$", "", regex=False).astype("float64").to_numpy()


def load_cards(ibm_dir: Path) -> pd.DataFrame:
    """Tabla de tarjetas con las columnas que usa el modelo, indexada por (User, Card)."""
    c = pd.read_csv(ibm_dir / CARDS_FILE)
    u = pd.read_csv(ibm_dir / USERS_FILE)
    out = pd.DataFrame({
        "user": c["User"].astype("int32"),
        "card": c["CARD INDEX"].astype("int32"),
        "brand": c["Card Brand"].astype(str),
        "card_type": c["Card Type"].astype(str),
        "credit_limit": _money(c["Credit Limit"]),
        "has_chip": (c["Has Chip"].astype(str).str.upper() == "YES").astype("int8"),
    })
    # El id de usuario es el número de fila de sd254_users.csv.
    out["home_state"] = u["State"].astype(str).to_numpy()[out["user"].to_numpy()]
    out["home_city"] = u["City"].astype(str).to_numpy()[out["user"].to_numpy()]
    return out


class _Vocab:
    """Diccionario texto -> código entero que crece conforme se leen los bloques."""

    def __init__(self):
        self.names: list[str] = []
        self.index: dict[str, int] = {}

    def encode(self, series: pd.Series) -> np.ndarray:
        codes, uniques = pd.factorize(series, sort=False)
        table = np.empty(len(uniques), dtype="int32")
        for i, name in enumerate(uniques):
            name = str(name)
            if name not in self.index:
                self.index[name] = len(self.names)
                self.names.append(name)
            table[i] = self.index[name]
        return table[codes]


def read_transactions(ibm_dir: Path, chunksize: int = 2_000_000, nrows: int | None = None) -> dict:
    """Lee el CSV por bloques y devuelve columnas compactas de numpy (cabe en ~4 GB de RAM).

    `loc` y `err` se devuelven como códigos enteros; sus textos están en `loc_names` y `err_names`.
    """
    cols = {k: [] for k in ("user", "card", "ts", "year", "amt", "chan", "merchant", "loc", "mcc", "err", "y")}
    reader = pd.read_csv(ibm_dir / TX_FILE, chunksize=chunksize, nrows=nrows,
                         dtype={"Time": str, "Amount": str, "Use Chip": str, "Merchant State": str,
                                "Merchant City": str, "Errors?": str, "Is Fraud?": str})
    chan_map = {name: i for i, name in enumerate(S.CHANNELS)}
    loc_vocab, err_vocab = _Vocab(), _Vocab()
    for chunk in reader:
        date = pd.to_datetime(dict(year=chunk["Year"], month=chunk["Month"], day=chunk["Day"]))
        days = date.to_numpy().astype("datetime64[D]").astype("int64")
        hhmm = chunk["Time"].str.split(":", expand=True).astype("int64").to_numpy()
        cols["ts"].append(days * 1440 + hhmm[:, 0] * 60 + hhmm[:, 1])
        cols["user"].append(chunk["User"].to_numpy("int32"))
        cols["card"].append(chunk["Card"].to_numpy("int32"))
        cols["year"].append(chunk["Year"].to_numpy("int16"))
        cols["amt"].append(_money(chunk["Amount"]).astype("float32"))
        cols["chan"].append(chunk["Use Chip"].map(chan_map).to_numpy("int16"))
        cols["merchant"].append(chunk["Merchant Name"].to_numpy("int64"))
        online = chunk["Merchant City"] == "ONLINE"
        loc = chunk["Merchant State"].where(~online, S.ONLINE_LOC).fillna(S.UNKNOWN_LOC)
        cols["loc"].append(loc_vocab.encode(loc))
        cols["mcc"].append(chunk["MCC"].to_numpy("int32"))
        cols["err"].append(err_vocab.encode(chunk["Errors?"].fillna("")))
        cols["y"].append((chunk["Is Fraud?"] == "Yes").to_numpy("uint8"))
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["loc_names"], out["err_names"] = loc_vocab.names, err_vocab.names
    return out


def build(cfg: dict, nrows: int | None = None, verbose: bool = True) -> dict:
    """Construye y guarda los arreglos procesados. Devuelve el diccionario `meta`."""
    t0 = time.time()
    ibm_dir, out_dir = Path(cfg["data"]["ibm_dir"]), Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log = (lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)) if verbose else (lambda *a: None)

    cards = load_cards(ibm_dir)
    tx = read_transactions(ibm_dir, nrows=nrows)
    loc_names, err_names = tx.pop("loc_names"), tx.pop("err_names")
    n = len(tx["y"])
    log(f"{n:,} transacciones leídas, {int(tx['y'].sum()):,} fraudes")

    # --- Orden: por tarjeta y luego por tiempo (estable: respeta el orden del archivo en empates)
    card_key = tx["user"].astype("int64") * 100 + tx["card"]
    order = np.lexsort((tx["ts"], card_key))          # lexsort es estable
    np.save(out_dir / "order.npy", order.astype("int32"))   # fila original del CSV
    for k in list(tx):
        tx[k] = tx[k][order]
    uniq_keys, card_idx = np.unique(card_key[order], return_inverse=True)
    del card_key, order
    card_idx = card_idx.astype("int32")
    pos = np.arange(n, dtype="int64")
    is_start = np.r_[True, card_idx[1:] != card_idx[:-1]]
    start = np.maximum.accumulate(np.where(is_start, pos, 0))
    del is_start
    prior_n = pos - start                             # transacciones previas de la tarjeta
    ts = tx["ts"]
    log("ordenado por tarjeta y tiempo")

    # --- Datos estáticos de la tarjeta
    cards = cards.set_index(cards["user"].astype("int64") * 100 + cards["card"]).reindex(uniq_keys)
    if cards["credit_limit"].isna().any():
        raise ValueError("Hay tarjetas en las transacciones que no existen en sd254_cards.csv")

    # --- Vocabularios (solo con el periodo de entrenamiento, para no mirar el futuro)
    train_mask = tx["year"] <= cfg["split"]["train_end_year"]
    loc_counts = np.bincount(tx["loc"][train_mask], minlength=len(loc_names))
    keep = [name for name, c in zip(loc_names, loc_counts)
            if (c >= cfg["features"]["min_loc_count"] or name == S.ONLINE_LOC) and name != S.UNKNOWN_LOC]
    loc_vocab = ["<otra>"] + sorted(keep)
    loc_map = {name: i for i, name in enumerate(loc_vocab)}
    loc_idx = np.array([loc_map.get(name, S.LOC_OTHER) for name in loc_names], dtype="int16")[tx["loc"]]
    mcc_vocab = [0] + sorted(np.unique(tx["mcc"][train_mask]).tolist())
    mcc_lookup = np.zeros(int(tx["mcc"].max()) + 1, dtype="int16")
    mcc_lookup[mcc_vocab[1:]] = np.arange(1, len(mcc_vocab))
    mcc_idx = mcc_lookup[tx["mcc"]]

    # --- Variables numéricas (se escriben columna por columna en un memmap para cuidar la RAM)
    X = np.lib.format.open_memmap(out_dir / "X_num.npy", mode="w+", dtype="float32", shape=(n, S.N_NUM))
    col = S.NUM_INDEX
    amt = tx["amt"]
    abs_amt = np.abs(amt).astype("float64")

    X[:, col["log_amt"]] = np.sign(amt) * np.log1p(abs_amt)
    X[:, col["is_refund"]] = amt < 0
    hour = ((ts % 1440) / 60.0).astype("float32")
    X[:, col["hour_sin"]] = np.sin(2 * np.pi * hour / 24)
    X[:, col["hour_cos"]] = np.cos(2 * np.pi * hour / 24)
    dow = (((ts // 1440) + 3) % 7).astype("float32")  # 1970-01-01 fue jueves; lunes = 0
    X[:, col["dow_sin"]] = np.sin(2 * np.pi * dow / 7)
    X[:, col["dow_cos"]] = np.cos(2 * np.pi * dow / 7)
    del hour, dow
    for label, feat in S.ERROR_FLAGS.items():
        has = np.array([label in name.split(",") for name in err_names])
        X[:, col[feat]] = has[tx["err"]]

    # Los estados de EE. UU. son códigos de 2 letras; los países vienen con nombre completo.
    is_online = np.array([name == S.ONLINE_LOC for name in loc_names])[tx["loc"]]
    X[:, col["is_foreign"]] = np.array([S.is_foreign_loc(name) for name in loc_names])[tx["loc"]]
    name_to_code = {name: i for i, name in enumerate(loc_names)}
    home_code = np.array([name_to_code.get(h, -1) for h in cards["home_state"]])
    X[:, col["away_from_home"]] = (~is_online) & (tx["loc"] != home_code[card_idx])
    foreign_by_vocab = np.array([S.is_foreign_loc(name) for name in loc_vocab])
    log("variables de la transacción listas")

    # --- Variables de historial (solo miran hacia atrás)
    dt = np.where(prior_n >= 1, ts - np.roll(ts, 1), S.DT_CAP_MIN)
    X[:, col["log_dt_prev"]] = np.log1p(np.clip(dt, 0, S.DT_CAP_MIN))
    dt = np.where(prior_n >= 5, ts - np.roll(ts, 5), S.DT_CAP_MIN)
    X[:, col["log_dt_5"]] = np.log1p(np.clip(dt, 0, S.DT_CAP_MIN))
    del dt

    cs_before = np.cumsum(abs_amt)                    # suma de |monto| de todas las filas anteriores
    cs_before -= abs_amt
    prior_mean = cs_before - cs_before[start]
    np.divide(prior_mean, prior_n, out=prior_mean, where=prior_n > 0)
    prior_mean[prior_n == 0] = abs_amt[prior_n == 0]
    X[:, col["log_amt_ratio"]] = np.log((abs_amt + 1.0) / (prior_mean + 1.0))
    del prior_mean
    X[:, col["log_prior_count"]] = np.log1p(prior_n)

    # "Nuevo" = primera vez que la tarjeta ve ese comercio / esa categoría.
    for feat, values in (("merchant_new", tx["merchant"]), ("mcc_new", tx["mcc"])):
        o = np.lexsort((values, card_idx))            # estable: conserva el orden temporal en empates
        v, c = values[o], card_idx[o]
        first = np.r_[True, (v[1:] != v[:-1]) | (c[1:] != c[:-1])]
        flag = np.zeros(n, dtype="float32")
        flag[o[first]] = 1.0
        X[:, col[feat]] = flag
        del o, v, c, first, flag
    X[:, col["state_changed"]] = (prior_n >= 1) & (loc_idx != np.roll(loc_idx, 1))

    # Compras previas de la tarjeta en esta misma ubicación (estado, país o en línea).
    o = np.lexsort((tx["loc"], card_idx))
    v, c = tx["loc"][o], card_idx[o]
    first = np.r_[True, (v[1:] != v[:-1]) | (c[1:] != c[:-1])]
    rank = np.arange(n, dtype="int64")
    seen = np.empty(n, dtype="int64")
    seen[o] = rank - np.maximum.accumulate(np.where(first, rank, 0))
    del o, v, c, first, rank
    X[:, col["loc_new"]] = seen == 0
    share = seen.astype("float64")
    np.divide(share, prior_n, out=share, where=prior_n > 0)
    X[:, col["loc_share_prior"]] = share
    del seen, share

    cs_on = np.cumsum(is_online, dtype="float64")
    cs_on -= is_online
    share = cs_on - cs_on[start]
    np.divide(share, prior_n, out=share, where=prior_n > 0)
    X[:, col["online_share_prior"]] = share
    del cs_on, share, is_online

    # Ventana de 24 h: primera fila de la tarjeta con ts >= ts_i - 1440, mirando a lo sumo 64 atrás.
    key = card_idx.astype("int64") * (1 << 40) + ts   # creciente porque está ordenado por tarjeta y tiempo
    lo = np.searchsorted(key, key - S.WINDOW_MIN, side="left")
    del key
    lo = np.maximum(np.maximum(lo, start), pos - S.WINDOW_MAX_TX)
    X[:, col["log_cnt_24h"]] = np.log1p(np.minimum(pos - lo, S.CNT_CLIP))
    X[:, col["log_sum_24h"]] = np.log1p(np.maximum(cs_before - cs_before[lo], 0.0))
    del lo, cs_before

    limit = np.maximum(cards["credit_limit"].to_numpy(), 1.0)
    X[:, col["log_credit_limit"]] = np.log1p(limit)[card_idx]
    X[:, col["amt_over_limit"]] = np.minimum(abs_amt / limit[card_idx], S.AMT_OVER_LIMIT_CLIP)
    X[:, col["card_credit"]] = (cards["card_type"].to_numpy() == "Credit")[card_idx]
    X[:, col["card_prepaid"]] = (cards["card_type"].to_numpy() == "Debit (Prepaid)")[card_idx]
    X[:, col["card_has_chip"]] = cards["has_chip"].to_numpy()[card_idx]
    X[:, col["no_history"]] = prior_n == 0
    del abs_amt, prior_n, pos
    log("variables de historial listas")

    # --- Estandarización (media y desviación del periodo de entrenamiento; binarias sin tocar)
    mean, std = np.zeros(S.N_NUM), np.ones(S.N_NUM)
    sample = np.flatnonzero(train_mask)
    sample = np.sort(np.random.default_rng(0).choice(sample, size=min(len(sample), 2_000_000), replace=False))
    Xs = np.asarray(X[sample])
    for name, j in col.items():
        if name not in S.BINARY_FEATURES:
            mean[j], std[j] = float(Xs[:, j].mean()), float(max(Xs[:, j].std(), 1e-6))
    X.flush()
    del X, Xs, sample

    # --- Guardar
    np.save(out_dir / "X_cat.npy", np.stack([tx["chan"], mcc_idx, loc_idx], axis=1).astype("int16"))
    np.save(out_dir / "y.npy", tx["y"])
    np.save(out_dir / "amt.npy", tx["amt"])
    np.save(out_dir / "ts.npy", ts)
    np.save(out_dir / "year.npy", tx["year"])
    np.save(out_dir / "card_idx.npy", card_idx)
    np.save(out_dir / "user.npy", tx["user"].astype("int16"))
    np.save(out_dir / "start.npy", start.astype("int32"))
    np.save(out_dir / "merchant.npy", tx["merchant"])
    np.save(out_dir / "mcc_raw.npy", tx["mcc"].astype("int16"))
    np.save(out_dir / "loc_raw.npy", tx["loc"].astype("int16"))
    np.save(out_dir / "err_raw.npy", tx["err"].astype("int16"))

    sp = cfg["split"]
    in_val, in_test = np.isin(tx["year"], sp["val_years"]), np.isin(tx["year"], sp["test_years"])
    meta = {
        "n": int(n), "n_fraud": int(tx["y"].sum()),
        "num_features": S.NUM_FEATURES, "cat_features": S.CAT_FEATURES,
        "cat_sizes": [len(S.CHANNELS), len(mcc_vocab), len(loc_vocab)],
        "loc_vocab": loc_vocab, "mcc_vocab": [int(m) for m in mcc_vocab],
        "loc_is_foreign": foreign_by_vocab.astype(int).tolist(),
        "loc_names_raw": loc_names, "err_names_raw": err_names,
        "scaler_mean": mean.tolist(), "scaler_std": std.tolist(),
        "card_keys": [int(k) for k in uniq_keys],
        "split": sp, "seq_len": cfg["features"]["seq_len"],
        "counts": {
            "train": int(train_mask.sum()), "train_fraud": int(tx["y"][train_mask].sum()),
            "val": int(in_val.sum()), "val_fraud": int(tx["y"][in_val].sum()),
            "test": int(in_test.sum()), "test_fraud": int(tx["y"][in_test].sum()),
        },
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    log(f"guardado en {out_dir}: {meta['counts']}")
    return meta
