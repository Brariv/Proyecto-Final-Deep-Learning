"""Paso 6: exporta lo que necesita el demo para correr SIN el dataset completo.

- artifacts/vocab.json: vocabularios y valores de referencia para las explicaciones.
- artifacts/demo.json: estado de algunas tarjetas y un flujo de transacciones de 2019 (periodo de
  prueba: el modelo nunca las vio al entrenar ni al elegir umbrales), con y sin fraude.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fraudscan import schema as S
from fraudscan.config import base_parser, load_config
from fraudscan.data import Processed
from fraudscan.replay import RawStore

DAY = 1440

if __name__ == "__main__":
    parser = base_parser(__doc__)
    parser.add_argument("--fraud-cards", type=int, default=8)
    parser.add_argument("--clean-cards", type=int, default=8)
    parser.add_argument("--max-per-card", type=int, default=45)
    args = parser.parse_args()
    cfg = load_config(args.config, args.profile)
    data = Processed(cfg["data"]["processed_dir"])
    store = RawStore(data, cfg["data"]["ibm_dir"])
    art = Path(cfg["data"]["artifacts_dir"])
    art.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    meta = data.meta

    # --- vocab.json
    train_idx = data.split_idx("train")
    sample = np.sort(rng.choice(train_idx, 500_000, replace=False))
    cat_baseline = [int(np.bincount(data.X_cat[sample, j]).argmax()) for j in range(data.X_cat.shape[1])]
    num_baseline = [0.0 if name in S.BINARY_FEATURES else float(meta["scaler_mean"][j])
                    for j, name in enumerate(meta["num_features"])]
    with open(art / "vocab.json", "w", encoding="utf-8") as f:
        json.dump({"loc_vocab": meta["loc_vocab"], "mcc_vocab": meta["mcc_vocab"],
                   "num_features": meta["num_features"], "seq_len": meta["seq_len"],
                   "num_baseline": num_baseline, "cat_baseline": cat_baseline}, f, ensure_ascii=False)

    # --- tarjetas del demo (solo del periodo de prueba)
    test_years = cfg["split"]["test_years"]
    rows_test = data.split_idx("test")
    fraud_rows = rows_test[data.y[rows_test] == 1]
    all_cards, counts = np.unique(data.card_idx[rows_test], return_counts=True)
    clean_pool = np.setdiff1d(all_cards[counts >= 150], np.unique(data.card_idx[fraud_rows]))

    # Variedad: se alternan tarjetas con fraude en línea, en el extranjero y presencial local.
    chan = data.X_cat[fraud_rows, 0]
    foreign = np.asarray(data.X_num[np.sort(fraud_rows), S.NUM_INDEX["is_foreign"]]) > 0.5
    kinds = np.where(foreign, 2, np.where(chan == S.CHANNELS.index("Online Transaction"), 1, 0))
    pools = {k: list(rng.permutation(np.unique(data.card_idx[fraud_rows[kinds == k]]))) for k in (2, 1, 0)}
    picked: list[int] = []
    while len(picked) < args.fraud_cards and any(pools.values()):
        for k in (2, 1, 2, 0):
            while pools[k] and int(pools[k][-1]) in picked:
                pools[k].pop()
            if pools[k] and len(picked) < args.fraud_cards:
                picked.append(int(pools[k].pop()))
    clean = [int(c) for c in rng.choice(clean_pool, args.clean_cards, replace=False)]

    cards_out, stream = {}, []
    for card in picked + clean:
        rows = store.card_rows(card)
        rows = rows[np.isin(data.year[rows], test_years)]
        ts, y = data.ts[rows], data.y[rows]
        if y.any():                                   # ventana alrededor de la primera ráfaga de fraude
            first = ts[y == 1][0]
            burst_end = ts[(y == 1) & (ts <= first + 10 * DAY)][-1]
            win = rows[(ts >= first - 4 * DAY) & (ts <= burst_end + 2 * DAY)]
        else:
            t0 = ts[0] + int(rng.integers(0, max(1, (ts[-1] - ts[0]) - 5 * DAY)))
            win = rows[(ts >= t0) & (ts <= t0 + 5 * DAY)]
        win = win[:args.max_per_card]
        if len(win) == 0:
            continue
        state = store.replay(card, until_row=int(win[0]))
        c_id = f"{state.info['user']:04d}-{state.info['card']}"
        cards_out[c_id] = state.to_dict()
        for row in win:
            tx = store.transaction(int(row))
            stream.append({"card_id": c_id, "ts": tx.ts, "amount": round(tx.amount, 2), "channel": tx.channel,
                           "merchant": tx.merchant, "mcc": tx.mcc, "loc": tx.loc, "errors": tx.errors,
                           "is_fraud": int(data.y[row])})

    # El flujo se reproduce en el orden real de cada tarjeta, intercalando tarjetas por tiempo relativo.
    first_ts = {}
    for s in stream:
        first_ts.setdefault(s["card_id"], s["ts"])
    stream.sort(key=lambda s: (s["ts"] - first_ts[s["card_id"]], s["card_id"]))
    with open(art / "demo.json", "w", encoding="utf-8") as f:
        json.dump({"cards": cards_out, "stream": stream,
                   "year": " y ".join(str(y) for y in test_years)}, f, ensure_ascii=False)
    n_fraud = sum(s["is_fraud"] for s in stream)
    print(f"{len(cards_out)} tarjetas, {len(stream)} transacciones en el flujo ({n_fraud} fraudes) -> {art / 'demo.json'}")
