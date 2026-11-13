"""Paso 7: genera estados de cuenta de ejemplo en PDF a partir del periodo de prueba.

Cada ejemplo es un mes real de una tarjeta que el modelo nunca vio al entrenar. Escribe:

- app/static/samples/<nombre>.pdf   los estados de cuenta, para subirlos en el demo
- artifacts/statements.json         el historial de cada tarjeta al inicio del mes y la etiqueta real
                                    de cada transacción (para comparar después de analizar)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fraudscan import statement
from fraudscan.config import ROOT, base_parser, load_config
from fraudscan.data import Processed
from fraudscan.replay import RawStore

# (archivo, título, descripción, índice de tarjeta, mes). Elegidos del periodo de prueba (2018-2019).
SAMPLES = [
    ("estado_sin_fraude", "Mes normal", "Compras habituales, sin fraude.", 6099, "2019-09"),
    ("estado_fraude_extranjero", "Fraude en el extranjero", "Ráfaga de compras presenciales en otro país.", 5776, "2018-08"),
    ("estado_fraude_en_linea", "Fraude en línea", "Cargos en línea que no son del titular.", 2263, "2018-11"),
    ("estado_viajero", "Titular de viaje", "Compras legítimas en el extranjero; no hay fraude.", 5091, "2018-06"),
]

if __name__ == "__main__":
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.profile)
    data = Processed(cfg["data"]["processed_dir"])
    store = RawStore(data, cfg["data"]["ibm_dir"])
    out_dir = ROOT / "app" / "static" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    month_of = (np.datetime64("1970-01-01") + (data.ts // 1440).astype("timedelta64[D]")).astype("datetime64[M]")

    index = {}
    for slug, title, description, card, month in SAMPLES:
        rows = store.card_rows(card)
        rows = rows[month_of[rows] == np.datetime64(month)]
        if len(rows) == 0:
            raise SystemExit(f"La tarjeta {card} no tiene transacciones en {month}")
        if not np.isin(data.year[rows], cfg["split"]["test_years"]).all():
            raise SystemExit(f"{month} no está en el periodo de prueba")
        state = store.replay(card, until_row=int(rows[0]))
        txs = [store.transaction(int(r)) for r in rows]
        info = state.info
        sid = f"EC-{info['user']:04d}-{info['card']}-{month.replace('-', '')}"
        header = {
            "statement_id": sid, "card_id": f"{info['user']:04d}-{info['card']}", "brand": info["brand"],
            "card_type": state.card_type, "credit_limit": state.credit_limit, "has_chip": state.has_chip,
            "home_state": state.home_state, "home_city": info["home_city"],
            "period_start": statement.ts_to_date_time(txs[0].ts)[0],
            "period_end": statement.ts_to_date_time(txs[-1].ts)[0],
        }
        statement.build_pdf(out_dir / f"{slug}.pdf", header, txs)
        truth = [int(v) for v in data.y[rows]]
        index[sid] = {"file": f"{slug}.pdf", "title": title, "description": description,
                      "state": state.to_dict(), "truth": truth}
        print(f"{slug}.pdf: {len(txs)} transacciones, {sum(truth)} fraudes reales ({sid})")

    with open(Path(cfg["data"]["artifacts_dir"]) / "statements.json", "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"Listo: {out_dir} y artifacts/statements.json")
