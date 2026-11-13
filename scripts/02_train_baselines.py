"""Paso 2: líneas base. Reglas fijas (proceso actual) y gradient boosting (alternativa clásica)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np

from fraudscan import baselines, runs
from fraudscan.config import base_parser, load_config
from fraudscan.data import Processed
from fraudscan.value import Costs

if __name__ == "__main__":
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.profile)
    data, costs = Processed(cfg["data"]["processed_dir"]), Costs.from_config(cfg)

    print("Reglas fijas (iteración 0)")
    for split in ("val", "test"):
        idx = data.split_idx(split)
        d = np.empty(len(idx), dtype="int8")
        for i in range(0, len(idx), 500_000):
            b = idx[i:i + 500_000]
            x_num, x_cat = data.tabular(b)
            d[i:i + len(b)] = baselines.rules_decision(x_num, x_cat, data.amt[b])
        runs.save_scores(cfg, "rules", split, d)
    runs.save_run(cfg, "rules", {"kind": "rules", "train_seconds": 0.0, "device": "cpu"})

    print("Gradient boosting (iteración 1)")
    model, info = baselines.fit_hgb(data, cfg, costs)
    for split in ("val", "test"):
        runs.save_scores(cfg, "hgb", split, baselines.predict_hgb(model, data, data.split_idx(split)))
    runs.save_run(cfg, "hgb", info)
    out = Path(cfg["data"]["artifacts_dir"]) / "hgb.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print("Listo.")
