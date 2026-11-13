"""Paso 3: modelo A (MLP). Tres iteraciones, cada una cambia una sola cosa:

  mlp_loc    A0: pérdida estándar y embedding de la ubicación (país o estado) del comercio
  mlp_plain  A1: igual, pero sin la identidad de la ubicación (solo señales genéricas)
  mlp_cost   A : igual que A1, con pérdida ponderada por el costo del error
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fraudscan import runs, train
from fraudscan.config import base_parser, load_config
from fraudscan.data import Processed
from fraudscan.models import pick_device, save_model
from fraudscan.value import Costs

if __name__ == "__main__":
    parser = base_parser(__doc__)
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (por defecto se detecta)")
    args = parser.parse_args()
    cfg = load_config(args.config, args.profile)
    data, costs, device = Processed(cfg["data"]["processed_dir"]), Costs.from_config(cfg), pick_device(args.device)

    all_cats = data.meta["cat_features"]
    for name, weighted, cat_use in (("mlp_loc", False, all_cats), ("mlp_plain", False, None), ("mlp_cost", True, None)):
        print(f"Modelo A ({name}): pérdida {'ponderada por costo' if weighted else 'estándar'}, "
              f"categóricas {cat_use or cfg['features']['cat_use']}")
        model, info = train.fit("mlp", data, cfg, costs, device, weighted=weighted, cat_use=cat_use)
        for split in ("val", "test"):
            runs.save_scores(cfg, name, split, train.predict(model, data, data.split_idx(split), device))
        runs.save_run(cfg, name, info)
        save_model(model, Path(cfg["data"]["artifacts_dir"]) / f"{name}.pt", extra={"name": name})
    print("Listo.")
