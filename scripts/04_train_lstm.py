"""Paso 4: modelo B (LSTM sobre las últimas 16 transacciones de la tarjeta, pérdida ponderada por costo)."""
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
    parser.add_argument("--in-memory", action="store_true", help="Cargar X_num completo en RAM (~3 GB)")
    args = parser.parse_args()
    cfg = load_config(args.config, args.profile)
    data = Processed(cfg["data"]["processed_dir"], in_memory=args.in_memory)
    costs, device = Costs.from_config(cfg), pick_device(args.device)

    print("Modelo B - LSTM (lstm_cost)")
    model, info = train.fit("lstm", data, cfg, costs, device, weighted=True)
    for split in ("val", "test"):
        print(f"  puntuando {split} completo")
        runs.save_scores(cfg, "lstm_cost", split,
                         train.predict(model, data, data.split_idx(split), device, progress=True))
    runs.save_run(cfg, "lstm_cost", info)
    save_model(model, Path(cfg["data"]["artifacts_dir"]) / "lstm_cost.pt", extra={"name": "lstm_cost"})
    print("Listo.")
