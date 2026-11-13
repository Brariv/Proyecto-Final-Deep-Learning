"""Corre todo el pipeline en orden. Ejemplos:

    python scripts/run_all.py                    # perfil rápido (CPU, minutos)
    python scripts/run_all.py --profile full     # perfil completo (pensado para GPU)
    python scripts/run_all.py --skip-prepare     # si data/processed ya existe
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["01_prepare_data.py", "02_train_baselines.py", "03_train_mlp.py", "04_train_lstm.py",
         "05_evaluate.py", "06_export_demo.py"]

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", default="quick", choices=["quick", "full"])
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--with-ulb", action="store_true", help="Corre también la verificación con el dataset ULB")
    args = p.parse_args()
    steps = STEPS[1:] if args.skip_prepare else STEPS
    if args.with_ulb:
        steps = steps + ["ulb_benchmark.py"]
    for step in steps:
        print(f"\n=== {step} ===", flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, str(HERE / step), "--profile", args.profile], check=True)
        print(f"=== {step} terminó en {time.time() - t0:.0f}s ===", flush=True)
