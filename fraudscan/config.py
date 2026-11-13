"""Carga de configuración y rutas del proyecto."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None, profile: str = "quick") -> dict:
    """Lee el YAML y fusiona el perfil elegido (quick | full) dentro de cfg["train"]."""
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if profile not in cfg["profiles"]:
        raise ValueError(f"Perfil desconocido: {profile}. Opciones: {list(cfg['profiles'])}")
    cfg["train"].update(cfg["profiles"][profile])
    cfg["profile"] = profile
    for key in ("ibm_dir", "ulb_csv", "processed_dir", "artifacts_dir", "reports_dir"):
        p = Path(cfg["data"][key])
        cfg["data"][key] = p if p.is_absolute() else ROOT / p
    return cfg


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="Ruta al YAML de configuración")
    p.add_argument("--profile", default="quick", choices=["quick", "full"],
                   help="quick: CPU en minutos; full: pensado para GPU")
    return p
