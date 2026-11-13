"""Guardado y lectura de los resultados de cada iteración (scores y metadatos)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def scores_path(cfg: dict, name: str, split: str) -> Path:
    return Path(cfg["data"]["reports_dir"]) / "scores" / f"{name}_{split}.npy"


def save_scores(cfg: dict, name: str, split: str, values: np.ndarray) -> None:
    p = scores_path(cfg, name, split)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, values)


def load_scores(cfg: dict, name: str, split: str) -> np.ndarray:
    return np.load(scores_path(cfg, name, split))


def save_run(cfg: dict, name: str, info: dict) -> None:
    p = Path(cfg["data"]["reports_dir"]) / "runs" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    info = dict(info, name=name, profile=cfg["profile"])
    with open(p, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)


def load_run(cfg: dict, name: str) -> dict | None:
    p = Path(cfg["data"]["reports_dir"]) / "runs" / f"{name}.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)
