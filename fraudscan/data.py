"""Acceso a los arreglos procesados: división temporal, muestreo y armado de lotes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Processed:
    """Arreglos generados por `prepare.build`, ordenados por tarjeta y tiempo."""

    def __init__(self, processed_dir: str | Path, in_memory: bool = False):
        d = Path(processed_dir)
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"No hay datos procesados en {d}. Corre primero scripts/01_prepare_data.py")
        with open(d / "meta.json", encoding="utf-8") as f:
            self.meta = json.load(f)
        # X_num pesa ~3 GB: por defecto se lee como memmap (el sistema operativo lo va cargando).
        self.X_num = np.load(d / "X_num.npy", mmap_mode=None if in_memory else "r")
        self.X_cat = np.load(d / "X_cat.npy")
        self.y = np.load(d / "y.npy")
        self.amt = np.load(d / "amt.npy")
        self.ts = np.load(d / "ts.npy")
        self.year = np.load(d / "year.npy")
        self.user = np.load(d / "user.npy")
        self.card_idx = np.load(d / "card_idx.npy")
        self.start = np.load(d / "start.npy").astype("int64")
        self.dir = d
        self.n = len(self.y)
        self.seq_len = int(self.meta["seq_len"])

    # ------------------------------------------------------------------ división temporal
    def split_idx(self, name: str) -> np.ndarray:
        sp = self.meta["split"]
        if name == "train":
            mask = self.year <= sp["train_end_year"]
        elif name == "val":
            mask = np.isin(self.year, sp["val_years"])
        elif name == "test":
            mask = np.isin(self.year, sp["test_years"])
        else:
            raise ValueError(name)
        return np.flatnonzero(mask)

    def train_sample(self, neg_per_pos: int, seed: int) -> np.ndarray:
        """Todos los fraudes de entrenamiento + `neg_per_pos` transacciones legítimas por fraude."""
        idx = self.split_idx("train")
        pos, neg = idx[self.y[idx] == 1], idx[self.y[idx] == 0]
        rng = np.random.default_rng(seed)
        neg = rng.choice(neg, size=min(len(neg), neg_per_pos * len(pos)), replace=False)
        return np.sort(np.concatenate([pos, neg]))

    def val_sample(self, n_neg: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Muestra de validación para early stopping: todos los fraudes + `n_neg` legítimas,
        con pesos que devuelven la proporción real."""
        idx = self.split_idx("val")
        pos, neg = idx[self.y[idx] == 1], idx[self.y[idx] == 0]
        rng = np.random.default_rng(seed)
        take = min(len(neg), n_neg)
        neg_s = rng.choice(neg, size=take, replace=False)
        out = np.concatenate([pos, neg_s])
        w = np.concatenate([np.ones(len(pos)), np.full(take, len(neg) / take)])
        o = np.argsort(out)
        return out[o], w[o]

    # ------------------------------------------------------------------ lotes
    def tabular(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Variables de las transacciones `idx` (índices ordenados para leer el memmap en orden)."""
        return np.asarray(self.X_num[idx]), self.X_cat[idx].astype("int64")

    def sequences(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Secuencias (B, K, ·) con las K-1 transacciones previas de la tarjeta y la actual al final.
        Las posiciones anteriores al inicio del historial quedan en cero y con máscara 0."""
        k = self.seq_len
        rows = idx[:, None] - np.arange(k - 1, -1, -1)[None, :]
        mask = rows >= self.start[idx][:, None]
        rows = np.where(mask, rows, idx[:, None])
        flat = rows.ravel()
        uniq, inv = np.unique(flat, return_inverse=True)       # lee cada fila una sola vez y en orden
        num = np.asarray(self.X_num[uniq])[inv].reshape(len(idx), k, -1)
        cat = self.X_cat[uniq][inv].reshape(len(idx), k, -1).astype("int64")
        m = mask.astype("float32")
        num = num * m[:, :, None]
        cat = cat * mask[:, :, None]
        return num, cat, m
