"""Reconstruye transacciones "crudas" a partir de los arreglos procesados y las reproduce por el
camino en línea. Lo usan la prueba de paridad y la exportación de datos para el demo."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import schema as S
from .data import Processed
from .online import CardState, Transaction, Vocab
from .prepare import load_cards


class RawStore:
    """Campos originales de cada transacción (comercio, ubicación, errores) en el orden procesado."""

    def __init__(self, data: Processed, ibm_dir: str | Path):
        d = data.dir
        self.data = data
        self.merchant = np.load(d / "merchant.npy", mmap_mode="r")
        self.mcc = np.load(d / "mcc_raw.npy", mmap_mode="r")
        self.loc = np.load(d / "loc_raw.npy", mmap_mode="r")
        self.err = np.load(d / "err_raw.npy", mmap_mode="r")
        self.loc_names = data.meta["loc_names_raw"]
        self.err_names = data.meta["err_names_raw"]
        self.vocab = Vocab(data.meta)
        cards = load_cards(Path(ibm_dir))
        self.cards = cards.set_index(cards["user"].astype("int64") * 100 + cards["card"])
        self.card_keys = data.meta["card_keys"]

    def transaction(self, row: int) -> Transaction:
        err = self.err_names[int(self.err[row])]
        return Transaction(
            ts=int(self.data.ts[row]), amount=float(self.data.amt[row]),
            channel=S.CHANNELS[int(self.data.X_cat[row, 0])], merchant=str(int(self.merchant[row])),
            mcc=int(self.mcc[row]), loc=self.loc_names[int(self.loc[row])],
            errors=[e for e in err.split(",") if e])

    def card_rows(self, card_idx: int) -> np.ndarray:
        lo = np.searchsorted(self.data.card_idx, card_idx, side="left")
        hi = np.searchsorted(self.data.card_idx, card_idx, side="right")
        return np.arange(lo, hi)

    def new_state(self, card_idx: int) -> CardState:
        key = self.card_keys[card_idx]
        c = self.cards.loc[key]
        info = {"card_key": int(key), "user": int(c["user"]), "card": int(c["card"]), "brand": c["brand"],
                "home_city": c["home_city"]}
        return CardState(c["credit_limit"], c["card_type"], bool(c["has_chip"]), c["home_state"],
                         seq_len=self.data.seq_len, info=info)

    def replay(self, card_idx: int, until_row: int | None = None, collect: bool = False):
        """Pasa el historial de la tarjeta por el camino en línea hasta `until_row` (sin incluirla).
        Con collect=True devuelve también las variables calculadas, para compararlas con el lote."""
        state = self.new_state(card_idx)
        feats = []
        for row in self.card_rows(card_idx):
            if until_row is not None and row >= until_row:
                break
            tx = self.transaction(int(row))
            x_num, x_cat = state.features(tx, self.vocab)
            if collect:
                feats.append((int(row), x_num, x_cat))
            state.commit(tx, x_num, x_cat)
        return (state, feats) if collect else state
