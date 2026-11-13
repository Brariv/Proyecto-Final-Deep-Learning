"""Cálculo de variables transacción por transacción, para el servicio en tiempo real.

Replica exactamente `prepare.py` (que trabaja por lotes), pero manteniendo en memoria el estado
de cada tarjeta. `tests/test_online_parity.py` comprueba que ambos caminos coinciden, para que
el modelo vea en producción lo mismo que vio al entrenar.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import schema as S


@dataclass
class Transaction:
    """Una transacción tal como llega al servicio de autorización."""
    ts: int                      # minutos desde 1970-01-01
    amount: float                # USD; negativo = devolución
    channel: str                 # uno de schema.CHANNELS
    merchant: str                # identificador del comercio
    mcc: int                     # categoría del comercio
    loc: str                     # estado de EE. UU. (2 letras), país, o "ONLINE"
    errors: list[str] = field(default_factory=list)


class Vocab:
    """Traduce canal, MCC y ubicación a los índices que usan los embeddings."""

    def __init__(self, meta: dict):
        self.loc_vocab: list[str] = meta["loc_vocab"]
        self.mcc_vocab: list[int] = meta["mcc_vocab"]
        self._loc = {name: i for i, name in enumerate(self.loc_vocab)}
        self._mcc = {int(m): i for i, m in enumerate(self.mcc_vocab)}
        self._chan = {name: i for i, name in enumerate(S.CHANNELS)}

    def loc(self, name: str) -> int:
        return self._loc.get(name, S.LOC_OTHER)

    def mcc(self, code: int) -> int:
        return self._mcc.get(int(code), S.MCC_OTHER)

    def channel(self, name: str) -> int:
        return self._chan[name]


class CardState:
    """Historial resumido de una tarjeta: lo mínimo para calcular las variables de la siguiente compra."""

    def __init__(self, credit_limit: float, card_type: str, has_chip: bool, home_state: str,
                 seq_len: int = 16, info: dict | None = None):
        self.credit_limit = max(float(credit_limit), 1.0)
        self.card_type = card_type
        self.has_chip = bool(has_chip)
        self.home_state = home_state
        self.seq_len = seq_len
        self.info = info or {}                       # datos solo para mostrar en la interfaz
        self.n = 0                                   # transacciones previas
        self.sum_abs = 0.0                           # suma histórica de |monto|
        self.n_online = 0
        self.merchants: set[str] = set()
        self.mccs: set[int] = set()
        self.last_loc_idx = -1
        self.loc_counts: dict[str, int] = {}         # compras previas por ubicación
        self.recent: deque = deque(maxlen=S.WINDOW_MAX_TX)        # (ts, |monto|) de las últimas 64
        self.hist_num: deque = deque(maxlen=seq_len - 1)          # variables de las últimas K-1
        self.hist_cat: deque = deque(maxlen=seq_len - 1)

    # ------------------------------------------------------------------ variables
    def features(self, tx: Transaction, vocab: Vocab) -> tuple[np.ndarray, np.ndarray]:
        """Variables de `tx` usando solo el historial previo. No modifica el estado."""
        x = np.zeros(S.N_NUM, dtype="float64")
        c = S.NUM_INDEX
        amt, abs_amt = float(tx.amount), abs(float(tx.amount))
        online = tx.loc == S.ONLINE_LOC
        loc_idx = vocab.loc(tx.loc)

        x[c["log_amt"]] = math.copysign(math.log1p(abs_amt), amt) if amt != 0 else 0.0
        x[c["is_refund"]] = amt < 0
        hour = (tx.ts % 1440) / 60.0
        dow = ((tx.ts // 1440) + 3) % 7
        x[c["hour_sin"]] = math.sin(2 * math.pi * hour / 24)
        x[c["hour_cos"]] = math.cos(2 * math.pi * hour / 24)
        x[c["dow_sin"]] = math.sin(2 * math.pi * dow / 7)
        x[c["dow_cos"]] = math.cos(2 * math.pi * dow / 7)
        for label in tx.errors:
            x[c[S.ERROR_FLAGS[label]]] = 1.0
        x[c["is_foreign"]] = S.is_foreign_loc(tx.loc)
        x[c["away_from_home"]] = (not online) and tx.loc != self.home_state

        dt_prev = tx.ts - self.recent[-1][0] if self.n >= 1 else S.DT_CAP_MIN
        dt_5 = tx.ts - self.recent[-5][0] if self.n >= 5 else S.DT_CAP_MIN
        x[c["log_dt_prev"]] = math.log1p(min(max(dt_prev, 0), S.DT_CAP_MIN))
        x[c["log_dt_5"]] = math.log1p(min(max(dt_5, 0), S.DT_CAP_MIN))
        prior_mean = self.sum_abs / self.n if self.n > 0 else abs_amt
        x[c["log_amt_ratio"]] = math.log((abs_amt + 1.0) / (prior_mean + 1.0))
        x[c["log_prior_count"]] = math.log1p(self.n)
        x[c["merchant_new"]] = str(tx.merchant) not in self.merchants
        x[c["mcc_new"]] = int(tx.mcc) not in self.mccs
        x[c["state_changed"]] = self.n >= 1 and loc_idx != self.last_loc_idx
        seen_here = self.loc_counts.get(tx.loc, 0)
        x[c["loc_new"]] = seen_here == 0
        x[c["loc_share_prior"]] = seen_here / self.n if self.n > 0 else 0.0
        x[c["online_share_prior"]] = self.n_online / self.n if self.n > 0 else 0.0
        window = [a for t, a in self.recent if t >= tx.ts - S.WINDOW_MIN]
        x[c["log_cnt_24h"]] = math.log1p(min(len(window), S.CNT_CLIP))
        x[c["log_sum_24h"]] = math.log1p(max(sum(window), 0.0))

        x[c["log_credit_limit"]] = math.log1p(self.credit_limit)
        x[c["amt_over_limit"]] = min(abs_amt / self.credit_limit, S.AMT_OVER_LIMIT_CLIP)
        x[c["card_credit"]] = self.card_type == "Credit"
        x[c["card_prepaid"]] = self.card_type == "Debit (Prepaid)"
        x[c["card_has_chip"]] = self.has_chip
        x[c["no_history"]] = self.n == 0
        cat = np.array([vocab.channel(tx.channel), vocab.mcc(tx.mcc), loc_idx], dtype="int64")
        return x.astype("float32"), cat

    def sequence(self, x_num: np.ndarray, x_cat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Secuencia (K, ·) alineada a la derecha: historial reciente + la transacción actual."""
        k = self.seq_len
        num = np.zeros((k, S.N_NUM), dtype="float32")
        cat = np.zeros((k, len(S.CAT_FEATURES)), dtype="int64")
        mask = np.zeros(k, dtype="float32")
        h = len(self.hist_num)
        if h:
            num[k - 1 - h:k - 1] = np.stack(self.hist_num)
            cat[k - 1 - h:k - 1] = np.stack(self.hist_cat)
        num[-1], cat[-1] = x_num, x_cat
        mask[k - 1 - h:] = 1.0
        return num, cat, mask

    # ------------------------------------------------------------------ actualización
    def commit(self, tx: Transaction, x_num: np.ndarray, x_cat: np.ndarray) -> None:
        """Agrega la transacción al historial (se llama después de puntuarla)."""
        abs_amt = abs(float(tx.amount))
        self.n += 1
        self.sum_abs += abs_amt
        self.n_online += tx.loc == S.ONLINE_LOC
        self.merchants.add(str(tx.merchant))
        self.mccs.add(int(tx.mcc))
        self.last_loc_idx = int(x_cat[2])
        self.loc_counts[tx.loc] = self.loc_counts.get(tx.loc, 0) + 1
        self.recent.append((int(tx.ts), abs_amt))
        self.hist_num.append(np.asarray(x_num, dtype="float32"))
        self.hist_cat.append(np.asarray(x_cat, dtype="int64"))

    # ------------------------------------------------------------------ serialización
    def to_dict(self) -> dict:
        return {
            "credit_limit": self.credit_limit, "card_type": self.card_type, "has_chip": self.has_chip,
            "home_state": self.home_state, "seq_len": self.seq_len, "info": self.info,
            "n": self.n, "sum_abs": self.sum_abs, "n_online": self.n_online,
            "merchants": sorted(self.merchants), "mccs": sorted(self.mccs),
            "last_loc_idx": self.last_loc_idx, "loc_counts": self.loc_counts, "recent": [list(r) for r in self.recent],
            "hist_num": [np.round(v.astype("float64"), 6).tolist() for v in self.hist_num],
            "hist_cat": [v.tolist() for v in self.hist_cat],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CardState":
        s = cls(d["credit_limit"], d["card_type"], d["has_chip"], d["home_state"], d["seq_len"], d.get("info"))
        s.n, s.sum_abs, s.n_online = d["n"], d["sum_abs"], d["n_online"]
        s.merchants, s.mccs = set(d["merchants"]), set(d["mccs"])
        s.last_loc_idx = d["last_loc_idx"]
        s.loc_counts = dict(d.get("loc_counts", {}))
        s.recent.extend((int(t), float(a)) for t, a in d["recent"])
        s.hist_num.extend(np.asarray(v, dtype="float32") for v in d["hist_num"])
        s.hist_cat.extend(np.asarray(v, dtype="int64") for v in d["hist_cat"])
        return s

    def copy(self) -> "CardState":
        return CardState.from_dict(self.to_dict())
