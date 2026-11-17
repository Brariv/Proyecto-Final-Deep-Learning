"""Explicabilidad del modelo B (LSTM) con Integrated Gradients.

El modelo se usa tal como quedó entrenado; aquí no se reentrena nada.

Idea: se atribuye el score (la probabilidad, entre 0 y 1) a la salida del codificador, que es
donde conviven las variables numéricas y los embeddings. Para una secuencia de K transacciones
el codificador produce un tensor (K, D) con D = 33 variables numéricas + 2 dimensiones del
embedding de canal + 8 del embedding de categoría. Integrated Gradients reparte el score sobre
ese tensor, así que una sola corrida da dos lecturas:

  - por variable, sumando las dimensiones que forman cada embedding;
  - por paso de la secuencia, sumando sobre D (cuánto aporta cada compra previa de la tarjeta).

La propiedad de completitud garantiza que la suma de todas las atribuciones es exactamente
score(x) - score(línea base). El error numérico de esa igualdad se reporta en cada corrida.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from captum.attr import IntegratedGradients
from torch import nn

from . import schema as S
from .data import Processed

# Dimensiones de la salida del codificador que corresponden a cada variable categórica.
CAT_SPAN: dict[str, tuple[int, int]] = {}

FEATURE_LABELS = {
    "log_amt": "Monto (log)",
    "is_refund": "Es devolución",
    "hour_sin": "Hora (seno)", "hour_cos": "Hora (coseno)",
    "dow_sin": "Día de semana (seno)", "dow_cos": "Día de semana (coseno)",
    "err_bad_pin": "Error: PIN", "err_bad_cvv": "Error: CVV", "err_bad_card": "Error: número",
    "err_bad_exp": "Error: vencimiento", "err_bad_zip": "Error: código postal",
    "err_insufficient": "Error: saldo", "err_glitch": "Error: falla técnica",
    "is_foreign": "Comercio en el extranjero",
    "away_from_home": "Fuera del estado del titular",
    "log_dt_prev": "Tiempo desde la compra anterior",
    "log_dt_5": "Tiempo desde la quinta anterior",
    "log_amt_ratio": "Monto frente al habitual",
    "log_prior_count": "Compras previas de la tarjeta",
    "merchant_new": "Comercio nuevo para la tarjeta",
    "mcc_new": "Categoría nueva para la tarjeta",
    "state_changed": "Cambió de ubicación",
    "loc_new": "Ubicación nueva para la tarjeta",
    "loc_share_prior": "Historial en esa ubicación",
    "online_share_prior": "Proporción histórica en línea",
    "log_cnt_24h": "Compras en las últimas 24 h",
    "log_sum_24h": "Monto acumulado en 24 h",
    "log_credit_limit": "Límite de la tarjeta",
    "amt_over_limit": "Monto sobre el límite",
    "card_credit": "Tarjeta de crédito", "card_prepaid": "Tarjeta prepago",
    "card_has_chip": "Tarjeta con chip", "no_history": "Primera compra de la tarjeta",
    "channel": "Canal de pago", "mcc": "Categoría del comercio",
}


def label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


# ------------------------------------------------------------------ modelo partido en dos
class Tail(nn.Module):
    """La parte del modelo que va después del codificador. Toma la salida del codificador sin
    enmascarar y reproduce exactamente `LSTMModel.forward`."""

    def __init__(self, model):
        super().__init__()
        self.proj, self.lstm, self.head = model.proj, model.lstm, model.head

    def forward(self, e_raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        e = e_raw * mask.unsqueeze(-1)
        out, _ = self.lstm(self.proj(e) * mask.unsqueeze(-1))
        return self.head(torch.cat([out[:, -1], e[:, -1]], dim=-1)).squeeze(-1)


class Explainer:
    """Calcula atribuciones para el modelo B sobre lotes de secuencias."""

    def __init__(self, model, meta: dict, vocab_meta: dict, device: str = "cpu"):
        if model.kind != "lstm":
            raise ValueError("Este explicador es para el modelo B (LSTM).")
        self.model = model.to(device).eval()
        self.tail = Tail(model).to(device).eval()
        self.device = device
        self.meta = meta
        self.num_features = meta["num_features"]
        self.n_num = len(self.num_features)
        self.cat_use = model.config["cat_use"] or list(range(len(meta["cat_sizes"])))
        self.seq_len = int(meta["seq_len"])

        # Dónde cae cada embedding dentro de la salida del codificador.
        self.spans, offset = {}, self.n_num
        for i, emb in zip(self.cat_use, model.encoder.embs):
            name = meta["cat_features"][i]
            self.spans[name] = (offset, offset + emb.embedding_dim)
            offset += emb.embedding_dim
        self.out_dim = offset
        assert self.out_dim == model.encoder.out_dim

        # Transacción "neutra": promedio de entrenamiento en lo continuo, 0 en lo binario y la
        # categoría más frecuente en lo categórico. Se calcula en 06_export_demo.py.
        self.base_num = np.asarray(vocab_meta["num_baseline"], dtype="float32")
        self.base_cat = np.asarray(vocab_meta["cat_baseline"], dtype="int64")
        self.ig = IntegratedGradients(self._score)

    # ------------------------------------------------------------------ utilidades
    def _score(self, e_raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Score del modelo (probabilidad) a partir de la salida del codificador."""
        return torch.sigmoid(self.tail(e_raw, mask))

    def encode(self, x_num: np.ndarray, x_cat: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return self.model.encoder(torch.from_numpy(np.ascontiguousarray(x_num)).to(self.device),
                                      torch.from_numpy(np.ascontiguousarray(x_cat)).to(self.device))

    def neutral_encoding(self, n: int) -> torch.Tensor:
        """Codificación de una secuencia entera de transacciones neutras, (n, K, D)."""
        num = np.broadcast_to(self.base_num, (n, self.seq_len, self.n_num))
        cat = np.broadcast_to(self.base_cat, (n, self.seq_len, len(self.base_cat)))
        return self.encode(num, cat)

    def verify_split(self, data: Processed, idx: np.ndarray) -> float:
        """Comprueba que Tail(codificador(x)) reproduce el modelo completo."""
        x_num, x_cat, mask = data.sequences(idx)
        e = self.encode(x_num, x_cat)
        m = torch.from_numpy(mask).to(self.device)
        with torch.no_grad():
            a = torch.sigmoid(self.model(torch.from_numpy(x_num).to(self.device),
                                         torch.from_numpy(x_cat).to(self.device), m))
            b = self._score(e, m)
        return float((a - b).abs().max())

    # ------------------------------------------------------------------ atribuciones
    def attribute(self, data: Processed, idx: np.ndarray, baseline: str = "neutral",
                  n_steps: int = 64, batch_size: int = 64) -> dict:
        """Integrated Gradients sobre `idx`.

        baseline:
          "neutral"  la secuencia completa se reemplaza por transacciones neutras. Responde
                     "cuánto sube el score frente a una tarjeta con historial corriente".
          "current"  se conserva el historial real y solo la compra actual se vuelve neutra.
                     Responde "cuánto aporta esta compra sobre el propio historial de la tarjeta".
        """
        if baseline not in ("neutral", "current"):
            raise ValueError(baseline)
        attrs, scores, base_scores, deltas = [], [], [], []
        t0 = time.perf_counter()
        for i in range(0, len(idx), batch_size):
            b = idx[i:i + batch_size]
            x_num, x_cat, mask = data.sequences(b)
            e = self.encode(x_num, x_cat)
            m = torch.from_numpy(mask).to(self.device)
            e_base = self.neutral_encoding(len(b))
            if baseline == "current":
                e_base = e.clone()
                e_base[:, -1] = self.neutral_encoding(len(b))[:, -1]
            a, delta = self.ig.attribute(inputs=e, baselines=e_base, additional_forward_args=(m,),
                                         n_steps=n_steps, return_convergence_delta=True)
            with torch.no_grad():
                scores.append(self._score(e, m).cpu().numpy())
                base_scores.append(self._score(e_base, m).cpu().numpy())
            attrs.append(a.cpu().numpy())
            deltas.append(delta.cpu().numpy())
        attr = np.concatenate(attrs)                      # (N, K, D)
        return {
            "attr": attr, "score": np.concatenate(scores), "base_score": np.concatenate(base_scores),
            "delta": np.concatenate(deltas), "idx": idx, "baseline": baseline, "n_steps": n_steps,
            "seconds": time.perf_counter() - t0,
        }

    def by_feature(self, attr: np.ndarray, step: int | None = -1) -> dict[str, np.ndarray]:
        """Atribución por variable. `step=-1` mira solo la compra actual; `step=None` suma la
        secuencia entera."""
        a = attr[:, step] if step is not None else attr.sum(axis=1)
        out = {name: a[:, j] for j, name in enumerate(self.num_features)}
        for name, (lo, hi) in self.spans.items():
            out[name] = a[:, lo:hi].sum(axis=1)
        return out

    @staticmethod
    def by_step(attr: np.ndarray) -> np.ndarray:
        """Atribución por paso de la secuencia, (N, K)."""
        return attr.sum(axis=2)

    # ------------------------------------------------------------------ oclusión (comparación)
    def occlusion_groups(self, data: Processed, idx: np.ndarray, groups: list,
                         batch_size: int = 64) -> dict:
        """Caída del score al neutralizar cada grupo de variables en la compra actual.

        Es el método que ya usa el servicio para mostrar razones en la interfaz; aquí sirve para
        contrastar el orden de importancia con el de Integrated Gradients.
        """
        names = [g[0] for g in groups]
        drops, t0 = [], time.perf_counter()
        for i in range(0, len(idx), batch_size):
            b = idx[i:i + batch_size]
            x_num, x_cat, mask = data.sequences(b)
            m = torch.from_numpy(mask).to(self.device)
            variants = []
            for _, num_names, cat_names in groups:
                n2, c2 = x_num.copy(), x_cat.copy()
                for nm in num_names:
                    n2[:, -1, S.NUM_INDEX[nm]] = self.base_num[S.NUM_INDEX[nm]]
                for nm in cat_names:
                    j = S.CAT_FEATURES.index(nm)
                    c2[:, -1, j] = self.base_cat[j]
                variants.append((n2, c2))
            with torch.no_grad():
                full = self._score(self.encode(x_num, x_cat), m).cpu().numpy()
                row = [full - self._score(self.encode(n2, c2), m).cpu().numpy() for n2, c2 in variants]
            drops.append(np.stack(row, axis=1))
        return {"names": names, "drop": np.concatenate(drops), "seconds": time.perf_counter() - t0}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Correlación de Spearman entre dos vectores (sin dependencias extra)."""
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), dtype="float64")
    r[order] = np.arange(len(x), dtype="float64")
    return r
