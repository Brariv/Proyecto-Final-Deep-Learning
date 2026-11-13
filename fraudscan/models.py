"""Arquitecturas: modelo A (MLP sobre la transacción) y modelo B (LSTM sobre la secuencia de la tarjeta)."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class TxEncoder(nn.Module):
    """Convierte una transacción en un vector: variables numéricas estandarizadas + embeddings
    de canal, categoría de comercio (MCC) y ubicación."""

    def __init__(self, n_num: int, cat_sizes: list[int], emb_dim: int, mean: list[float], std: list[float],
                 cat_use: list[int] | None = None):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        # cat_use: columnas categóricas que entran como embedding (las demás se ignoran).
        self.cat_use = list(range(len(cat_sizes))) if cat_use is None else list(cat_use)
        self.embs = nn.ModuleList([nn.Embedding(cat_sizes[i], min(emb_dim, max(2, cat_sizes[i] // 2)))
                                   for i in self.cat_use])
        self.out_dim = n_num + sum(e.embedding_dim for e in self.embs)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        x = (x_num - self.mean) / self.std
        embs = [emb(x_cat[..., i]) for i, emb in zip(self.cat_use, self.embs)]
        return torch.cat([x, *embs], dim=-1)


class MLPModel(nn.Module):
    """Modelo A: red densa que mira solo la transacción actual (con sus variables de historial)."""
    kind = "mlp"

    def __init__(self, n_num, cat_sizes, mean, std, emb_dim=8, hidden=(128, 64), dropout=0.1, cat_use=None):
        super().__init__()
        self.config = dict(n_num=n_num, cat_sizes=list(cat_sizes), mean=list(mean), std=list(std),
                           emb_dim=emb_dim, hidden=list(hidden), dropout=dropout, cat_use=cat_use)
        self.encoder = TxEncoder(n_num, cat_sizes, emb_dim, mean, std, cat_use)
        layers, d = [], self.encoder.out_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        return self.net(self.encoder(x_num, x_cat)).squeeze(-1)


class LSTMModel(nn.Module):
    """Modelo B: una LSTM lee las últimas K transacciones de la tarjeta (la actual es la última) y
    su estado final se combina con la transacción actual para decidir."""
    kind = "lstm"

    def __init__(self, n_num, cat_sizes, mean, std, emb_dim=8, hidden=64, head=64, dropout=0.1, cat_use=None):
        super().__init__()
        self.config = dict(n_num=n_num, cat_sizes=list(cat_sizes), mean=list(mean), std=list(std),
                           emb_dim=emb_dim, hidden=hidden, head=head, dropout=dropout, cat_use=cat_use)
        self.encoder = TxEncoder(n_num, cat_sizes, emb_dim, mean, std, cat_use)
        d = self.encoder.out_dim
        self.proj = nn.Sequential(nn.Linear(d, hidden), nn.ReLU())
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + d, head), nn.ReLU(), nn.Dropout(dropout), nn.Linear(head, 1))

    def forward(self, x_num, x_cat, mask):
        e = self.encoder(x_num, x_cat) * mask.unsqueeze(-1)     # el relleno entra como ceros
        out, _ = self.lstm(self.proj(e) * mask.unsqueeze(-1))
        return self.head(torch.cat([out[:, -1], e[:, -1]], dim=-1)).squeeze(-1)


MODELS = {"mlp": MLPModel, "lstm": LSTMModel}


def build_model(kind: str, meta: dict, train_cfg: dict, cat_use: list[str] | None = None) -> nn.Module:
    """cat_use: nombres de las variables categóricas que el modelo puede usar (por defecto, todas)."""
    use = None if cat_use is None else [meta["cat_features"].index(name) for name in cat_use]
    common = dict(n_num=len(meta["num_features"]), cat_sizes=meta["cat_sizes"], mean=meta["scaler_mean"],
                  std=meta["scaler_std"], emb_dim=train_cfg["emb_dim"], dropout=train_cfg["dropout"], cat_use=use)
    if kind == "mlp":
        return MLPModel(hidden=train_cfg["mlp_hidden"], **common)
    if kind == "lstm":
        return LSTMModel(hidden=train_cfg["lstm_hidden"], **common)
    raise ValueError(kind)


def save_model(model: nn.Module, path: str | Path, extra: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": model.kind, "config": model.config,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "extra": extra or {}}, path)


def load_model(path: str | Path, device: str = "cpu") -> nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MODELS[ckpt["kind"]](**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.extra = ckpt.get("extra", {})
    return model.to(device).eval()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def pick_device(prefer: str | None = None) -> str:
    """cuda (GPU NVIDIA) si existe; si no, mps (Mac) y por último cpu."""
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
