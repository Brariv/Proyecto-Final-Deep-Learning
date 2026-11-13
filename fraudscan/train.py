"""Entrenamiento y predicción de los modelos de deep learning (A: MLP, B: LSTM)."""
from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from .data import Processed
from .models import build_model, count_params
from .value import Costs, best_thresholds, fraud_loss, oracle_value


def cost_weights(y: np.ndarray, amt: np.ndarray, costs: Costs, clip: tuple[float, float]) -> np.ndarray:
    """Pesos de la pérdida alineados con el negocio: un fraude pesa en proporción al dinero que se
    pierde si pasa (monto + contracargo), normalizado para que el fraude promedio pese 1."""
    w = np.ones(len(y), dtype="float32")
    pos = y == 1
    loss = fraud_loss(amt[pos], costs)
    w[pos] = np.clip(loss / loss.mean(), clip[0], clip[1])
    return w


def _to(device, *arrays):
    return [torch.from_numpy(np.ascontiguousarray(a)).to(device) for a in arrays]


def _forward(model, data: Processed, idx: np.ndarray, device: str) -> torch.Tensor:
    """Arma el lote según el tipo de modelo y devuelve los logits."""
    if model.kind == "mlp":
        x_num, x_cat = data.tabular(idx)
        return model(*_to(device, x_num, x_cat))
    x_num, x_cat, mask = data.sequences(idx)
    return model(*_to(device, x_num, x_cat, mask))


@torch.no_grad()
def predict(model, data: Processed, idx: np.ndarray, device: str, batch_size: int = 8192,
            progress: bool = False) -> np.ndarray:
    """Score (probabilidad sin calibrar, entre 0 y 1) para las transacciones `idx`."""
    model.eval()
    out = np.empty(len(idx), dtype="float32")
    t0 = time.time()
    for i in range(0, len(idx), batch_size):
        b = idx[i:i + batch_size]
        out[i:i + len(b)] = torch.sigmoid(_forward(model, data, b, device)).float().cpu().numpy()
        if progress and (i // batch_size) % 100 == 0:
            print(f"    predicción {i:,}/{len(idx):,} ({time.time() - t0:.0f}s)", flush=True)
    return out


def fit(kind: str, data: Processed, cfg: dict, costs: Costs, device: str, weighted: bool = True,
        cat_use: list[str] | None = None, log=print) -> tuple[torch.nn.Module, dict]:
    """Entrena un modelo y devuelve el mejor (según VALOR en validación) junto con su historial."""
    tc = cfg["train"]
    torch.manual_seed(tc["seed"])
    rng = np.random.default_rng(tc["seed"])

    train_idx = data.train_sample(tc["neg_per_pos"], tc["seed"])
    y_tr, amt_tr = data.y[train_idx], data.amt[train_idx]
    w_tr = cost_weights(y_tr, amt_tr, costs, tuple(tc["weight_clip"])) if weighted else np.ones(len(y_tr), "float32")
    val_idx, val_w = data.val_sample(tc["val_neg_sample"], tc["seed"])
    y_val, amt_val = data.y[val_idx], data.amt[val_idx]
    val_oracle = oracle_value(y_val, amt_val, costs, val_w)

    cat_use = cfg["features"]["cat_use"] if cat_use is None else cat_use
    model = build_model(kind, data.meta, tc, cat_use).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    steps = int(np.ceil(len(train_idx) / tc["batch_size"])) * tc["epochs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], total_steps=steps)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    log(f"  {kind}: {count_params(model):,} parámetros | {len(train_idx):,} filas de entrenamiento "
        f"({int(y_tr.sum()):,} fraudes) | dispositivo: {device}")

    history, best_state, best_value, bad = [], None, -np.inf, 0
    t_start = time.time()
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        perm = rng.permutation(len(train_idx))
        total, seen = 0.0, 0
        for i in range(0, len(perm), tc["batch_size"]):
            p = np.sort(perm[i:i + tc["batch_size"]])
            logits = _forward(model, data, train_idx[p], device)
            yb, wb = _to(device, y_tr[p].astype("float32"), w_tr[p])
            loss = (loss_fn(logits, yb) * wb).sum() / wb.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total += float(loss.detach()) * len(p)
            seen += len(p)

        s_val = predict(model, data, val_idx, device)
        eps = 1e-7
        val_loss = float(np.average(-(y_val * np.log(s_val + eps) + (1 - y_val) * np.log(1 - s_val + eps)),
                                    weights=val_w))
        val_ap = float(average_precision_score(y_val, s_val, sample_weight=val_w))
        val_value = best_thresholds(s_val, y_val, amt_val, costs, val_w)["value"]
        row = {"epoch": epoch, "train_loss": total / seen, "val_loss": val_loss, "val_ap": val_ap,
               "val_value_pct": val_value / val_oracle, "seconds": time.time() - t_start}
        history.append(row)
        log(f"  época {epoch}: pérdida train {row['train_loss']:.4f} | pérdida val {val_loss:.5f} | "
            f"AP val {val_ap:.3f} | valor val {100 * row['val_value_pct']:.1f} % del máximo | {row['seconds']:.0f}s")
        if val_value > best_value:
            best_value, bad = val_value, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= tc["patience"]:
                log(f"  early stopping en la época {epoch}")
                break

    model.load_state_dict(best_state)
    info = {"kind": kind, "weighted_loss": weighted, "cat_use": cat_use, "params": count_params(model),
            "train_rows": int(len(train_idx)), "train_seconds": time.time() - t_start,
            "device": device, "history": history,
            "best_epoch": int(np.argmax([h["val_value_pct"] for h in history])) + 1}
    return model.eval(), info
