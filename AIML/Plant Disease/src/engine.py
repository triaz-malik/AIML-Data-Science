"""Shared training / evaluation engine (used by Phases 5-7).

A single ``fit`` function trains any model from ``models.build_model`` with:
  * mixed-precision (AMP) on CUDA for speed
  * inverse-frequency class weights
  * cosine LR schedule
  * best-checkpoint selection on validation macro-F1

Kept dependency-light so Optuna (Phase 7) can call ``fit`` directly.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from . import config as C
from .models import build_model


def make_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    raise ValueError(f"Unknown optimizer '{name}'")


@torch.no_grad()
def evaluate(model, loader, device=C.DEVICE, return_preds=False):
    model.eval()
    ys, ps, probs = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            logits = model(x)
        p = logits.softmax(1).float().cpu().numpy()
        probs.append(p)
        ps.append(p.argmax(1))
        ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    prob = np.concatenate(probs)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    if return_preds:
        return acc, f1m, y_true, y_pred, prob
    return acc, f1m


@dataclass
class FitResult:
    model_name: str
    best_val_acc: float
    best_val_f1: float
    history: dict = field(default_factory=dict)
    ckpt_path: str = ""
    seconds: float = 0.0


def fit(
    model_name: str,
    train_loader,
    val_loader,
    n_classes: int,
    class_weights: np.ndarray | None = None,
    *,
    epochs: int = C.EPOCHS,
    lr: float = C.LR,
    weight_decay: float = C.WEIGHT_DECAY,
    optimizer: str = "adamw",
    dropout: float | None = None,
    label_smoothing: float = C.LABEL_SMOOTHING,
    device: str = C.DEVICE,
    save_path: str | None = None,
    le=None,
    verbose: bool = True,
) -> FitResult:
    model = build_model(model_name, n_classes, dropout=dropout).to(device)

    weight = None
    if class_weights is not None:
        weight = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)

    opt = make_optimizer(optimizer, model.parameters(), lr, weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    history = {"train_loss": [], "val_acc": [], "val_f1": [], "lr": []}
    best_f1, best_state = -1.0, None
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
        sched.step()

        train_loss = running / len(train_loader.dataset)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(opt.param_groups[0]["lr"])

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"  [{model_name}] epoch {ep:02d}/{epochs}  "
                  f"loss={train_loss:.3f}  val_acc={val_acc:.4f}  val_f1={val_f1:.4f}",
                  flush=True)

    ckpt_path = ""
    if save_path and best_state is not None:
        payload = {
            "model_name": model_name,
            "state_dict": best_state,
            "n_classes": n_classes,
            "dropout": dropout,
            "classes": list(le.classes_) if le is not None else None,
            "img_size": C.IMG_SIZE,
            "val_f1": best_f1,
            "val_acc": best_val_acc,
        }
        torch.save(payload, save_path)
        ckpt_path = str(save_path)

    return FitResult(
        model_name=model_name,
        best_val_acc=float(best_val_acc),
        best_val_f1=float(best_f1),
        history=history,
        ckpt_path=ckpt_path,
        seconds=time.time() - t0,
    )
