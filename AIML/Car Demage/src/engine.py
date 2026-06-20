"""Shared training / evaluation engine (used by train.py and tune.py)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import DEVICE


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


def _run_epoch(model, loader, criterion, optimizer, scaler, train: bool):
    model.train() if train else model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    torch.set_grad_enabled(train)
    use_amp = DEVICE.type == "cuda"
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)
        if train:
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    torch.set_grad_enabled(True)
    return loss_sum / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 4,
    verbose: bool = True,
):
    """Train with AdamW + cosine schedule + early stopping. Returns (best_state, history, best_val_acc)."""
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=DEVICE.type == "cuda")

    history = History()
    best_val_acc, best_state, no_improve = 0.0, None, 0

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, scaler, train=True)
        va_loss, va_acc = _run_epoch(model, val_loader, criterion, optimizer, scaler, train=False)
        scheduler.step()

        history.train_loss.append(tr_loss); history.train_acc.append(tr_acc)
        history.val_loss.append(va_loss); history.val_acc.append(va_acc)

        if verbose:
            print(f"  epoch {ep:2d}/{epochs}  "
                  f"train_loss={tr_loss:.4f} acc={tr_acc:.4f}  "
                  f"val_loss={va_loss:.4f} acc={va_acc:.4f}  ({time.time()-t0:.1f}s)")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  early stopping at epoch {ep} (best val_acc={best_val_acc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_state, history, best_val_acc


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader):
    """Return (y_true, y_pred, y_prob_positive) numpy arrays. Positive class index = 1 (damage)."""
    model.eval().to(DEVICE)
    ys, preds, probs = [], [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            logits = model(x)
        p = torch.softmax(logits.float(), dim=1)
        ys.append(y.numpy())
        preds.append(p.argmax(1).cpu().numpy())
        probs.append(p[:, 1].cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)
