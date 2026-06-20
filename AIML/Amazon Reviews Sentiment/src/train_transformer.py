"""
Phase 6 + 7 + 8 — Transformer fine-tuning: DistilBERT and BERT (3-class sentiment).

Uses the SAME canonical modeling frame + data/split.npz as the baseline, so all
three models are evaluated on the identical held-out test set.

- Tokenizes the RAW reviewText (max 256 tokens).
- Fine-tunes distilbert-base-uncased then bert-base-uncased on GPU (fp16).
- Metrics: accuracy, macro/weighted F1, macro precision/recall, ROC-AUC (OVR).
- Saves models/<name>/, per-model confusion matrices, and transformer_metrics.json.

TRAIN_CAP subsamples the training split for a tractable first run; raise/remove it
to use all data. Test set is always full.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from train_baseline import LABELS, load_modeling_frame

BASE = r"C:\Working\AI ML Projetcs\Amazon Reviews Sentiment"
FIG = os.path.join(BASE, "outputs", "figures")
METRICS = os.path.join(BASE, "outputs", "metrics")
MODELS = os.path.join(BASE, "models")

TRAIN_CAP = 100_000      # subsample train split for speed; test stays full
MAX_LEN = 256
SEED = 42
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

# per-model hyperparameters (Phase 8 tuning knobs)
CONFIGS = {
    "distilbert": dict(ckpt="distilbert-base-uncased", lr=3e-5, batch=32,
                       epochs=2, weight_decay=0.01, fig="09_confusion_distilbert.png"),
    "bert": dict(ckpt="bert-base-uncased", lr=2e-5, batch=16,
                 epochs=2, weight_decay=0.01, fig="10_confusion_bert.png"),
}


class ReviewDS(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc = enc
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def metrics_from_logits(logits, y_true):
    proba = torch.softmax(torch.tensor(logits), dim=1).numpy()
    pred = proba.argmax(1)
    y_bin = label_binarize(y_true, classes=list(range(len(LABELS))))
    return {
        "test_accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "test_f1_macro": round(float(f1_score(y_true, pred, average="macro")), 4),
        "test_f1_weighted": round(float(f1_score(y_true, pred, average="weighted")), 4),
        "test_precision_macro": round(float(precision_score(y_true, pred, average="macro", zero_division=0)), 4),
        "test_recall_macro": round(float(recall_score(y_true, pred, average="macro", zero_division=0)), 4),
        "test_roc_auc_ovr_macro": round(float(roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")), 4),
    }, pred


def run_model(name, cfg, train_texts, train_y, test_texts, test_y):
    print(f"\n=== Fine-tuning {name} ({cfg['ckpt']}) ===")
    tok = AutoTokenizer.from_pretrained(cfg["ckpt"])
    enc_tr = tok(train_texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    enc_te = tok(test_texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    ds_tr = ReviewDS(enc_tr, torch.tensor(train_y))
    ds_te = ReviewDS(enc_te, torch.tensor(test_y))

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["ckpt"], num_labels=len(LABELS),
        id2label={i: l for l, i in LABEL2ID.items()}, label2id=LABEL2ID)

    args = TrainingArguments(
        output_dir=os.path.join(MODELS, name, "_trainer"),
        per_device_train_batch_size=cfg["batch"],
        per_device_eval_batch_size=64,
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=200,
        report_to="none",
        seed=SEED,
        dataloader_pin_memory=True,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_te)
    trainer.train()

    out = trainer.predict(ds_te)
    m, pred = metrics_from_logits(out.predictions, test_y)
    m["model"] = name
    m["hyperparams"] = {k: cfg[k] for k in ["lr", "batch", "epochs", "weight_decay"]}
    print(json.dumps(m, indent=2))

    model.save_pretrained(os.path.join(MODELS, name))
    tok.save_pretrained(os.path.join(MODELS, name))

    cm = confusion_matrix(test_y, pred, labels=list(range(len(LABELS))), normalize="true")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Purples",
                xticklabels=LABELS, yticklabels=LABELS, cbar_kws={"label": "recall"})
    plt.title(f"Confusion matrix — {name}")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, cfg["fig"]), dpi=130, bbox_inches="tight")
    plt.close()
    return m


def main():
    assert torch.cuda.is_available(), "No CUDA GPU detected."
    print("GPU:", torch.cuda.get_device_name(0))
    df = load_modeling_frame()
    split = np.load(os.path.join(BASE, "data", "split.npz"))
    tr_idx, te_idx = split["train"], split["test"]

    rng = np.random.default_rng(SEED)
    if TRAIN_CAP and len(tr_idx) > TRAIN_CAP:
        tr_idx = rng.choice(tr_idx, size=TRAIN_CAP, replace=False)

    texts = df["reviewText"].astype(str).tolist()
    y = df["sentiment"].map(LABEL2ID).to_numpy()

    train_texts = [texts[i] for i in tr_idx]
    test_texts = [texts[i] for i in te_idx]
    train_y = y[tr_idx].tolist()
    test_y = y[te_idx]
    print(f"Train {len(train_texts):,} | Test {len(test_texts):,}")

    results = {}
    for name, cfg in CONFIGS.items():
        results[name] = run_model(name, cfg, train_texts, train_y, test_texts, test_y)

    with open(os.path.join(METRICS, "transformer_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved transformer_metrics.json")


if __name__ == "__main__":
    main()
