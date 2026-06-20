"""
Phase 6 - Models 2 & 3: DistilBERT / BERT fine-tuning (GPU)
AI-Powered Telecom Fraud, Phishing & SMS Spam Detection System

Usage:
    python src/06_model_transformers.py distilbert-base-uncased distilbert
    python src/06_model_transformers.py bert-base-uncased bert

- Same stratified split as the baseline (utils.load_split)
- Class-weighted cross-entropy to handle imbalance
- Reports accuracy / precision / recall / F1 (macro)
- Saves metrics -> reports/metrics_<tag>.json, confusion -> outputs/figures
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support,
                             classification_report)
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import load_split, CLASSES, SEED

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
REP = ROOT / "reports"
MODELS = ROOT / "outputs" / "models"
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}


def make_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0)
        return {"accuracy": accuracy_score(labels, preds),
                "precision_macro": p, "recall_macro": r, "f1_macro": f1}
    return compute_metrics


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "distilbert-base-uncased"
    tag = sys.argv[2] if len(sys.argv) > 2 else "distilbert"
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 2e-5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== {tag} ({model_name}) on {device} | epochs={epochs} lr={lr} ===")

    train_df, test_df = load_split()
    for d in (train_df, test_df):
        d["label_id"] = d["label5"].map(LABEL2ID)

    tok = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=128)

    ds_tr = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
    ds_te = Dataset.from_pandas(test_df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
    ds_tr = ds_tr.map(tokenize, batched=True, remove_columns=["text"])
    ds_te = ds_te.map(tokenize, batched=True, remove_columns=["text"])

    # class weights for imbalance
    cw = compute_class_weight("balanced", classes=np.arange(len(CLASSES)),
                              y=train_df["label_id"].values)
    class_weights = torch.tensor(cw, dtype=torch.float, device=device)
    print("class weights:", {CLASSES[i]: round(float(w), 2) for i, w in enumerate(cw)})

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(CLASSES), id2label=ID2LABEL, label2id=LABEL2ID)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=class_weights)
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=str(MODELS / f"_{tag}_ckpt"),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=lr,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
        seed=SEED,
        fp16=(device == "cuda"),
    )

    from transformers import DataCollatorWithPadding
    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=ds_tr, eval_dataset=ds_te,
        compute_metrics=make_compute_metrics(),
        data_collator=DataCollatorWithPadding(tok),
        processing_class=tok,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print("\n=== TEST metrics ===")
    for k, v in metrics.items():
        if k.startswith("eval_"):
            print(f"  {k[5:]}: {v:.4f}" if isinstance(v, float) else f"  {k[5:]}: {v}")

    pred_logits = trainer.predict(ds_te).predictions
    preds = np.argmax(pred_logits, axis=-1)
    y_true = test_df["label_id"].values
    print("\n", classification_report(y_true, preds, target_names=CLASSES, digits=3, zero_division=0))

    # binary view
    bin_true = (test_df["binary_label"] == "spam").astype(int).values
    bin_pred = (preds != LABEL2ID["Normal"]).astype(int)
    bin_acc = accuracy_score(bin_true, bin_pred)
    bin_f1 = f1_score(bin_true, bin_pred)

    cm = confusion_matrix(y_true, preds, labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(f"{tag} Confusion (acc={metrics['eval_accuracy']:.3f}, "
                 f"F1m={metrics['eval_f1_macro']:.3f})", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(FIG / f"07_{tag}_confusion.png", dpi=130)
    plt.close(fig)

    out = {
        "model": tag, "hf_name": model_name, "epochs": epochs, "lr": lr,
        "test_accuracy": float(metrics["eval_accuracy"]),
        "test_f1_macro": float(metrics["eval_f1_macro"]),
        "test_precision_macro": float(metrics["eval_precision_macro"]),
        "test_recall_macro": float(metrics["eval_recall_macro"]),
        "binary_accuracy": float(bin_acc), "binary_f1": float(bin_f1),
        "report": classification_report(y_true, preds, target_names=CLASSES,
                                        output_dict=True, zero_division=0),
    }
    (REP / f"metrics_{tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved metrics -> reports/metrics_{tag}.json")
    print(f"Saved confusion -> outputs/figures/07_{tag}_confusion.png")


if __name__ == "__main__":
    main()
