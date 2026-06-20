"""
Fine-tune a transformer (DistilBERT or BERT) for complaint classification.

Usage:
    python src/train_transformer.py distilbert-base-uncased distilbert
    python src/train_transformer.py bert-base-uncased bert

Set env SMOKE=1 to run a fast 1-epoch sanity check on a small subset.
Evaluates on the SAME held-out test split saved by train_baseline.py.
"""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments)

BASE = r"C:\Working\AI ML Projetcs\Consumer Complaints"
DATA = os.path.join(BASE, "data", "features.parquet")
FIG = os.path.join(BASE, "outputs", "figures")
METRICS = os.path.join(BASE, "outputs", "metrics")

MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "distilbert-base-uncased"
SHORT = sys.argv[2] if len(sys.argv) > 2 else "distilbert"
SMOKE = os.environ.get("SMOKE") == "1"

MAX_LEN = 256
EPOCHS = 1 if SMOKE else 2
BATCH = 32 if "distil" in MODEL_NAME else 16
LR = 3e-5


def main():
    df = pd.read_parquet(DATA, columns=["narrative", "category"])
    df["narrative"] = df["narrative"].astype(str)
    labels = sorted(df["category"].astype(str).unique())
    lab2id = {l: i for i, l in enumerate(labels)}
    df["label"] = df["category"].astype(str).map(lab2id)

    split = np.load(os.path.join(BASE, "data", "split.npz"))
    tr_idx, te_idx = split["train"], split["test"]
    train_df = df.iloc[tr_idx][["narrative", "label"]].reset_index(drop=True)
    test_df = df.iloc[te_idx][["narrative", "label"]].reset_index(drop=True)

    if SMOKE:
        train_df = train_df.sample(2000, random_state=0).reset_index(drop=True)
        test_df = test_df.sample(1000, random_state=0).reset_index(drop=True)
    print(f"[{SHORT}] train {len(train_df):,} | test {len(test_df):,} | "
          f"epochs {EPOCHS} | batch {BATCH} | device {torch.cuda.get_device_name(0)}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tok(batch["narrative"], truncation=True, max_length=MAX_LEN)

    train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True, remove_columns=["narrative"])
    test_ds = Dataset.from_pandas(test_df).map(tokenize, batched=True, remove_columns=["narrative"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels),
        id2label={i: l for l, i in lab2id.items()}, label2id=lab2id)

    use_bf16 = torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=os.path.join(BASE, "outputs", f"_ckpt_{SHORT}"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=64,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.06,
        bf16=use_bf16, fp16=not use_bf16,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=100,
        report_to="none",
        dataloader_num_workers=0,   # Windows: worker processes stall the loop; 0 is fastest here
        dataloader_pin_memory=True,
    )

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        return {
            "accuracy": accuracy_score(p.label_ids, preds),
            "f1_weighted": f1_score(p.label_ids, preds, average="weighted"),
            "f1_macro": f1_score(p.label_ids, preds, average="macro"),
        }

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=test_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        processing_class=tok,
    )

    t0 = time.time()
    trainer.train()
    train_secs = round(time.time() - t0, 1)

    out = trainer.predict(test_ds)
    preds = out.predictions.argmax(-1)
    yte = out.label_ids

    metrics = {
        "model": MODEL_NAME,
        "params": {"max_len": MAX_LEN, "epochs": EPOCHS, "batch_size": BATCH,
                   "learning_rate": LR, "weight_decay": 0.01, "warmup_ratio": 0.06,
                   "precision": "bf16" if use_bf16 else "fp16"},
        "train_seconds": train_secs,
        "test_accuracy": round(float(accuracy_score(yte, preds)), 4),
        "test_f1_macro": round(float(f1_score(yte, preds, average="macro")), 4),
        "test_f1_weighted": round(float(f1_score(yte, preds, average="weighted")), 4),
        "test_precision_weighted": round(float(precision_score(yte, preds, average="weighted", zero_division=0)), 4),
        "test_recall_weighted": round(float(recall_score(yte, preds, average="weighted")), 4),
    }
    from sklearn.metrics import classification_report
    rep = classification_report(yte, preds, target_names=labels, output_dict=True, zero_division=0)
    metrics["per_class_f1"] = {l: round(rep[l]["f1-score"], 3) for l in labels}

    suffix = "_smoke" if SMOKE else ""
    with open(os.path.join(METRICS, f"{SHORT}_metrics{suffix}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class_f1"}, indent=2))

    if not SMOKE:
        cm = confusion_matrix(yte, preds, normalize="true")
        plt.figure(figsize=(9, 7.5))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Greens",
                    xticklabels=labels, yticklabels=labels, cbar_kws={"label": "recall"})
        plt.title(f"Confusion matrix — {SHORT} (row-normalized)")
        plt.ylabel("True category"); plt.xlabel("Predicted category")
        plt.xticks(rotation=40, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, f"11_confusion_{SHORT}.png"), dpi=130, bbox_inches="tight")
        plt.close()
        print(f"saved 11_confusion_{SHORT}.png")


if __name__ == "__main__":
    main()
