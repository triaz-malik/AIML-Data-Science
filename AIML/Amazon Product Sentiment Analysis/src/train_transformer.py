"""Fine-tune DistilBERT or BERT for binary sentiment classification.

Uses HuggingFace Transformers + the Trainer API on the *raw* review text
(the transformer tokenizer handles normalization — do not pre-clean).

    python -m src.train_transformer --model distilbert-base-uncased --sample 100000
    python -m src.train_transformer --model bert-base-uncased       --sample 100000

Outputs go to models/<short-name>/ (weights + tokenizer + metrics.json).
Requires a CUDA GPU for reasonable speed; falls back to CPU (slow).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from . import config, data

SHORT_NAMES = {
    "distilbert-base-uncased": "distilbert",
    "bert-base-uncased": "bert",
}


def _to_hf_dataset(df, tokenizer, max_len: int) -> Dataset:
    ds = Dataset.from_dict({"text": df["text"].tolist(), "label": df["label"].tolist()})

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    return ds.map(tok, batched=True, remove_columns=["text"])


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": p,
        "recall": r,
        "f1": f1,
    }


def train(
    model_name: str,
    sample_size: int | None,
    epochs: int,
    batch_size: int,
    lr: float,
    max_len: int,
    weight_decay: float,
) -> dict:
    short = SHORT_NAMES.get(model_name, model_name.split("/")[-1])
    out_dir = config.MODELS_DIR / short
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model={model_name}  device={device}  sample={sample_size}")

    train_df = data.load_split("train", sample_size=sample_size)
    test_size = None if sample_size is None else max(20_000, sample_size // 5)
    test_df = data.load_split("test", sample_size=test_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
        id2label=config.LABEL_NAMES, label2id={v: k for k, v in config.LABEL_NAMES.items()},
    )

    train_ds = _to_hf_dataset(train_df, tokenizer, max_len)
    eval_ds = _to_hf_dataset(test_df, tokenizer, max_len)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=100,
        fp16=(device == "cuda"),
        report_to="none",
        seed=config.SEED,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=_compute_metrics,
    )

    t0 = time.time()
    trainer.train()
    fit_secs = round(time.time() - t0, 1)

    eval_metrics = trainer.evaluate()
    print(f"\nEval: {eval_metrics}  (train {fit_secs}s)")

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    metrics = {
        "model": short,
        "base": model_name,
        "sample_size": sample_size,
        "epochs": epochs,
        "accuracy": round(float(eval_metrics.get("eval_accuracy", 0)), 4),
        "f1": round(float(eval_metrics.get("eval_f1", 0)), 4),
        "precision": round(float(eval_metrics.get("eval_precision", 0)), 4),
        "recall": round(float(eval_metrics.get("eval_recall", 0)), 4),
        "train_seconds": fit_secs,
        "hyperparams": {"lr": lr, "batch_size": batch_size, "weight_decay": weight_decay, "max_len": max_len},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model + metrics -> {out_dir}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Fine-tune DistilBERT/BERT for sentiment.")
    ap.add_argument("--model", default="distilbert-base-uncased",
                    help="HF model id (e.g. distilbert-base-uncased, bert-base-uncased)")
    ap.add_argument("--sample", type=int, default=config.DEFAULT_SAMPLE_SIZE,
                    help="balanced training subsample (0 = full dataset)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    args = ap.parse_args()
    sample = None if args.sample == 0 else args.sample
    train(args.model, sample, args.epochs, args.batch_size, args.lr, args.max_len, args.weight_decay)


if __name__ == "__main__":
    main()
