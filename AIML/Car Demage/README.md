# 🚗 Vehicle Damage Assessment

Binary computer-vision system that classifies a car photo as **damaged** or **whole**,
explains its decision with Grad-CAM, and ships as a Streamlit app. Built end-to-end:
EDA → cleaning → augmentation → three models → tuning → evaluation → explainability →
deployment.

> **Business problem.** Insurers inspect vehicle damage manually — slow (days per claim),
> error-prone, and exposed to fraud. An automated first-pass classifier turns a
> multi-day review into a few seconds, flags likely-damaged vehicles for adjusters, and
> provides a visual audit trail (Grad-CAM) for every decision.

## Results (held-out validation set, 456 images, after de-leaking)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|---------:|----------:|-------:|---:|--------:|
| Baseline CNN (from scratch) | 0.871 | 0.860 | 0.886 | 0.873 | 0.934 |
| **ResNet50** (transfer) | **0.961** | 0.949 | 0.974 | **0.961** | **0.992** |
| EfficientNet-B0 (transfer) | 0.943 | 0.939 | 0.947 | 0.943 | 0.984 |

**Best model: ResNet50** (F1 0.961, ROC-AUC 0.992). Positive class = *damage*.
Trained on an RTX 5080 (CUDA 12.8), mixed-precision, ~4 s/epoch.

![ROC curves](reports/figures/roc_curves.png)

## Why the numbers are trustworthy

EDA found **exact-duplicate images that straddled the train and validation splits** —
classic data leakage that silently inflates validation scores. The cleaning step
quarantines the validation copies (keeping the training copy) so the model is never
tested on an image it trained on. See [`reports/cleaning_report.md`](reports/cleaning_report.md).

## Project structure

```
Vehicle-Damage-Assessment/
├── training/ validation/        # dataset (00-damage, 01-whole) — not in git
├── quarantine/                  # files removed by cleaning (reversible)
├── src/
│   ├── config.py                # paths, classes, hyper-params, device
│   ├── data.py                  # dataset, augmentation transforms, loaders
│   ├── eda.py                   # Phase 1  — EDA + plots + report
│   ├── clean.py                 # Phase 2  — corrupt/duplicate/leakage/blur cleaning
│   ├── models.py                # Phases 4-6 — BaselineCNN, ResNet50, EfficientNetB0
│   ├── engine.py                # shared train/eval loop (AMP, early stopping)
│   ├── train.py                 # Phases 4-6 — training CLI
│   ├── tune.py                  # Phase 7  — Optuna hyper-parameter search
│   ├── evaluate.py              # Phase 8  — metrics, confusion matrix, ROC/PR
│   ├── gradcam.py               # Phase 9  — Grad-CAM explainability
│   ├── predict.py               # single-image inference helper
│   └── future_work.py           # Phases 10-11 — severity & repair-cost scaffolds
├── app/app.py                   # Streamlit deployment
├── models/                      # saved .pth weights
├── reports/                     # generated reports, metrics, figures/
├── gradcam/                     # Grad-CAM outputs
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
# torch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Dataset layout (already present):

```
training/00-damage  training/01-whole
validation/00-damage  validation/01-whole
```

## Reproduce the whole pipeline

```bash
python src/eda.py                              # Phase 1  -> reports/eda_report.md + figures
python src/clean.py --apply                    # Phase 2  -> quarantines leaked/dup/blurry files
python src/train.py --model all --epochs 15    # Phases 4-6 -> models/*.pth + history plots
python src/tune.py  --model efficientnet_b0 --trials 12 --epochs 4   # Phase 7
python src/evaluate.py --model all             # Phase 8  -> reports/model_comparison.md
python src/gradcam.py --model resnet50 --n 8   # Phase 9  -> gradcam/gradcam_resnet50.png
streamlit run app/app.py                       # Deployment UI
```

Single-image prediction:

```bash
python src/predict.py --model resnet50 --image path/to/car.jpg
```

## Phase-by-phase

| Phase | What | Output |
|------|------|--------|
| 1 — EDA | counts, class balance, resolution, RGB, sample grid, duplicate/corrupt scan | `reports/eda_report.md`, `reports/figures/01-05_*.png` |
| 2 — Cleaning | remove corrupt, duplicates, **cross-split leakage**, very-blurry | `reports/cleaning_report.md`, `quarantine/` |
| 3 — Augmentation | resize-crop, flip, rotation, color jitter (day/night/angle robustness) | `src/data.py` |
| 4 — Baseline CNN | Conv→Pool×4 + dense head, trained from scratch | `models/cnn.pth` |
| 5 — ResNet50 | ImageNet transfer learning | `models/resnet50.pth` |
| 6 — EfficientNet-B0 | smaller/faster transfer model | `models/efficientnet_b0.pth` |
| 7 — Tuning | Optuna TPE over lr / batch / dropout / weight-decay | `reports/optuna_*.json` |
| 8 — Evaluation | accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC/PR | `reports/model_comparison.md` |
| 9 — Explainable AI | Grad-CAM heatmaps (where the model looks) | `gradcam/` |
| 10 — Severity *(future)* | No/Minor/Moderate/Severe head | `src/future_work.py` (needs labels) |
| 11 — Repair cost *(future)* | regression on damage features | `src/future_work.py` (needs labels) |
| 12 — Business impact | manual multi-day review → seconds; fraud & cost reduction | this README |

### A note on Phases 10–11

This dataset only has **binary** labels (damage vs whole). Severity grades and
repair-cost figures don't exist in it, so those models can't be honestly trained
here. [`src/future_work.py`](src/future_work.py) provides the model heads and a
transparent heuristic cost baseline, ready to train the moment labelled data is added.

## Business impact (Phase 12)

| | Before | After (this system) |
|--|--------|---------------------|
| Time per first-pass review | hours–days | seconds |
| Consistency | varies by adjuster | deterministic + auditable |
| Fraud signal | manual | confidence score + Grad-CAM evidence |
| Cost | manual labour | one GPU inference |

## Tech stack

PyTorch · torchvision · scikit-learn · Optuna · pytorch-grad-cam · OpenCV · Streamlit
