# Pneumonia Detection from Chest X-Rays

Deep-learning pipeline that classifies pediatric chest X-rays as **NORMAL** vs
**PNEUMONIA**. Built around the hospital priority: **minimise false negatives**
(missed pneumonia), so **Recall is the primary KPI** (Recall > F1 > AUC > Accuracy).

## Business framing
Radiologists are overloaded and pneumonia diagnosis can be delayed. An automated
screen triages X-rays, flags high-risk patients first, and reduces missed cases.
Grad-CAM heatmaps show *where* the model is looking so clinicians can trust it.

## Dataset (as found on disk)
| split | NORMAL | PNEUMONIA |
|-------|-------:|----------:|
| train | 1,341  | 3,875     |
| val   | 8      | 8         |
| test  | 234    | 390       |

Two issues handled in code:
1. **Tiny validation set (16 imgs)** → `data_prep.py` merges train+val and does a
   **stratified re-split** (15% val). The official **test set is left untouched**.
2. **Class imbalance (~73% pneumonia, 2.69:1)** → class-weighted loss + augmentation;
   recall-first evaluation.

Nothing on disk is moved or deleted — splits live as CSV **manifests** in
`outputs/manifests/`. Duplicate/corrupt/empty files are detected and excluded.

## Setup
```powershell
# 1) GPU build of PyTorch FIRST (RTX 5080 / Blackwell needs the cu128 wheels)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# 2) everything else
pip install -r requirements.txt
# 3) confirm the GPU is visible
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Pipeline (run in order)
```powershell
python -m src.data_prep                       # clean + build split manifests
python -m src.eda                             # EDA figures + summary

# Train the three models (Model 1 -> 2 -> 3 per the project plan)
python -m src.train --model custom_cnn      --epochs 20
python -m src.train --model resnet50        --epochs 25
python -m src.train --model efficientnet_b0 --epochs 25

# Evaluate each on the held-out test set (recall-tuned threshold included)
python -m src.evaluate --model custom_cnn
python -m src.evaluate --model resnet50
python -m src.evaluate --model efficientnet_b0

python -m src.compare                         # comparison table + bar chart
python -m src.gradcam --model efficientnet_b0 --n 8   # explainability
```

## What each module does
| file | role |
|------|------|
| `src/config.py`   | paths, classes, hyperparameter defaults (`TrainConfig`) |
| `src/data_prep.py`| cleaning, duplicate detection, stratified re-split → manifests |
| `src/eda.py`      | class balance, sample montages, size & intensity analysis |
| `src/dataset.py`  | manifest `Dataset`, augmentation, class weights, dataloaders |
| `src/models.py`   | Custom CNN, ResNet50, EfficientNetB0 + freeze/unfreeze, Grad-CAM target |
| `src/metrics.py`  | recall/precision/F1/AUC + recall-tuned threshold selection |
| `src/train.py`    | two-phase transfer learning, class-weighted loss, early stop, checkpointing |
| `src/evaluate.py` | test metrics, confusion/ROC/PR figures, error-analysis CSV |
| `src/compare.py`  | aggregates all models into one comparison table/chart |
| `src/gradcam.py`  | Grad-CAM (original / heatmap / overlay) |

## Outputs
```
outputs/
  manifests/   train.csv val.csv test.csv data_quality_report.csv
  figures/     EDA plots, training history, confusion/ROC/PR, gradcam, comparison
  checkpoints/ <model>_best.pt   (best by val recall)
  reports/     eda_summary.txt, history_*.csv, test_metrics_*.json, misclassified_*.csv
```

## Modeling notes
- **Transfer learning is two-phase**: freeze backbone for `--freeze-epochs`, train the
  head, then unfreeze and fine-tune at a lower LR (`--finetune-lr`).
- **Checkpoint selection** is by **validation recall**, not accuracy.
- **Threshold tuning**: evaluation reports metrics at 0.5 *and* at a recall-maximising
  threshold subject to a precision floor (`--min-precision`, default 0.80) — the
  hospital trade-off knob.
- Grayscale X-rays are replicated to 3 channels to use ImageNet-pretrained backbones.

## Reusing this for the other two projects
The pipeline is dataset-agnostic given a `train/val/test/<class>/` folder layout:
- **Vehicle Damage Assessment** and **Manufacturing Defect Detection** can reuse
  `data_prep → dataset → models → train → evaluate → gradcam` by pointing
  `config.DATA_DIR`/`CLASSES` at the new data. Multi-class works once `CLASSES`
  has >2 entries (swap the recall/threshold logic for macro-averaged metrics).
```
```
