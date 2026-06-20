# AI-Powered Steel Surface Defect Detection & Quality Inspection Platform

End-to-end computer-vision platform on the **Severstal Steel Defect Detection**
dataset that combines EDA, classification, segmentation, explainability, a
business decision engine, and MLOps deployment.

> Steel images → validation → EDA → CLAHE enhancement → classification
> (CNN/ResNet/EfficientNet) → segmentation (U-Net) → Grad-CAM → severity &
> quality decision → Power BI dashboard → FastAPI/Docker/Azure deployment.

## Dataset

| Item | Value |
|------|-------|
| Images | 1600 × 256 RGB |
| Train images | 12,568 |
| Test images | 5,506 |
| Defect classes | 4 (Class1–Class4) |
| Annotations | `train.csv` — RLE-encoded, column-major, 1-indexed; one row per defect |

Images **not** present in `train.csv` are defect-free. The classification task
uses 5 labels (`NoDefect`, `Class1`–`Class4`); segmentation predicts a 4-channel
pixel mask.

## Project structure

```
.
├── train_images/  test_images/  train.csv     # data (gitignored)
├── src/                # shared library
│   ├── config.py       # paths, constants, severity thresholds
│   ├── rle.py          # RLE encode/decode, mask building & coloring
│   ├── dataset.py      # build_image_df + PyTorch Datasets (clf & seg)
│   └── utils.py        # seeding, device, CLAHE, Dice/IoU, severity
├── notebooks/          # 01_eda → 06_business_kpi
├── eda/                # Phase 1 figures
├── classification/     # Phase 3-4 model code & checkpoints meta
├── segmentation/       # Phase 5 U-Net
├── gradcam/            # Phase 6 heatmaps
├── reports/            # severity / KPI csv exports
├── dashboard/          # Power BI dataset + .pbix
├── deployment/         # FastAPI app + Dockerfile
├── models/             # trained weights (gitignored)
└── requirements.txt
```

## Phases

| # | Phase | Output |
|---|-------|--------|
| 1 | EDA | defect/area/frequency distributions, correlations |
| 2 | Image enhancement | CLAHE before/after |
| 3 | Classification | Custom CNN, ResNet50, EfficientNetB0 |
| 4 | Hyperparameter tuning | Optuna study, accuracy/loss comparison |
| 5 | Segmentation | U-Net / DeepLabV3, Dice/IoU, mask overlays |
| 6 | Explainability | Grad-CAM heatmaps |
| 7 | Severity score | defect-area % → Minor/Moderate/Critical |
| 8 | Decision engine | Accept / Rework / Reject |
| 9 | Business KPIs | Power BI dataset & dashboard |
| 10 | MLOps | FastAPI + Docker (+ Azure ML notes) |

## Setup

```powershell
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
# Install a CUDA build of torch matching your GPU first, e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Quick start

```python
from src import config, utils
from src.dataset import load_annotations, build_image_df

utils.seed_everything()
ann = load_annotations()
img_df = build_image_df(ann)          # one row per image with labels
print(img_df["label"].value_counts())
```

Run notebooks in order from `notebooks/`. Each phase writes artifacts into its
matching folder.

## Hardware

Developed/tested on an NVIDIA RTX 5080 (CUDA), PyTorch 2.11.
