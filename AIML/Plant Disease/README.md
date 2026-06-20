# Plant Disease Detection — End-to-End (PyTorch)

Detect crop-leaf disease from a single image, **explain** the prediction with
Grad-CAM, and return a **treatment recommendation**. Built on the PlantVillage
dataset (9 crops, 27 classes, ~50k images) with a leakage-safe pipeline.

> **Framework note:** the original plan referenced TensorFlow/`.h5`. This repo
> uses **PyTorch** (a working CUDA GPU was available; TF has no stable Python
> 3.13/Windows build). Artifacts: `best_model.pt` + `label_encoder.pkl`.

## Why this pipeline is honest
Many leaves are *pre-augmented* (rotations/flips share a GUID prefix before
`___`). A naive split leaks near-duplicate copies across train/val/test and
inflates accuracy. We split **grouped by source-leaf GUID**, stratified by class
(`src/data.make_split`).

## Setup
```powershell
pip install -r requirements.txt   # torch GPU build already present in this env
```

## Run the phases
```powershell
python -m src.eda          # Phase 1  EDA plots + report
python -m src.clean        # Phase 2  corrupt/duplicate detection (--delete to remove)
python -m src.train        # Phases 3-6  augment + split + 3 models + comparison
python -m src.tune         # Phase 7  Optuna tuning -> best_model.pt
python -m src.evaluate --ckpt outputs/models/best_model.pt   # Phase 9 confusion/ROC/errors
python -m src.gradcam  --ckpt outputs/models/best_model.pt   # Phase 8 explainability
python -m src.predict  --ckpt outputs/models/best_model.pt   # Phase 10 inference + advice
python -m src.report       # Phases 11-12  business + final reports
```

## Project layout
```
src/
  config.py     paths, hyper-params, seeding
  data.py       scan, GUID-grouped split, augmentation, loaders
  models.py     BaselineCNN, ResNet50, EfficientNetB0
  engine.py     AMP training/eval loop (shared by train + tune)
  eda.py        Phase 1
  clean.py      Phase 2
  train.py      Phases 3-6
  tune.py       Phase 7  (Optuna)
  gradcam.py    Phase 8
  evaluate.py   Phase 9
  recommend.py  Phase 10  (27-class treatment dictionary)
  predict.py    inference on test/ images
  report.py     Phases 11-12
outputs/
  plots/  models/  reports/  splits/
```

## Key outputs
- `outputs/models/best_model.pt`, `label_encoder.pkl`
- `outputs/reports/`: EDA, cleaning, model comparison, error analysis, business, final summary
- `outputs/plots/`: class distribution, healthy-vs-diseased, training curves,
  confusion matrix, ROC, Grad-CAM, predictions grid
