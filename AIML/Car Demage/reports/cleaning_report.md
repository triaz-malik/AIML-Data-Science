# Phase 2 — Data Cleaning Report

- Images inspected: **2300**
- Corrupt / unreadable: **0**
- Cross-split leakage (validation copies of training images): **3**
- Within-split exact duplicates removed: **10**
- Very blurry (Laplacian var < 25.0): **1**
- **Total files flagged: 14**
- Mode: APPLIED (moved to quarantine/) — moved 14

Blur (Laplacian variance) stats: min=21.6, p5=351.7, median=1902.5, max=18730.5

## Why this matters

Cross-split leakage is the most important issue: identical images appearing in both `training/` and `validation/` would make the model appear more accurate than it really is. We always keep the training copy and quarantine the validation copy so evaluation stays honest.

### Leaked validation files (quarantined)
  - `C:\Working\AI ML Projetcs\Car Demage\validation\01-whole\0203.JPEG`
  - `C:\Working\AI ML Projetcs\Car Demage\validation\00-damage\0205.JPEG`
  - `C:\Working\AI ML Projetcs\Car Demage\validation\00-damage\0021.JPEG`