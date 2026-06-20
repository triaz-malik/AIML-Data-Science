# Phase 1 — EDA Report: Vehicle Damage Detection

## Dataset overview

- **Total readable images:** 2300
- **Classes:** damage, whole (binary)
- **Corrupt / unreadable images:** 0
- **Image modes:** {'RGB': 2299, 'L': 1}
- **Resolution:** width 112–4912 (median 276), height 74–3264 (median 194)
- **Median aspect ratio:** 1.340

### Counts by split & class

| split | damage | whole |
|---|---|---|
| train | 920 | 920 |
| val | 230 | 230 |

## Class balance (Phase 5)

- Training set is **50.0% damage / 50.0% whole**.
- The dataset is **balanced** → no class weights / focal loss needed. Standard `CrossEntropyLoss` is appropriate.

## Data quality (Phase 2 inputs)

- **Exact duplicate groups (md5):** 13
- **Near-duplicate pairs (aHash, hamming<=5):** 1739

### Sample exact-duplicate groups
  - 2 files: `C:\Working\AI ML Projetcs\Car Demage\training\01-whole\0849.JPEG`, `C:\Working\AI ML Projetcs\Car Demage\validation\01-whole\0203.JPEG`
  - 2 files: `C:\Working\AI ML Projetcs\Car Demage\training\00-damage\0007.JPEG`, `C:\Working\AI ML Projetcs\Car Demage\training\00-damage\0493.jpeg`
  - 2 files: `C:\Working\AI ML Projetcs\Car Demage\training\00-damage\0012.JPEG`, `C:\Working\AI ML Projetcs\Car Demage\training\00-damage\0700.jpeg`
  - 2 files: `C:\Working\AI ML Projetcs\Car Demage\training\00-damage\0036.JPEG`, `C:\Working\AI ML Projetcs\Car Demage\training\00-damage\0639.JPEG`
  - 2 files: `C:\Working\AI ML Projetcs\Car Demage\training\00-damage\0043.JPEG`, `C:\Working\AI ML Projetcs\Car Demage\training\00-damage\0355.JPEG`
  - 2 files: `C:\Working\AI ML Projetcs\Car Demage\training\00-damage\0050.JPEG`, `C:\Working\AI ML Projetcs\Car Demage\training\00-damage\0877.JPEG`
  - 2 files: `C:\Working\AI ML Projetcs\Car Demage\training\00-damage\0134.JPEG`, `C:\Working\AI ML Projetcs\Car Demage\validation\00-damage\0205.JPEG`
  - 2 files: `C:\Working\AI ML Projetcs\Car Demage\training\00-damage\0200.JPEG`, `C:\Working\AI ML Projetcs\Car Demage\training\00-damage\0518.JPEG`
  - 2 files: `C:\Working\AI ML Projetcs\Car Demage\training\00-damage\0240.JPEG`, `C:\Working\AI ML Projetcs\Car Demage\training\00-damage\0717.JPEG`
  - 2 files: `C:\Working\AI ML Projetcs\Car Demage\training\00-damage\0251.JPEG`, `C:\Working\AI ML Projetcs\Car Demage\training\00-damage\0539.JPEG`

## Notes

- **Vehicle-type distribution (Plot 2)** is not available: this dataset has no make/model/body-type labels. Listed as future work.
- Recommended resize: **224×224** (matches ImageNet pretrained ResNet50 / EfficientNetB0).

## Figures

![01_class_distribution.png](figures/01_class_distribution.png)
![02_resolution_hist.png](figures/02_resolution_hist.png)
![03_aspect_ratio.png](figures/03_aspect_ratio.png)
![04_sample_images.png](figures/04_sample_images.png)
![05_rgb_means.png](figures/05_rgb_means.png)