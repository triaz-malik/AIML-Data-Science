# Phase 1 — EDA Report

## Dataset at a glance
- **Image files:** 49,984
- **Unique source leaves (GUID):** 26,573  → on average **1.9 (augmented) files per leaf**
- **Classes:** 27  (Apple___Apple_Scab, Apple___Black_Rot, Apple___Cedar_Apple_Rust, Apple___Healthy, Cherry___Healthy, Cherry___Powdery_Mildew, Corn___Common_Rust, Corn___Gray_Leaf_Spot, Corn___Healthy, Corn___Northern_Leaf_Blight, Grape___Black_Rot, Grape___Esca, Grape___Healthy, Grape___Leaf_Blight, Peach___Bacterial_Spot, Peach___Healthy, Pepper___Bacterial_Spot, Pepper___Healthy, Potato___Early_Blight, Potato___Healthy, Potato___Late_Blight, Strawberry___Healthy, Strawberry___Leaf_Scorch, Tomato___Bacterial_Spot, Tomato___Early_Blight, Tomato___Healthy, Tomato___Late_Blight)
- **Crops:** Apple (7771), Tomato (7399), Corn (7316), Grape (7222), Potato (5702), Pepper (3901), Strawberry (3598), Peach (3566), Cherry (3509)
- **Dominant resolution:** 256x256

## Class balance

| Class | Files | Unique leaves |
|---|---:|---:|
| Apple___Apple_Scab | 2016 | 630 |
| Apple___Healthy | 2008 | 1466 |
| Pepper___Healthy | 1988 | 1205 |
| Apple___Black_Rot | 1987 | 619 |
| Potato___Late_Blight | 1939 | 937 |
| Potato___Early_Blight | 1939 | 935 |
| Tomato___Healthy | 1926 | 1318 |
| Tomato___Early_Blight | 1920 | 944 |
| Grape___Esca | 1920 | 1279 |
| Pepper___Bacterial_Spot | 1913 | 936 |
| Corn___Northern_Leaf_Blight | 1908 | 918 |
| Corn___Common_Rust | 1907 | 1907 |
| Grape___Black_Rot | 1888 | 1136 |
| Corn___Healthy | 1859 | 1112 |
| Tomato___Late_Blight | 1851 | 1592 |
| Peach___Bacterial_Spot | 1838 | 1838 |
| Cherry___Healthy | 1826 | 825 |
| Potato___Healthy | 1824 | 152 |
| Strawberry___Healthy | 1824 | 456 |
| Strawberry___Leaf_Scorch | 1774 | 1064 |
| Apple___Cedar_Apple_Rust | 1760 | 275 |
| Peach___Healthy | 1728 | 360 |
| Grape___Leaf_Blight | 1722 | 1029 |
| Tomato___Bacterial_Spot | 1702 | 1702 |
| Grape___Healthy | 1692 | 423 |
| Cherry___Powdery_Mildew | 1683 | 1003 |
| Corn___Gray_Leaf_Spot | 1642 | 512 |

- **Healthy vs Diseased:** 16,675 healthy (33%) vs 33,309 diseased (67%).
- **Imbalance ratio (max/min class):** 1.23× (largest: *Apple___Apple_Scab*, smallest: *Corn___Gray_Leaf_Spot*).

## Findings
1. The dataset spans **9 crops** (Apple, Tomato, Corn, Grape, Potato, Pepper, Strawberry, Peach, Cherry) across **27 classes**. The largest crop is **Apple** at 16% of all images — big, but not a 70% monopoly; the crop mix is fairly even.
2. Class balance is **mild at the file level** (1.23×, largest *Apple___Apple_Scab*, smallest *Corn___Gray_Leaf_Spot*), but **much sharper per unique leaf** (12.5×). **Potato___Healthy** has the fewest distinct leaves — it is heavily augmented and the most under-represented class in real diversity. Inverse-frequency class weights are applied in training.
3. **Healthy vs Diseased:** 33% / 67% — diseased-leaning, expected for a disease dataset.
4. Images are uniformly **256x256** — resizing to 224×224 for pretrained backbones is aspect-lossless.
5. **Critical:** ~1.9 files per source leaf are pre-augmented rotations/flips. Splitting must be **grouped by GUID** or val/test accuracy is inflated by near-duplicate leakage (handled in `data.make_split`).
6. Colour histograms separate **healthy (greener)** from several diseased classes (brown/yellow lesions shift the red/green balance) — even simple colour features carry signal, though fine-grained look-alikes (e.g. Tomato Early vs Late Blight) will need the CNN.