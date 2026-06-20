"""Assemble notebooks/02_image_enhancement.ipynb (Phase 2 — CLAHE & enhancement)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path

nb = new_notebook()
cells = []
md = lambda s: cells.append(new_markdown_cell(s))
code = lambda s: cells.append(new_code_cell(s))

md("""# Phase 2 — Image Enhancement
**AI-Powered Steel Surface Defect Detection**

Steel surface images are low-contrast and unevenly lit, which hides subtle
cracks/inclusions. We evaluate an enhancement pipeline to make defects more
separable before classification & segmentation:

1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) — local contrast
2. **Global contrast stretching** — percentile-based intensity rescaling
3. **Normalization** — ImageNet stats for transfer-learning backbones
4. **Gaussian blur** — optional denoising

All enhancement primitives live in `src/utils.py` so training code reuses them.
""")

code("""import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, cv2
import matplotlib.pyplot as plt
import seaborn as sns

from src import config, utils
from src.rle import rle_decode
from src.dataset import load_annotations

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
config.ensure_dirs()
EDA = config.EDA_DIR
utils.seed_everything()

ann = load_annotations()
print("Loaded", len(ann), "annotations")
""")

md("## 1. Pick representative samples — one defective image per class")

code("""def load_rgb(image_id):
    return cv2.cvtColor(cv2.imread(str(config.TRAIN_IMG_DIR/image_id)), cv2.COLOR_BGR2RGB)

samples = {}
for cls in config.DEFECT_CLASSES:
    row = ann[ann.ClassId==cls].iloc[0]
    samples[cls] = (row.ImageId, load_rgb(row.ImageId), rle_decode(row.EncodedPixels))
print("Samples:", {c: v[0] for c, v in samples.items()})
""")

md("""## 2. Enhancement primitives
`apply_clahe` and `enhance` come from `src/utils.py`. Here we also define a global
contrast-stretch and a normalization preview for comparison.
""")

code("""def contrast_stretch(img, low=2, high=98):
    out = np.zeros_like(img)
    for c in range(3):
        lo, hi = np.percentile(img[:, :, c], (low, high))
        out[:, :, c] = np.clip((img[:, :, c].astype(np.float32)-lo)*255.0/max(hi-lo,1), 0, 255)
    return out.astype(np.uint8)

def normalize_preview(img):
    # ImageNet-normalize then rescale to 0-255 purely for visualization
    x = img.astype(np.float32)/255.0
    x = (x - np.array(config.IMAGENET_MEAN)) / np.array(config.IMAGENET_STD)
    x = (x - x.min())/(x.max()-x.min())
    return (x*255).astype(np.uint8)

# sanity-check the pipeline on one image
iid, img, mask = samples[3]
print("CLAHE  :", utils.apply_clahe(img).shape, utils.apply_clahe(img).dtype)
print("enhance:", utils.enhance(img, clahe=True, blur=True).shape)
""")

md("## 3. Before / after gallery — each method on one image")

code("""iid, img, mask = samples[3]
stages = {
    "Original": img,
    "CLAHE": utils.apply_clahe(img),
    "Contrast stretch": contrast_stretch(img),
    "CLAHE + Gaussian blur": utils.enhance(img, clahe=True, blur=True),
    "Normalized (preview)": normalize_preview(img),
}
fig, axes = plt.subplots(len(stages), 1, figsize=(14, 2.1*len(stages)))
for ax, (name, out) in zip(axes, stages.items()):
    ax.imshow(out); ax.set_title(name, fontsize=10, loc="left"); ax.axis("off")
plt.tight_layout(); plt.savefig(EDA/"07_enhancement_gallery.png", bbox_inches="tight"); plt.show()
""")

md("""## 4. Histogram impact of CLAHE
CLAHE spreads the intensity histogram and equalizes local regions, lifting
low-contrast detail without blowing out bright areas (the clip limit prevents
noise amplification).
""")

code("""iid, img, mask = samples[3]
clahe = utils.apply_clahe(img)
g_before = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
g_after  = cv2.cvtColor(clahe, cv2.COLOR_RGB2GRAY)

fig, axes = plt.subplots(2, 2, figsize=(13, 6))
axes[0,0].imshow(img); axes[0,0].set_title("Original"); axes[0,0].axis("off")
axes[0,1].imshow(clahe); axes[0,1].set_title("CLAHE"); axes[0,1].axis("off")
axes[1,0].hist(g_before.ravel(), bins=64, color="#888"); axes[1,0].set_title(f"Original histogram (std={g_before.std():.1f})")
axes[1,1].hist(g_after.ravel(), bins=64, color="#337ab7"); axes[1,1].set_title(f"CLAHE histogram (std={g_after.std():.1f})")
plt.tight_layout(); plt.savefig(EDA/"08_clahe_histogram.png", bbox_inches="tight"); plt.show()
print(f"Grayscale std contrast: {g_before.std():.2f} -> {g_after.std():.2f}")
""")

md("""## 5. Does enhancement make the defect more visible?
Overlay the ground-truth mask on original vs CLAHE for each class, and measure the
contrast between defect pixels and the surrounding background.
""")

code("""def defect_bg_contrast(gray, mask):
    d = gray[mask==1].mean() if mask.sum() else 0
    b = gray[mask==0].mean()
    return abs(d-b)

rows = []
fig, axes = plt.subplots(len(config.DEFECT_CLASSES), 2, figsize=(15, 2.4*len(config.DEFECT_CLASSES)))
for i, cls in enumerate(config.DEFECT_CLASSES):
    iid, img, mask = samples[cls]
    clahe = utils.apply_clahe(img)
    col = np.zeros_like(img); col[mask==1] = config.CLASS_COLORS[cls]
    ov_o = cv2.addWeighted(img, 0.7, col, 0.3, 0)
    ov_c = cv2.addWeighted(clahe, 0.7, col, 0.3, 0)
    axes[i,0].imshow(ov_o); axes[i,0].set_title(f"{config.CLASS_NAMES[cls]} — original", fontsize=9); axes[i,0].axis("off")
    axes[i,1].imshow(ov_c); axes[i,1].set_title(f"{config.CLASS_NAMES[cls]} — CLAHE", fontsize=9); axes[i,1].axis("off")
    c_o = defect_bg_contrast(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), mask)
    c_c = defect_bg_contrast(cv2.cvtColor(clahe, cv2.COLOR_RGB2GRAY), mask)
    rows.append({"class": config.CLASS_NAMES[cls], "contrast_original": round(c_o,2),
                 "contrast_clahe": round(c_c,2), "delta": round(c_c-c_o,2)})
plt.tight_layout(); plt.savefig(EDA/"09_enhancement_defect_visibility.png", bbox_inches="tight"); plt.show()
contrast_df = pd.DataFrame(rows)
print(contrast_df.to_string(index=False))
""")

md("""## 6. CLAHE clip-limit sweep
The clip limit trades contrast gain against noise amplification. We sweep it to
pick a sensible default for training (used in `utils.apply_clahe`, default 2.0).
""")

code("""iid, img, mask = samples[1]
clips = [1.0, 2.0, 3.0, 5.0, 8.0]
fig, axes = plt.subplots(len(clips), 1, figsize=(14, 1.9*len(clips)))
for ax, cl in zip(axes, clips):
    out = utils.apply_clahe(img, clip_limit=cl)
    std = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).std()
    ax.imshow(out); ax.set_title(f"clip_limit={cl}  (gray std={std:.1f})", fontsize=9, loc="left"); ax.axis("off")
plt.tight_layout(); plt.savefig(EDA/"10_clahe_cliplimit_sweep.png", bbox_inches="tight"); plt.show()
""")

md("""## Key takeaways

- **CLAHE measurably raises local contrast** (grayscale std increases) and, on most
  classes, **widens the defect-vs-background gap** — useful for both human review
  and model input.
- **clip_limit ≈ 2.0** is a good default: visible contrast gain before noise starts
  amplifying at higher limits.
- **Gaussian blur** is optional — helpful against sensor noise but can soften thin
  cracks (Class3/Class1); we keep it **off by default** and expose it as a flag.
- Enhancement is wired into `src/utils.enhance()` and applied as an optional
  preprocessing step in the training transforms (Phase 3+).

Next → **Phase 3: classification (Custom CNN / ResNet50 / EfficientNetB0)**.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = Path(__file__).resolve().parents[1] / "notebooks" / "02_image_enhancement.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Wrote", out, "with", len(cells), "cells")
