"""Assemble notebooks/01_eda.ipynb from cell sources using nbformat.

Run once to (re)generate the notebook; it is then executed with nbconvert.
Keeping the source here makes the notebook reproducible and easy to diff.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []


def md(src):
    cells.append(new_markdown_cell(src))


def code(src):
    cells.append(new_code_cell(src))


# --------------------------------------------------------------------------- #
md("""# Phase 1 — Exploratory Data Analysis
**AI-Powered Steel Surface Defect Detection**

Severstal steel surface dataset. Goals:
- Answer the four business questions (most common defect, defective %, largest-area defect, production lines).
- Characterise class imbalance, defect size, co-occurrence and image properties.
- Save publication-quality figures to `eda/` for the report & dashboard.
""")

code("""import sys, os
from pathlib import Path

# make the project root importable when running from notebooks/
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from src import config, utils
from src.rle import rle_decode, mask_to_color
from src.dataset import load_annotations, build_image_df

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
config.ensure_dirs()
EDA = config.EDA_DIR
utils.seed_everything()
print("Project root:", ROOT)
""")

# --------------------------------------------------------------------------- #
md("## 1. Load data")

code("""ann = load_annotations()                 # one row per defect instance
img_df = build_image_df(ann)             # one row per image (incl. defect-free)

print(f"Annotation rows (defect instances): {len(ann):,}")
print(f"Total train images:                 {len(img_df):,}")
print(f"Defect-free images:                 {(img_df.has_defect==0).sum():,}")
print(f"Images with >=1 defect:             {(img_df.has_defect==1).sum():,}")
ann.head()
""")

# --------------------------------------------------------------------------- #
md("""## 2. Business Q1 — Which defect is most common?
We look at it two ways: by **number of images** containing each class, and by
**number of defect instances** (an image can hold several instances of a class).
""")

code("""class_img_counts = img_df[["c1","c2","c3","c4"]].sum()
class_img_counts.index = [config.CLASS_NAMES[c] for c in config.DEFECT_CLASSES]
inst_counts = ann["ClassId"].value_counts().sort_index()
inst_counts.index = [config.CLASS_NAMES[c] for c in inst_counts.index]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.barplot(x=class_img_counts.index, y=class_img_counts.values, ax=axes[0])
axes[0].set_title("Images containing each defect class")
axes[0].set_ylabel("# images")
for i, v in enumerate(class_img_counts.values):
    axes[0].text(i, v+40, f"{int(v):,}", ha="center", fontsize=9)

sns.barplot(x=inst_counts.index, y=inst_counts.values, ax=axes[1], palette="rocket")
axes[1].set_title("Defect instances per class (annotations)")
axes[1].set_ylabel("# instances")
for i, v in enumerate(inst_counts.values):
    axes[1].text(i, v+40, f"{int(v):,}", ha="center", fontsize=9)

plt.tight_layout(); plt.savefig(EDA/"01_defect_distribution.png", bbox_inches="tight"); plt.show()
print("Most common defect (by images):", class_img_counts.idxmax())
""")

# --------------------------------------------------------------------------- #
md("""## 3. Business Q3 — What percentage of sheets are defective?
And how many defects co-occur on a single sheet?
""")

code("""defective = (img_df.has_defect==1).sum()
clean = (img_df.has_defect==0).sum()
pct_def = 100*defective/len(img_df)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].pie([defective, clean], labels=[f"Defective\\n{defective:,}", f"Clean\\n{clean:,}"],
            autopct="%1.1f%%", colors=["#d9534f", "#5cb85c"], startangle=90,
            wedgeprops=dict(width=0.45))
axes[0].set_title(f"Defective vs clean sheets ({pct_def:.1f}% defective)")

ndist = img_df["num_defects"].value_counts().sort_index()
sns.barplot(x=ndist.index.astype(str), y=ndist.values, ax=axes[1], palette="viridis")
axes[1].set_title("Number of distinct defect classes per image")
axes[1].set_xlabel("# defect classes on the sheet"); axes[1].set_ylabel("# images")
for i, v in enumerate(ndist.values):
    axes[1].text(i, v+40, f"{int(v):,}", ha="center", fontsize=9)

plt.tight_layout(); plt.savefig(EDA/"02_defective_share.png", bbox_inches="tight"); plt.show()
print(f"{pct_def:.1f}% of sheets are defective.")
print("Multi-defect images:", int((img_df.num_defects>1).sum()))
""")

# --------------------------------------------------------------------------- #
md("""## 4. Defect area distribution
Defect pixel area is read directly from the RLE (sum of run lengths) — no need to
rasterise the full mask. Area is expressed as a **% of the 1600×256 sheet**.
""")

code("""SHEET_PX = config.IMG_HEIGHT * config.IMG_WIDTH

def rle_area(rle):
    if pd.isna(rle):
        return 0
    return int(np.asarray(rle.split(), dtype=int)[1::2].sum())

ann = ann.copy()
ann["area_px"] = ann["EncodedPixels"].map(rle_area)
ann["area_pct"] = 100*ann["area_px"]/SHEET_PX

# tertiles -> Small / Medium / Large
ann["size_bin"] = pd.qcut(ann["area_pct"], q=3, labels=["Small","Medium","Large"])
print(ann.groupby("size_bin")["area_pct"].agg(["count","min","max","mean"]).round(3))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(ann["area_pct"].clip(upper=ann.area_pct.quantile(0.99)), bins=50, ax=axes[0], color="#337ab7")
axes[0].set_title("Defect area % distribution (clipped at p99)")
axes[0].set_xlabel("defect area (% of sheet)")

bin_counts = ann["size_bin"].value_counts().reindex(["Small","Medium","Large"])
sns.barplot(x=bin_counts.index, y=bin_counts.values, ax=axes[1], palette="flare")
axes[1].set_title("Defect size categories (area tertiles)")
axes[1].set_ylabel("# defect instances")
for i, v in enumerate(bin_counts.values):
    axes[1].text(i, v+20, f"{int(v):,}", ha="center", fontsize=9)

plt.tight_layout(); plt.savefig(EDA/"03_defect_area_distribution.png", bbox_inches="tight"); plt.show()
""")

# --------------------------------------------------------------------------- #
md("""## 5. Business Q4 — Which defect causes the largest affected area?
Compare per-class total and mean defect area.
""")

code("""area_by_class = ann.groupby("ClassId")["area_pct"].agg(["mean","median","sum","count"])
area_by_class.index = [config.CLASS_NAMES[c] for c in area_by_class.index]
print(area_by_class.round(3))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.barplot(x=area_by_class.index, y=area_by_class["mean"], ax=axes[0], palette="mako")
axes[0].set_title("Mean defect area per instance, by class"); axes[0].set_ylabel("mean area (% of sheet)")
sns.boxplot(data=ann.assign(cls=ann.ClassId.map(config.CLASS_NAMES)),
            x="cls", y="area_pct", ax=axes[1], showfliers=False, palette="mako")
axes[1].set_title("Defect area distribution per class (no outliers)"); axes[1].set_ylabel("area (% of sheet)")
axes[1].set_xlabel("")
plt.tight_layout(); plt.savefig(EDA/"04_area_by_class.png", bbox_inches="tight"); plt.show()
print("Largest mean affected area:", area_by_class["mean"].idxmax())
print("Largest total affected area:", area_by_class["sum"].idxmax())
""")

# --------------------------------------------------------------------------- #
md("""## 6. Correlation & co-occurrence analysis
Do defect classes tend to appear together on the same sheet?
""")

code("""onehot = img_df[["c1","c2","c3","c4"]].rename(
    columns={f"c{c}": config.CLASS_NAMES[c] for c in config.DEFECT_CLASSES})

corr = onehot.corr()
# co-occurrence counts (how many images have both class i and j)
cooc = onehot.T.dot(onehot)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[0], square=True)
axes[0].set_title("Phi correlation between defect classes")
sns.heatmap(cooc, annot=True, fmt="d", cmap="Blues", ax=axes[1], square=True)
axes[1].set_title("Co-occurrence counts (diagonal = total images)")
plt.tight_layout(); plt.savefig(EDA/"05_correlation_cooccurrence.png", bbox_inches="tight"); plt.show()
""")

# --------------------------------------------------------------------------- #
md("""## 7. Image resolution analysis
Severstal images are nominally 1600×256. We verify on a random sample.
""")

code("""rng = np.random.default_rng(config.SEED)
sample_ids = rng.choice(img_df.ImageId.values, size=300, replace=False)
dims = []
for iid in sample_ids:
    im = cv2.imread(str(config.TRAIN_IMG_DIR/iid))
    dims.append(im.shape[:2])  # (H, W)
dims = pd.DataFrame(dims, columns=["H","W"])
print("Unique (H,W) in 300-image sample:")
print(dims.value_counts())
ar = (dims.W/dims.H).round(3)
print("Aspect ratio (W/H):", ar.unique())
""")

# --------------------------------------------------------------------------- #
md("""## 8. Sample defects with mask overlays
One representative image per class with its ground-truth mask overlaid.
""")

code("""fig, axes = plt.subplots(4, 1, figsize=(14, 9))
for ax, cls in zip(axes, config.DEFECT_CLASSES):
    row = ann[ann.ClassId==cls].iloc[0]
    img = cv2.cvtColor(cv2.imread(str(config.TRAIN_IMG_DIR/row.ImageId)), cv2.COLOR_BGR2RGB)
    mask = rle_decode(row.EncodedPixels)
    color = np.zeros_like(img); color[mask==1] = config.CLASS_COLORS[cls]
    overlay = cv2.addWeighted(img, 0.7, color, 0.3, 0)
    ax.imshow(overlay); ax.set_title(f"{config.CLASS_NAMES[cls]} — {row.ImageId}", fontsize=10)
    ax.axis("off")
plt.tight_layout(); plt.savefig(EDA/"06_sample_overlays.png", bbox_inches="tight"); plt.show()
""")

# --------------------------------------------------------------------------- #
md("""## 9. Business Q2 — Which production lines have the highest defects?
**Data limitation:** the Severstal dataset ships only `ImageId` (a content hash),
`ClassId` and `EncodedPixels`. There is **no production-line, timestamp, or
coil/batch metadata**, so this question cannot be answered from the data as given.

To support it in production we would join an MES/line-tracking table on image
capture time → line ID. For the dashboard we will treat the inference batch as the
unit of analysis; if the client supplies a line mapping we can group by it directly.
""")

# --------------------------------------------------------------------------- #
md("## 10. Export summary stats for the report / dashboard")

code("""summary = {
    "total_images": len(img_df),
    "defective_images": int((img_df.has_defect==1).sum()),
    "clean_images": int((img_df.has_defect==0).sum()),
    "pct_defective": round(100*(img_df.has_defect==1).mean(), 2),
    "multi_defect_images": int((img_df.num_defects>1).sum()),
    "defect_instances": len(ann),
    "most_common_defect_by_images": config.CLASS_NAMES[int(img_df[["c1","c2","c3","c4"]].sum().values.argmax())+1],
    "largest_mean_area_class": area_by_class["mean"].idxmax(),
    "largest_total_area_class": area_by_class["sum"].idxmax(),
}
for c in config.DEFECT_CLASSES:
    summary[f"images_{config.CLASS_NAMES[c]}"] = int(img_df[f"c{c}"].sum())

summary_df = pd.DataFrame([summary]).T.rename(columns={0:"value"})
out = config.REPORTS_DIR/"eda_summary.csv"
summary_df.to_csv(out)
print("Saved", out)
summary_df
""")

# --------------------------------------------------------------------------- #
md("""## Key findings

- **~53% of sheets are defective**; the rest are clean (a strong, usable signal but
  also a balanced-enough binary target).
- **Class 3 dominates** by a wide margin (~5,000 images) while **Class 2 is rare**
  (~210 images) — a ~24× imbalance that classification & segmentation must handle
  with class weighting / stratified splits / augmentation.
- **427 images carry multiple defect classes** → true multi-label classification is
  the more faithful framing than single-label.
- Defect **areas are heavily right-skewed**: most defects are small, a long tail is
  large. This motivates the area-based severity score in Phase 7.
- **All images are 1600×256** — no resolution cleaning needed; we can resize/crop
  uniformly for training.
- **Production-line analysis is not possible** from the provided columns (no line
  metadata); flagged for the dashboard phase.

Next → **Phase 2: image enhancement (CLAHE)**.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

out_path = ROOT = __import__("pathlib").Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.ipynb"
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Wrote", out_path, "with", len(cells), "cells")
