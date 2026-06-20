"""Phase 1 - Exploratory Data Analysis for Vehicle Damage Detection.

Produces:
  reports/figures/01_class_distribution.png
  reports/figures/02_resolution_hist.png
  reports/figures/03_aspect_ratio.png
  reports/figures/04_sample_images.png
  reports/figures/05_rgb_means.png
  reports/eda_report.md
  reports/image_metadata.csv
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from config import CLASS_TO_IDX, FIGURES_DIR, IDX_TO_CLASS, REPORTS_DIR, TRAIN_DIR, VAL_DIR
from data import list_images

sns_ok = True
try:
    import seaborn as sns

    sns.set_theme(style="whitegrid")
except Exception:  # pragma: no cover
    sns_ok = False


def md5_of(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ahash(img: Image.Image) -> str:
    """8x8 average hash for near-duplicate detection."""
    g = img.convert("L").resize((8, 8), Image.BILINEAR)
    arr = np.asarray(g, dtype=np.float32)
    bits = (arr > arr.mean()).flatten()
    return "".join("1" if b else "0" for b in bits)


def hamming(a: str, b: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def scan(split_name: str, root):
    """Walk a split, collect metadata + flag corrupt files."""
    rows = []
    corrupt = []
    for path, label in list_images(root):
        try:
            with Image.open(path) as im:
                im.verify()                      # detect truncation/corruption
            with Image.open(path) as im:
                w, h = im.size
                mode = im.mode
                rgb = np.asarray(im.convert("RGB").resize((64, 64)))
            rows.append(
                {
                    "split": split_name,
                    "path": str(path),
                    "class": IDX_TO_CLASS[label],
                    "label": label,
                    "width": w,
                    "height": h,
                    "aspect": round(w / h, 3),
                    "mode": mode,
                    "r_mean": float(rgb[..., 0].mean()),
                    "g_mean": float(rgb[..., 1].mean()),
                    "b_mean": float(rgb[..., 2].mean()),
                    "md5": md5_of(path),
                    "ahash": ahash(Image.open(path)),
                }
            )
        except Exception as e:  # noqa: BLE001
            corrupt.append((str(path), repr(e)))
    return rows, corrupt


def find_duplicates(df: pd.DataFrame):
    """Exact (md5) and near (ahash hamming<=5) duplicates."""
    exact = defaultdict(list)
    for _, r in df.iterrows():
        exact[r["md5"]].append(r["path"])
    exact_groups = [paths for paths in exact.values() if len(paths) > 1]

    near = []
    records = df[["path", "ahash"]].to_dict("records")
    seen = set()
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if records[i]["md5"] if False else False:
                continue
            d = hamming(records[i]["ahash"], records[j]["ahash"])
            if d <= 5:
                key = (records[i]["path"], records[j]["path"])
                if key not in seen:
                    near.append((records[i]["path"], records[j]["path"], d))
                    seen.add(key)
    return exact_groups, near


def main() -> None:
    print("Scanning training and validation splits ...")
    train_rows, train_corrupt = scan("train", TRAIN_DIR)
    val_rows, val_corrupt = scan("val", VAL_DIR)
    rows = train_rows + val_rows
    corrupt = train_corrupt + val_corrupt
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "image_metadata.csv", index=False)

    total = len(df)
    print(f"Total readable images: {total}  (corrupt: {len(corrupt)})")

    # ----- Plot 1 / 5: class distribution & balance -----------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, split in zip(axes, ["train", "val"]):
        sub = df[df.split == split]
        counts = sub["class"].value_counts().reindex(["whole", "damage"]).fillna(0)
        bars = ax.bar(counts.index, counts.values, color=["#4C72B0", "#C44E52"])
        ax.set_title(f"{split} — class distribution (n={len(sub)})")
        ax.set_ylabel("images")
        for b, v in zip(bars, counts.values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{int(v)}", ha="center", va="bottom")
    fig.suptitle("Phase 1 — Damage vs Whole distribution", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_class_distribution.png", dpi=120)
    plt.close(fig)

    # ----- Plot 2 (vehicle type): not available in this dataset -----------
    # The Car Damage Detection dataset has no vehicle-type labels, so this
    # plot is intentionally omitted (documented in the report).

    # ----- Plot 3: resolution histograms ----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df["width"], bins=40, color="#4C72B0", alpha=0.8)
    axes[0].set_title("Image width distribution")
    axes[0].set_xlabel("width (px)")
    axes[1].hist(df["height"], bins=40, color="#55A868", alpha=0.8)
    axes[1].set_title("Image height distribution")
    axes[1].set_xlabel("height (px)")
    fig.suptitle("Phase 1 — Resolution distribution", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_resolution_hist.png", dpi=120)
    plt.close(fig)

    # ----- Plot 3b: aspect ratio ------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["aspect"], bins=40, color="#8172B3", alpha=0.85)
    ax.axvline(1.0, color="k", ls="--", lw=1, label="square (1:1)")
    ax.set_title("Aspect ratio (width / height)")
    ax.set_xlabel("aspect ratio")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_aspect_ratio.png", dpi=120)
    plt.close(fig)

    # ----- Plot 4: sample images ------------------------------------------
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    for row, cls in enumerate(["damage", "whole"]):
        paths = df[(df["class"] == cls) & (df.split == "train")]["path"].tolist()[:5]
        for col, p in enumerate(paths):
            ax = axes[row, col]
            ax.imshow(Image.open(p).convert("RGB"))
            ax.set_title(cls, color="#C44E52" if cls == "damage" else "#4C72B0")
            ax.axis("off")
    fig.suptitle("Phase 1 — Sample images (top: damage, bottom: whole)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_sample_images.png", dpi=120)
    plt.close(fig)

    # ----- Plot 5: RGB channel means by class -----------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    means = df.groupby("class")[["r_mean", "g_mean", "b_mean"]].mean().reindex(["whole", "damage"])
    means.plot(kind="bar", ax=ax, color=["#C44E52", "#55A868", "#4C72B0"])
    ax.set_title("Mean RGB intensity by class")
    ax.set_ylabel("mean pixel value (0-255)")
    ax.set_xticklabels(means.index, rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_rgb_means.png", dpi=120)
    plt.close(fig)

    # ----- duplicates ------------------------------------------------------
    print("Detecting duplicates ...")
    exact_groups, near = find_duplicates(df)

    # ----- write report ----------------------------------------------------
    counts = df.groupby(["split", "class"]).size().unstack(fill_value=0)
    train_total = len(df[df.split == "train"])
    dmg_pct = 100 * len(df[(df.split == "train") & (df["class"] == "damage")]) / max(train_total, 1)

    lines = []
    lines.append("# Phase 1 — EDA Report: Vehicle Damage Detection\n")
    lines.append("## Dataset overview\n")
    lines.append(f"- **Total readable images:** {total}")
    lines.append(f"- **Classes:** {', '.join(sorted(df['class'].unique()))} (binary)")
    lines.append(f"- **Corrupt / unreadable images:** {len(corrupt)}")
    lines.append(f"- **Image modes:** {dict(Counter(df['mode']))}")
    lines.append(
        f"- **Resolution:** width {df.width.min()}–{df.width.max()} "
        f"(median {int(df.width.median())}), height {df.height.min()}–{df.height.max()} "
        f"(median {int(df.height.median())})"
    )
    lines.append(f"- **Median aspect ratio:** {df.aspect.median():.3f}\n")
    lines.append("### Counts by split & class\n")
    cols = list(counts.columns)
    lines.append("| split | " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for split_name, row in counts.iterrows():
        lines.append(f"| {split_name} | " + " | ".join(str(int(row[c])) for c in cols) + " |")
    lines.append("")
    lines.append("## Class balance (Phase 5)\n")
    lines.append(f"- Training set is **{dmg_pct:.1f}% damage / {100 - dmg_pct:.1f}% whole**.")
    if abs(dmg_pct - 50) < 5:
        lines.append("- The dataset is **balanced** → no class weights / focal loss needed. "
                     "Standard `CrossEntropyLoss` is appropriate.\n")
    else:
        lines.append("- The dataset is **imbalanced** → consider class weights or focal loss.\n")
    lines.append("## Data quality (Phase 2 inputs)\n")
    lines.append(f"- **Exact duplicate groups (md5):** {len(exact_groups)}")
    lines.append(f"- **Near-duplicate pairs (aHash, hamming<=5):** {len(near)}")
    if corrupt:
        lines.append("\n### Corrupt files")
        for p, e in corrupt[:20]:
            lines.append(f"  - `{p}` — {e}")
    if exact_groups:
        lines.append("\n### Sample exact-duplicate groups")
        for g in exact_groups[:10]:
            lines.append(f"  - {len(g)} files: " + ", ".join(f"`{x}`" for x in g[:3]) + (" ..." if len(g) > 3 else ""))
    lines.append("\n## Notes\n")
    lines.append("- **Vehicle-type distribution (Plot 2)** is not available: this dataset has no "
                 "make/model/body-type labels. Listed as future work.")
    lines.append(f"- Recommended resize: **224×224** (matches ImageNet pretrained ResNet50 / EfficientNetB0).")
    lines.append("\n## Figures\n")
    for f in ["01_class_distribution.png", "02_resolution_hist.png", "03_aspect_ratio.png",
              "04_sample_images.png", "05_rgb_means.png"]:
        lines.append(f"![{f}](figures/{f})")

    (REPORTS_DIR / "eda_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"EDA complete. Report -> {REPORTS_DIR / 'eda_report.md'}")
    print(f"Figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
