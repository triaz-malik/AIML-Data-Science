"""Exploratory Data Analysis for the chest X-ray dataset.

Produces (in outputs/figures/):
  01_class_distribution.png   bar chart of counts per split/class
  02_class_balance_pie.png    overall NORMAL vs PNEUMONIA share
  03_sample_normal.png        10 sample NORMAL X-rays
  04_sample_pneumonia.png     10 sample PNEUMONIA X-rays
  05_image_size_dist.png      width/height distributions
  06_pixel_intensity.png      mean pixel-intensity histogram by class
And prints a text summary (also saved to outputs/reports/eda_summary.txt).

Run:
    python -m src.eda
Reads the manifests if present (post-cleaning); otherwise scans folders raw.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, no GUI needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from . import config as C
from .utils import list_images


# --------------------------------------------------------------------------- #
# Data loading: prefer cleaned manifests, fall back to raw folder scan
# --------------------------------------------------------------------------- #
def load_inventory() -> pd.DataFrame:
    manifests = list(C.MANIFEST_DIR.glob("*.csv"))
    have = {p.stem for p in manifests}
    if {"train", "val", "test"} <= have:
        df = pd.concat([pd.read_csv(C.MANIFEST_DIR / f"{s}.csv")
                        for s in ("train", "val", "test")], ignore_index=True)
        print("  using cleaned manifests")
        return df
    print("  manifests not found -> scanning raw folders")
    rows = []
    for split_dir, name in [(C.TRAIN_DIR, "train"), (C.VAL_DIR, "val"),
                            (C.TEST_DIR, "test")]:
        for cls in C.CLASSES:
            for p in list_images(split_dir / cls):
                rows.append(dict(path=str(p), split=name, label=cls))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_class_distribution(df: pd.DataFrame) -> None:
    ct = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    ct = ct.reindex(index=["train", "val", "test"], columns=C.CLASSES)
    ax = ct.plot(kind="bar", figsize=(8, 5),
                 color={"NORMAL": "#4C72B0", "PNEUMONIA": "#C44E52"})
    ax.set_title("Image count by split and class")
    ax.set_xlabel("split")
    ax.set_ylabel("count")
    for cont in ax.containers:
        ax.bar_label(cont, fontsize=8)
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "01_class_distribution.png", dpi=120)
    plt.close()


def plot_class_balance(df: pd.DataFrame) -> None:
    counts = df["label"].value_counts().reindex(C.CLASSES)
    plt.figure(figsize=(5, 5))
    plt.pie(counts, labels=C.CLASSES, autopct="%1.1f%%", startangle=90,
            colors=["#4C72B0", "#C44E52"], explode=(0, 0.03))
    plt.title("Overall class balance (all splits)")
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "02_class_balance_pie.png", dpi=120)
    plt.close()


def plot_samples(df: pd.DataFrame, label: str, fname: str, n: int = 10) -> None:
    paths = df[df["label"] == label]["path"].sample(
        min(n, (df["label"] == label).sum()), random_state=C.SEED).tolist()
    cols = 5
    rows = int(np.ceil(len(paths) / cols))
    plt.figure(figsize=(cols * 2.2, rows * 2.4))
    for i, p in enumerate(paths):
        plt.subplot(rows, cols, i + 1)
        with Image.open(p) as im:
            plt.imshow(im.convert("L"), cmap="gray")
        plt.axis("off")
    plt.suptitle(f"Sample {label} chest X-rays", y=1.0)
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / fname, dpi=120, bbox_inches="tight")
    plt.close()


def plot_image_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Measure width/height (uses manifest cols if present, else reads files)."""
    if {"width", "height"} <= set(df.columns) and df["width"].notna().all():
        sizes = df[["width", "height"]].copy()
    else:
        sample = df.sample(min(800, len(df)), random_state=C.SEED)
        dims = []
        for p in sample["path"]:
            with Image.open(p) as im:
                dims.append(im.size)
        sizes = pd.DataFrame(dims, columns=["width", "height"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(sizes["width"], bins=40, color="#4C72B0")
    axes[0].set_title("Width distribution (px)")
    axes[1].hist(sizes["height"], bins=40, color="#C44E52")
    axes[1].set_title("Height distribution (px)")
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "05_image_size_dist.png", dpi=120)
    plt.close()
    return sizes


def plot_pixel_intensity(df: pd.DataFrame, per_class: int = 150) -> None:
    plt.figure(figsize=(8, 5))
    for label, color in [("NORMAL", "#4C72B0"), ("PNEUMONIA", "#C44E52")]:
        paths = df[df["label"] == label]["path"].sample(
            min(per_class, (df["label"] == label).sum()),
            random_state=C.SEED).tolist()
        means = []
        for p in paths:
            with Image.open(p) as im:
                arr = np.asarray(im.convert("L"), dtype=np.float32)
            means.append(arr.mean())
        plt.hist(means, bins=30, alpha=0.6, label=label, color=color)
    plt.xlabel("mean pixel intensity (0-255)")
    plt.ylabel("image count")
    plt.title("Mean pixel-intensity distribution by class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "06_pixel_intensity.png", dpi=120)
    plt.close()


# --------------------------------------------------------------------------- #
def run() -> None:
    print("Loading inventory...")
    df = load_inventory()
    print(f"  {len(df)} images total")

    print("Plotting...")
    plot_class_distribution(df)
    plot_class_balance(df)
    plot_samples(df, "NORMAL", "03_sample_normal.png")
    plot_samples(df, "PNEUMONIA", "04_sample_pneumonia.png")
    sizes = plot_image_sizes(df)
    plot_pixel_intensity(df)

    # ---- text summary --------------------------------------------------------
    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    overall = df["label"].value_counts()
    pneu_share = overall.get("PNEUMONIA", 0) / len(df) * 100
    imbalance = (overall.max() / overall.min()) if overall.min() else float("inf")

    summary = textwrap.dedent(f"""\
        ===== EDA SUMMARY =====
        Total images: {len(df)}

        Counts by split/class:
        {counts.to_string()}

        Overall class balance:
        {overall.to_string()}
        PNEUMONIA share: {pneu_share:.1f}%   (imbalance ratio ~{imbalance:.2f}:1)

        Image size (px):
          width : min={sizes['width'].min()}  max={sizes['width'].max()}  median={int(sizes['width'].median())}
          height: min={sizes['height'].min()}  max={sizes['height'].max()}  median={int(sizes['height'].median())}

        Business insights:
          * Strong class imbalance -> use class-weighted loss + augmentation;
            report RECALL (not accuracy) as the primary KPI.
          * Variable resolutions -> standardise to {C.IMG_SIZE}x{C.IMG_SIZE}.
          * Pneumonia X-rays show denser/whiter opacities -> visible as a shift
            in the mean-intensity histogram (see 06_pixel_intensity.png).

        Figures written to: {C.FIG_DIR}
    """)
    print(summary)
    out = C.REPORT_DIR / "eda_summary.txt"
    out.write_text(summary, encoding="utf-8")
    print(f"Saved summary -> {out}")


if __name__ == "__main__":
    run()
