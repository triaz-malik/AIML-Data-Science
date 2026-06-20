"""Phase 2 - Data Cleaning for Vehicle Damage Detection.

Detects and (optionally) quarantines:
  * corrupt / unreadable images
  * exact duplicates (md5)              -> keep one copy
  * cross-split leakage (train ∩ val)   -> always drop the validation copy
  * very blurry images (variance of Laplacian below threshold)

Files are MOVED to ./quarantine/<reason>/ (reversible), never deleted.

Usage:
    python src/clean.py            # report only (dry run)
    python src/clean.py --apply    # actually move flagged files to quarantine/
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import REPORTS_DIR, ROOT, TRAIN_DIR, VAL_DIR
from data import list_images

QUARANTINE = ROOT / "quarantine"
BLUR_THRESHOLD = 25.0   # variance of Laplacian; below this = "very blurry"


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def laplacian_var(path: Path) -> float:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:  # opencv can't read some modes; fall back to PIL
        img = np.asarray(Image.open(path).convert("L"))
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def split_of(path: Path) -> str:
    s = str(path).lower()
    if "validation" in s:
        return "val"
    return "train"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="move flagged files to quarantine/")
    ap.add_argument("--blur-threshold", type=float, default=BLUR_THRESHOLD)
    args = ap.parse_args()

    all_paths = [p for p, _ in list_images(TRAIN_DIR)] + [p for p, _ in list_images(VAL_DIR)]
    print(f"Inspecting {len(all_paths)} images ...")

    corrupt: list[Path] = []
    md5_map: dict[str, list[Path]] = defaultdict(list)
    blur_scores: dict[Path, float] = {}

    for p in all_paths:
        try:
            with Image.open(p) as im:
                im.verify()
            md5_map[md5_of(p)].append(p)
            blur_scores[p] = laplacian_var(p)
        except Exception:  # noqa: BLE001
            corrupt.append(p)

    # ---- decide what to quarantine --------------------------------------
    to_quarantine: dict[Path, str] = {}      # path -> reason

    for p in corrupt:
        to_quarantine[p] = "corrupt"

    leakage_pairs = []
    dup_within = []
    for h, paths in md5_map.items():
        if len(paths) < 2:
            continue
        splits = {split_of(p) for p in paths}
        if "train" in splits and "val" in splits:
            # leakage: drop every validation copy, keep a training copy
            for p in paths:
                if split_of(p) == "val":
                    to_quarantine[p] = "leakage_val"
                    leakage_pairs.append(p)
        else:
            # plain duplicate within one split: keep first, drop the rest
            for p in sorted(paths)[1:]:
                to_quarantine.setdefault(p, "duplicate")
                dup_within.append(p)

    very_blurry = [p for p, v in blur_scores.items() if v < args.blur_threshold]
    for p in very_blurry:
        to_quarantine.setdefault(p, "blurry")

    # ---- act / report ---------------------------------------------------
    moved = 0
    if args.apply:
        for p, reason in to_quarantine.items():
            dest_dir = QUARANTINE / reason / split_of(p) / p.parent.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(p), str(dest_dir / p.name))
                moved += 1
            except Exception as e:  # noqa: BLE001
                print(f"  ! could not move {p}: {e}")

    blur_arr = np.array(list(blur_scores.values()))
    lines = ["# Phase 2 — Data Cleaning Report\n"]
    lines.append(f"- Images inspected: **{len(all_paths)}**")
    lines.append(f"- Corrupt / unreadable: **{len(corrupt)}**")
    lines.append(f"- Cross-split leakage (validation copies of training images): **{len(leakage_pairs)}**")
    lines.append(f"- Within-split exact duplicates removed: **{len(dup_within)}**")
    lines.append(f"- Very blurry (Laplacian var < {args.blur_threshold}): **{len(very_blurry)}**")
    lines.append(f"- **Total files flagged: {len(to_quarantine)}**")
    lines.append(f"- Mode: {'APPLIED (moved to quarantine/)' if args.apply else 'DRY RUN (no files moved)'}"
                 + (f' — moved {moved}' if args.apply else ''))
    lines.append("")
    lines.append(f"Blur (Laplacian variance) stats: min={blur_arr.min():.1f}, "
                 f"p5={np.percentile(blur_arr,5):.1f}, median={np.median(blur_arr):.1f}, "
                 f"max={blur_arr.max():.1f}")
    lines.append("\n## Why this matters\n")
    lines.append("Cross-split leakage is the most important issue: identical images appearing in both "
                 "`training/` and `validation/` would make the model appear more accurate than it really "
                 "is. We always keep the training copy and quarantine the validation copy so evaluation "
                 "stays honest.")
    if leakage_pairs:
        lines.append("\n### Leaked validation files (quarantined)")
        for p in leakage_pairs[:30]:
            lines.append(f"  - `{p}`")
    (REPORTS_DIR / "cleaning_report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[1:8]))
    print(f"\nReport -> {REPORTS_DIR / 'cleaning_report.md'}")
    if not args.apply:
        print("Dry run only. Re-run with --apply to move flagged files to quarantine/.")


if __name__ == "__main__":
    main()
