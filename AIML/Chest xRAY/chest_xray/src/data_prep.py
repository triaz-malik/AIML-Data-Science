"""Data cleaning + non-destructive split manifests.

What this does (and *why*):
  * Scans train/ val/ test/ for all images.
  * Verifies each image opens and is non-empty -> flags CORRUPT / EMPTY files.
  * Detects exact-duplicate images via content hash (md5 of the file bytes).
  * The official `val/` split has only 16 images -> far too small to choose a
    model on. We MERGE train+val and re-split into a fresh, stratified
    train/val (VAL_FRACTION). The official `test/` set is left UNTOUCHED so
    results stay comparable to published benchmarks.
  * Nothing is moved or deleted on disk: the split is written as CSV manifests
    (path,label,label_idx,split). All downstream code reads these manifests.

Run:
    python -m src.data_prep
Outputs:
    outputs/manifests/{train,val,test}.csv
    outputs/manifests/data_quality_report.csv
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

from . import config as C
from .utils import list_images, set_seed


def _hash_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_split(split_dir: Path, split_name: str) -> tuple[list[dict], list[dict]]:
    """Return (good_rows, problem_rows) for one split directory."""
    good, problems = [], []
    for cls in C.CLASSES:
        for path in list_images(split_dir / cls):
            size = path.stat().st_size
            if size == 0:
                problems.append(dict(path=str(path), split=split_name,
                                     label=cls, issue="EMPTY"))
                continue
            try:
                with Image.open(path) as im:
                    im.verify()                      # detects truncated/corrupt
                with Image.open(path) as im:
                    w, h = im.size                    # re-open: verify() consumes
            except Exception as e:                    # noqa: BLE001
                problems.append(dict(path=str(path), split=split_name,
                                     label=cls, issue=f"CORRUPT:{type(e).__name__}"))
                continue
            good.append(dict(path=str(path), split=split_name, label=cls,
                             label_idx=C.CLASS_TO_IDX[cls], width=w, height=h,
                             bytes=size))
    return good, problems


def find_duplicates(rows: list[dict]) -> list[dict]:
    """Flag files whose byte-content hash collides (exact duplicates)."""
    by_hash: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_hash[_hash_file(Path(r["path"]))].append(r["path"])
    dups = []
    for hsh, paths in by_hash.items():
        if len(paths) > 1:
            # keep the first, mark the rest as duplicates
            for p in paths[1:]:
                dups.append(dict(path=p, issue="DUPLICATE", kept=paths[0]))
    return dups


def build_manifests() -> None:
    set_seed(C.SEED)

    all_good, all_problems = [], []
    for split_dir, name in [(C.TRAIN_DIR, "train"), (C.VAL_DIR, "val"),
                            (C.TEST_DIR, "test")]:
        good, problems = scan_split(split_dir, name)
        all_good += good
        all_problems += problems
        print(f"  scanned {name:5s}: {len(good):5d} ok, {len(problems)} problem(s)")

    df = pd.DataFrame(all_good)

    # ---- duplicate detection (within train+val pool, the part we re-split) ---
    pool_mask = df["split"].isin(["train", "val"])
    dups = find_duplicates(df[pool_mask].to_dict("records"))
    dup_paths = {d["path"] for d in dups}
    if dup_paths:
        print(f"  removing {len(dup_paths)} duplicate file(s) from train/val pool")
    df = df[~df["path"].isin(dup_paths)].reset_index(drop=True)

    # ---- stratified re-split of (train + val) -------------------------------
    from sklearn.model_selection import train_test_split

    pool = df[df["split"].isin(["train", "val"])].copy()
    test = df[df["split"] == "test"].copy()

    tr_idx, val_idx = train_test_split(
        pool.index, test_size=C.VAL_FRACTION,
        stratify=pool["label_idx"], random_state=C.SEED,
    )
    pool.loc[tr_idx, "split"] = "train"
    pool.loc[val_idx, "split"] = "val"
    test["split"] = "test"

    final = pd.concat([pool, test], ignore_index=True)

    # ---- write manifests -----------------------------------------------------
    for name in ("train", "val", "test"):
        sub = final[final["split"] == name].reset_index(drop=True)
        out = C.MANIFEST_DIR / f"{name}.csv"
        sub.to_csv(out, index=False)
        dist = sub["label"].value_counts().to_dict()
        print(f"  wrote {out.name:10s} n={len(sub):5d}  {dist}")

    # ---- data quality report -------------------------------------------------
    problems_df = pd.DataFrame(all_problems +
                               [dict(path=d["path"], issue=d["issue"]) for d in dups])
    qpath = C.MANIFEST_DIR / "data_quality_report.csv"
    problems_df.to_csv(qpath, index=False)
    print(f"  data quality issues: {len(problems_df)} -> {qpath.name}")


if __name__ == "__main__":
    print("Building data manifests (cleaning + stratified re-split)...")
    build_manifests()
    print("Done.")
