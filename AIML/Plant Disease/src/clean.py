"""Phase 2 — Data cleaning.

Checks every image for:
  * corruption / unreadable files (PIL verify + reload)
  * exact duplicates (sha1 of decoded pixels, so re-encodes are caught too)
  * missing / zero-byte files

Writes a cleaning report and a ``clean_catalog.csv`` (the catalog minus any
dropped files). Nothing is deleted from disk by default — pass ``--delete`` to
actually remove corrupted/zero-byte files.

Run:  python -m src.clean          (report only)
      python -m src.clean --delete (also remove corrupt/empty files)
"""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

from . import config as C
from .data import scan_dataset


def _pixel_hash(path: str) -> str | None:
    """sha1 over decoded RGB bytes — robust to re-encoding, catches true dups."""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return hashlib.sha1(im.tobytes()).hexdigest()
    except Exception:
        return None


def check(df: pd.DataFrame):
    corrupted, zero_byte, ok = [], [], []
    hashes: dict[str, list[str]] = defaultdict(list)

    for p in df["path"]:
        fp = Path(p)
        if not fp.exists():
            corrupted.append((p, "missing"))
            continue
        if fp.stat().st_size == 0:
            zero_byte.append(p)
            continue
        h = _pixel_hash(p)
        if h is None:
            corrupted.append((p, "unreadable"))
            continue
        hashes[h].append(p)
        ok.append(p)

    # duplicate groups = hash buckets with >1 member
    dup_groups = {h: ps for h, ps in hashes.items() if len(ps) > 1}
    # keep the first of each group, mark the rest as removable duplicates
    dup_drop = [p for ps in dup_groups.values() for p in ps[1:]]
    return corrupted, zero_byte, dup_groups, dup_drop, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true",
                    help="physically delete corrupted/zero-byte files")
    args = ap.parse_args()

    print("Scanning + hashing (this reads every image)...")
    df = scan_dataset()
    corrupted, zero_byte, dup_groups, dup_drop, ok = check(df)

    n = len(df)
    drop = set(p for p, _ in corrupted) | set(zero_byte) | set(dup_drop)
    clean = df[~df["path"].isin(drop)].reset_index(drop=True)
    clean.to_csv(C.SPLITS_DIR / "clean_catalog.csv", index=False)

    lines = [
        "# Phase 2 — Data Cleaning Report",
        "",
        f"- **Scanned:** {n:,} files",
        f"- **Corrupted / unreadable / missing:** {len(corrupted)}",
        f"- **Zero-byte:** {len(zero_byte)}",
        f"- **Exact duplicate groups (identical pixels):** {len(dup_groups)} "
        f"→ {len(dup_drop)} redundant files",
        f"- **Clean files kept:** {len(clean):,} "
        f"({len(clean)/n:.1%} of scanned)",
        "",
    ]
    if corrupted:
        lines += ["## Corrupted / missing", ""]
        lines += [f"- `{Path(p).name}` ({why})" for p, why in corrupted[:50]]
        if len(corrupted) > 50:
            lines.append(f"- ...and {len(corrupted)-50} more")
        lines.append("")
    if dup_groups:
        lines += ["## Sample duplicate groups", ""]
        for h, ps in list(dup_groups.items())[:10]:
            names = ", ".join(Path(p).name for p in ps)
            lines.append(f"- {names}")
        lines.append("")
    if not corrupted and not zero_byte and not dup_groups:
        lines.append("**No issues found — dataset is clean.**")

    deleted = 0
    if args.delete:
        for p, why in corrupted:
            if why != "missing" and Path(p).exists():
                Path(p).unlink(); deleted += 1
        for p in zero_byte:
            if Path(p).exists():
                Path(p).unlink(); deleted += 1
        lines.append(f"\n**Deleted {deleted} corrupted/zero-byte files from disk.**")
    else:
        lines.append("\n*(Run with `--delete` to physically remove corrupt/empty "
                     "files. Duplicates are excluded from `clean_catalog.csv` but "
                     "left on disk.)*")

    path = C.REPORTS_DIR / "CLEANING_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  corrupted={len(corrupted)} zero={len(zero_byte)} "
          f"dup_groups={len(dup_groups)} dup_files={len(dup_drop)}")
    print(f"  wrote {path.relative_to(C.ROOT)}")
    print(f"  wrote {(C.SPLITS_DIR/'clean_catalog.csv').relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
