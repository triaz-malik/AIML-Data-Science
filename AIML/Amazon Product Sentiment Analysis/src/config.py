"""Central configuration: paths and project-wide constants.

Keeping every path in one place means notebooks and scripts agree on where
the data, cached frames, figures, and models live.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
# Project root = parent of the `src/` directory holding this file.
ROOT = Path(__file__).resolve().parents[1]

# Raw fastText archives. They currently sit at the project root (as shipped),
# but we also look in data/raw so you can move them there if you prefer.
_RAW_CANDIDATES_TRAIN = [ROOT / "train.ft.txt.bz2", ROOT / "data" / "raw" / "train.ft.txt.bz2"]
_RAW_CANDIDATES_TEST = [ROOT / "test.ft.txt.bz2", ROOT / "data" / "raw" / "test.ft.txt.bz2"]

PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"
MODELS_DIR = ROOT / "models"

for _d in (PROCESSED_DIR, FIGURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _first_existing(candidates: list[Path], name: str) -> Path:
    for c in candidates:
        if c.exists():
            return c
    # Fall back to the first candidate so error messages point somewhere sane.
    raise FileNotFoundError(
        f"Could not find the {name} archive. Looked in:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def train_archive() -> Path:
    return _first_existing(_RAW_CANDIDATES_TRAIN, "train")


def test_archive() -> Path:
    return _first_existing(_RAW_CANDIDATES_TEST, "test")


# --- Labels ----------------------------------------------------------------
# The polarity dataset encodes __label__1 = negative, __label__2 = positive.
# We map to integers 0/1 for modeling.
LABEL_NAMES = {0: "negative", 1: "positive"}
NEG, POS = 0, 1

# --- Reproducibility -------------------------------------------------------
SEED = 42

# --- Defaults --------------------------------------------------------------
# A balanced subsample keeps EDA snappy and transformer training tractable on
# a single GPU. Raise/lower or set to None for the full ~3.6M rows.
DEFAULT_SAMPLE_SIZE = 100_000
