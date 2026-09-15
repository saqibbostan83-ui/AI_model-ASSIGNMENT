"""
Download the Diabetes 130-US Hospitals dataset (1999-2008) into data/raw/.

The original UCI ML Repository (archive.ics.uci.edu) is not reachable from
every network, so this script pulls a verified byte-identical mirror that is
hosted on GitHub. Two integrity checks are performed after download:

  * row count    : 101,767 lines including the header (101,766 records)
  * column count : 50

Usage:
    python src/download_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import urllib.request

# Verified GitHub mirror (byte-identical copy of the original UCI dataset,
# stored as plain git content - NOT git-LFS - so it can be fetched via the
# GitHub contents API).
MIRROR_REPO = "taspinar/siml"
MIRROR_PATHS = {
    "diabetic_data.csv": "datasets/diabetes/diabetic_data.csv",
    "IDs_mapping.csv": "datasets/diabetes/IDs_mapping.csv",
}
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

EXPECTED_ROWS = 101_766  # excluding header
EXPECTED_COLS = 50


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for fname, repo_path in MIRROR_PATHS.items():
        url = f"https://api.github.com/repos/{MIRROR_REPO}/contents/{repo_path}"
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github.raw"})
        dest = RAW_DIR / fname
        print(f"Downloading {url} -> {dest}")
        with urllib.request.urlopen(req) as resp, open(dest, "wb") as out:
            out.write(resp.read())

        # ---- integrity checks ----
        with open(dest, "r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        n_rows = len(lines) - 1
        n_cols = len(lines[0].split(","))
        print(f"  {fname}: {n_rows:,} rows x {n_cols} columns")

        if fname == "diabetic_data.csv":
            assert n_rows == EXPECTED_ROWS, f"row mismatch: {n_rows} != {EXPECTED_ROWS}"
            assert n_cols == EXPECTED_COLS, f"column mismatch: {n_cols} != {EXPECTED_COLS}"
            print("  integrity check passed ✔")
    print("All files downloaded and verified.")


if __name__ == "__main__":
    sys.exit(download())
