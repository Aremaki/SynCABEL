#!/usr/bin/env python
"""Extract specified UMLS releases then prepare dataset-specific synonym files.

It expects UMLS release zip archives to be located under data/UMLS_raw/.
Default mapping:
  UMLS_2014AB.zip -> QUAERO
  UMLS_2017AA.zip -> MM

Outputs:
  Extracted parquet files -> data/UMLS_processed/<DATASET>/ (codes, semantic, title_syn)
  Prepared synonym parquets -> data/UMLS_processed/<DATASET>/ (all_disambiguated.parquet, fr_disambiguated.parquet)

Run:
  python scripts/1_preprocess_UMLS/run_extract_and_prepare_umls.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

RAW_DIR = Path("data/UMLS_raw")
EXTRACT_OUT_BASE = Path("data/UMLS_processed")
EXTRACT_SCRIPT = Path("scripts/1_preprocess_UMLS/extract_umls_data.py")
PREPARE_SCRIPT = Path("scripts/1_preprocess_UMLS/prepare_umls_data.py")

# You may edit this list to add more releases.
RELEASES = [("UMLS_2014AB.zip", "QUAERO"), ("UMLS_2017AA.zip", "MM")]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def extract(release: tuple[str, str]) -> Path:
    zip_name, dataset_name = release
    zip_path = RAW_DIR / zip_name
    if not zip_path.exists():
        print(f"[WARN] Missing {zip_path}, skipping {dataset_name} ({zip_name}).")
        return None  # type: ignore
    out_dir = EXTRACT_OUT_BASE / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Extracting {zip_path} -> {out_dir}")
    cmd = [
        "python",
        str(EXTRACT_SCRIPT),
        "all",
        "--umls-zip",
        str(zip_path),
        "--out-dir",
        str(out_dir),
    ]
    run(cmd)
    return out_dir


def prepare(release: tuple[str, str], umls_dir: Path) -> None:
    _, dataset_name = release
    print(
        f"[INFO] Preparing dataset {dataset_name} from {umls_dir}"  # noqa: E501
    )
    cmd = [
        "python",
        str(PREPARE_SCRIPT),
        "--dataset",
        dataset_name,
        "--umls-dir",
        str(umls_dir),
    ]
    run(cmd)


def main() -> None:
    if not EXTRACT_SCRIPT.exists():
        raise SystemExit(f"Extraction script not found: {EXTRACT_SCRIPT}")
    if not PREPARE_SCRIPT.exists():
        raise SystemExit(f"Preparation script not found: {PREPARE_SCRIPT}")

    for release in RELEASES:
        out_dir = extract(release)
        if out_dir is None:
            continue
        prepare(release, out_dir)

    print("✅ UMLS extraction + preparation complete.")


if __name__ == "__main__":  # pragma: no cover
    main()
