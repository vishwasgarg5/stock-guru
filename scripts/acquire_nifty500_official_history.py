"""Acquire the official NSE Indices NIFTY 500 inclusion/exclusion archive.

The archive is an official primary NSE Indices workbook.  It is used as
historical membership evidence; no secondary reconstruction is promoted to
certified evidence by this script.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import requests

OFFICIAL_URL = "https://archives.nseindia.com/content/indices/IndexInclExcl.xls"


def acquire(output_csv: Path, manifest_csv: Path) -> None:
    response = requests.get(OFFICIAL_URL, timeout=60)
    response.raise_for_status()
    raw = response.content
    source_sha256 = hashlib.sha256(raw).hexdigest()

    workbook = output_csv.with_suffix(".xls")
    workbook.write_bytes(raw)
    sheets = pd.ExcelFile(workbook, engine="xlrd").sheet_names
    matches = [name for name in sheets if "nifty 500" in str(name).strip().lower()]
    if not matches:
        raise ValueError(f"Official workbook has no Nifty 500 sheet; sheets={sheets!r}")

    frame = pd.read_excel(workbook, sheet_name=matches[0], engine="xlrd")
    frame = frame.dropna(how="all")
    if frame.empty:
        raise ValueError("Official Nifty 500 sheet is empty")

    frame.to_csv(output_csv, index=False)
    file_sha256 = hashlib.sha256(output_csv.read_bytes()).hexdigest()
    manifest_csv.write_text(
        "dataset,source_url,sheet,source_sha256,file_sha256\n"
        f"nifty500_membership_history,{OFFICIAL_URL},{matches[0]},{source_sha256},{file_sha256}\n",
        encoding="utf-8",
    )
    workbook.unlink()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.output, args.manifest)
