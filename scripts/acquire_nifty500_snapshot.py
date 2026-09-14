from __future__ import annotations

import argparse
import hashlib
from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

NIFTY500_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
EXPECTED_COUNT = 500


def fetch_snapshot() -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0 (Stock-Guru free public evidence acquisition)",
        "Accept": "text/csv,*/*",
        "Referer": "https://www.niftyindices.com/",
    }
    response = requests.get(NIFTY500_URL, headers=headers, timeout=60)
    response.raise_for_status()
    frame = pd.read_csv(BytesIO(response.content))
    symbol_col = next((c for c in frame.columns if c.strip().lower() in {"symbol", "ticker"}), None)
    if symbol_col is None:
        raise ValueError("NIFTY 500 CSV has no Symbol/Ticker column")
    symbols = frame[symbol_col].astype(str).str.strip().str.upper()
    if len(frame) != EXPECTED_COUNT:
        raise ValueError(f"Expected {EXPECTED_COUNT} NIFTY 500 rows, got {len(frame)}")
    if symbols.eq("").any() or symbols.duplicated().any():
        raise ValueError("NIFTY 500 snapshot contains blank or duplicate symbols")
    out = frame.copy()
    out.insert(0, "as_of", date.today().isoformat())
    out["symbol"] = symbols
    return out


def save_snapshot(output: Path, manifest: Path) -> None:
    frame = fetch_snapshot()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest.write_text(
        "dataset,source_name,source_url,retrieved_at,license_or_terms,source_sha256,file_sha256\\n"
        f"nifty500_membership,NSE Indices,{NIFTY500_URL},{date.today().isoformat()},public index constituent download,{digest},{digest}\\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire the current public NIFTY 500 constituent snapshot")
    parser.add_argument("--output", default="data/nifty500_universe.csv")
    parser.add_argument("--manifest", default="data/nifty500_universe_manifest.csv")
    args = parser.parse_args()
    save_snapshot(Path(args.output), Path(args.manifest))
    print(f"Wrote {args.output} and {args.manifest}")


if __name__ == "__main__":
    main()
