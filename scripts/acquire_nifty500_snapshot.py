from __future__ import annotations

import argparse
import hashlib
from datetime import date, datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

NIFTY500_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
EXPECTED_COUNT = 500


def _validate_date(value: str) -> str:
    date.fromisoformat(value)
    return value


def fetch_snapshot() -> tuple[pd.DataFrame, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Stock-Guru free public evidence acquisition)",
        "Accept": "text/csv,*/*",
        "Referer": "https://www.niftyindices.com/",
    }
    response = requests.get(NIFTY500_URL, headers=headers, timeout=60)
    response.raise_for_status()
    source_sha256 = hashlib.sha256(response.content).hexdigest()
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
    out["symbol"] = symbols
    return out, source_sha256


def save_snapshot(output: Path, manifest: Path, *, as_of: str) -> None:
    as_of = _validate_date(as_of)
    frame, source_sha256 = fetch_snapshot()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    frame.insert(0, "as_of", as_of)
    frame.to_csv(output, index=False)
    file_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest.write_text(
        "dataset,source_name,source_url,as_of,retrieved_at,license_or_terms,source_sha256,file_sha256\n"
        f"nifty500_membership,NSE Indices,{NIFTY500_URL},{as_of},{retrieved_at},public index constituent download,{source_sha256},{file_sha256}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire a public NIFTY 500 constituent snapshot")
    parser.add_argument("--output", default="data/nifty500_universe.csv")
    parser.add_argument("--manifest", default="data/nifty500_universe_manifest.csv")
    parser.add_argument("--as-of", required=True, help="Evidence date represented by the downloaded constituent snapshot (YYYY-MM-DD)")
    args = parser.parse_args()
    save_snapshot(Path(args.output), Path(args.manifest), as_of=args.as_of)
    print(f"Wrote {args.output} and {args.manifest}")


if __name__ == "__main__":
    main()
