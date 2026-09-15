"""Augment NSE PIT evidence with the post-2024 Integrated Filing feed.

NSE moved financial results to /api/integrated-filing-results. This script is
an additive fallback: it never invents values and reuses the same XBRL parser
and provenance fields as the legacy acquisition path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests

from acquire_nse_pit_fundamentals import (
    EXPECTED_COUNT,
    HEADERS,
    NIFTY500_URL,
    NSE_HOME,
    parse_xbrl,
    sha256_bytes,
)

INTEGRATED = "https://www.nseindia.com/api/integrated-filing-results"
FIELDS = [
    "symbol", "reported_date", "available_date", "available_timestamp",
    "statement_type", "filing_type", "source", "source_id", "source_url",
    "source_sha256", "version", "metric_name", "metric_value", "currency_units",
    "period_metadata_json", "xbrl_url", "xbrl_sha256", "xbrl_context_ref",
    "xbrl_parse_status",
]
_THREAD = threading.local()


def first(row: dict[str, Any], *names: str) -> Any:
    lowered = {str(k).lower(): v for k, v in row.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def clean_url(value: Any) -> str | None:
    if isinstance(value, dict):
        value = first(value, "url", "href", "link", "xbrl")
    if isinstance(value, list):
        value = next((x for x in value if isinstance(x, str)), None)
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value in {"-", "NA", "N/A"}:
        return None
    return urljoin(NSE_HOME, value)


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({**HEADERS, "X-Requested-With": "XMLHttpRequest"})
    s.get(NSE_HOME, timeout=30)
    return s


def worker_session() -> requests.Session:
    s = getattr(_THREAD, "session", None)
    if s is None:
        s = session()
        _THREAD.session = s
    return s


def get_with_retry(s: requests.Session, url: str, **kwargs: Any) -> requests.Response:
    last: Exception | None = None
    for attempt in range(4):
        try:
            r = s.get(url, **kwargs)
            if r.status_code not in {403, 429, 500, 502, 503, 504}:
                r.raise_for_status()
                return r
            last = requests.HTTPError(f"HTTP {r.status_code} for {url}")
        except requests.RequestException as exc:
            last = exc
        time.sleep(min(2 ** attempt, 8))
    if last:
        raise last
    raise RuntimeError(f"request failed: {url}")


def symbols() -> list[str]:
    s = session()
    r = get_with_retry(s, NIFTY500_URL, headers={**HEADERS, "Referer": "https://www.niftyindices.com/"}, timeout=60)
    df = pd.read_csv(io.BytesIO(r.content))
    col = next(c for c in df.columns if str(c).strip().upper() == "SYMBOL")
    out = sorted({str(x).strip().upper() for x in df[col].dropna() if str(x).strip()})
    if len(out) != EXPECTED_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_COUNT} NIFTY 500 symbols, got {len(out)}")
    return out


def normalize(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("data", "results", "rows", "financialResults"):
            if isinstance(payload.get(key), list):
                return [x for x in payload[key] if isinstance(x, dict)]
    return [x for x in payload if isinstance(x, dict)] if isinstance(payload, list) else []


def fetch_symbol(symbol: str) -> tuple[list[dict[str, Any]], int]:
    s = worker_session()
    params = {"index": "equities", "symbol": symbol, "type": "Integrated Filing- Financials", "page": 1, "size": 100}
    r = get_with_retry(s, INTEGRATED, params=params, timeout=60)
    catalog_hash = sha256_bytes(r.content)
    rows = normalize(r.json())
    records: list[dict[str, Any]] = []
    docs = 0
    for row in rows:
        filing_url = clean_url(first(row, "xbrl", "xbrlFile", "xbrlFileLink", "xbrl_url"))
        if not filing_url:
            continue
        broadcast = first(row, "broadcastDate", "broadcastDateTime", "broadCastDate", "sort_date", "filingDate", "filing_date")
        period = first(row, "period_ended", "periodEnded", "periodEnd", "period")
        base = {
            "symbol": symbol,
            "reported_date": str(period or ""),
            "available_date": str(broadcast or "")[:10],
            "available_timestamp": str(broadcast or ""),
            "statement_type": str(first(row, "consolidated", "consolidatedNonConsolidated") or ""),
            "filing_type": "Integrated Filing- Financials",
            "source": "NSE India Integrated Filing Financials",
            "source_id": f"nse-integrated-financials:{symbol}:{hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()[:16]}",
            "source_url": INTEGRATED,
            "source_sha256": catalog_hash,
            "version": "original",
            "metric_name": "__FILING_CATALOG__",
            "metric_value": 1.0,
            "currency_units": "filing-record",
            "period_metadata_json": json.dumps(row, sort_keys=True, default=str),
        }
        try:
            xr = get_with_retry(s, filing_url, timeout=60)
            xhash = sha256_bytes(xr.content)
            facts = parse_xbrl(xr.content)
        except requests.RequestException:
            continue
        if not facts:
            continue
        docs += 1
        for metric, context, value, unit in facts:
            rec = dict(base)
            rec.update({"metric_name": metric, "metric_value": value, "currency_units": unit or "", "xbrl_url": filing_url, "xbrl_sha256": xhash, "xbrl_context_ref": context, "xbrl_parse_status": "parsed"})
            records.append(rec)
    return records, docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--max-workers", type=int, default=2)
    args = ap.parse_args()

    syms = symbols()
    all_records: list[dict[str, Any]] = []
    stats: dict[str, int] = {}
    errors: dict[str, str] = {}
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        futures = {pool.submit(fetch_symbol, sym): sym for sym in syms}
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            sym = futures[f]
            try:
                rows, docs = f.result()
                all_records.extend(rows)
                stats[sym] = docs
            except Exception as exc:
                stats[sym] = 0
                errors[sym] = f"{type(exc).__name__}: {exc}"
            if i % 25 == 0:
                print(f"integrated processed {i}/{len(syms)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for rec in sorted(all_records, key=lambda x: (x["symbol"], x["reported_date"], x["available_timestamp"], x["metric_name"])):
            w.writerow(rec)

    manifest = {
        "status": "complete" if len(stats) == EXPECTED_COUNT and not errors and sum(v > 0 for v in stats.values()) > 0 else "blocked",
        "source": "NSE India Integrated Filing Financials + linked NSE XBRL/iXBRL",
        "endpoint": INTEGRATED,
        "symbols_expected": EXPECTED_COUNT,
        "symbols_processed": len(stats),
        "records": len(all_records),
        "symbols_with_xbrl_numeric_facts": sum(v > 0 for v in stats.values()),
        "symbols_without_numeric_facts": [s for s, v in sorted(stats.items()) if v == 0],
        "errors": errors,
        "policy": "Missing publication evidence or unavailable filings remain BLOCKED; no synthetic values are permitted.",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
