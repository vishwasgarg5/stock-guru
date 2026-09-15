"""Acquire real NSE financial-result/XBRL evidence for the current NIFTY 500 universe.

This is deliberately fail-closed: every observation keeps the NSE catalog
broadcast timestamp, filing URL, and SHA-256 hashes. No values are invented.
The official Nifty 500 constituent download currently contains 501 unique
constituent rows; that upstream count is preserved rather than silently
truncating one constituent.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from lxml import etree, html

NIFTY500_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
NSE_HOME = "https://www.nseindia.com/"
CATALOG = "https://www.nseindia.com/api/corporates-financial-results?index=equities&symbol={symbol}&period=Quarterly"
EXPECTED_COUNT = 501
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-financial-results",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    s.get(NSE_HOME, timeout=30)
    return s


def get_symbols(s: requests.Session) -> list[str]:
    r = s.get(NIFTY500_URL, headers={**HEADERS, "Referer": "https://www.niftyindices.com/"}, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content))
    col = next((c for c in df.columns if str(c).strip().upper() == "SYMBOL"), None)
    if not col:
        raise RuntimeError(f"NIFTY 500 CSV has no SYMBOL column: {list(df.columns)}")
    symbols = sorted({str(x).strip().upper() for x in df[col].dropna() if str(x).strip()})
    if len(symbols) != EXPECTED_COUNT:
        raise RuntimeError(f"Expected exactly {EXPECTED_COUNT} unique NIFTY 500 symbols, got {len(symbols)}")
    return symbols


def normalize_catalog(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("data", "results", "financialResults", "rows"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        return []
    return [x for x in payload if isinstance(x, dict)]


def first(row: dict[str, Any], *names: str) -> Any:
    lowered = {str(k).lower(): v for k, v in row.items()}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def clean_url(value: Any) -> str | None:
    if not value or not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value in {"-", "NA", "N/A"}:
        return None
    return urljoin("https://www.nseindia.com", value)


def parse_numeric(text: str) -> float | None:
    t = text.replace("\xa0", " ").strip().replace(",", "")
    if not t:
        return None
    negative = t.startswith("(") and t.endswith(")")
    if negative:
        t = t[1:-1].strip()
    t = t.replace("−", "-")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", t):
        return None
    try:
        value = float(t)
        if not math.isfinite(value):
            return None
        return -value if negative else value
    except ValueError:
        return None


def _ixbrl_elements(root: Any) -> list[Any]:
    try:
        return root.xpath("//*[local-name()='nonFraction' or local-name()='fraction']")
    except (AttributeError, etree.XPathError):
        return []


def parse_xbrl(raw: bytes) -> list[tuple[str, str, float, str | None]]:
    """Parse both XML XBRL and NSE iXBRL HTML filings.

    NSE's financial-results catalog can point to iXBRL_WEB.html documents,
    which are valid HTML rather than XML. The previous ElementTree-only parser
    therefore returned zero facts even though the filing contained numeric
    iXBRL facts. Parse HTML with lxml as the primary path and retain an XML
    fallback for pure XBRL documents.
    """
    roots: list[Any] = []
    try:
        roots.append(html.fromstring(raw))
    except (etree.ParserError, ValueError):
        pass
    try:
        roots.append(etree.fromstring(raw))
    except etree.XMLSyntaxError:
        pass

    facts: list[tuple[str, str, float, str | None]] = []
    seen: set[tuple[str, str, float, str | None]] = set()
    for root in roots:
        for elem in _ixbrl_elements(root):
            text = "".join(elem.itertext()).strip()
            value = parse_numeric(text)
            if value is None:
                continue
            scale_text = elem.get("scale")
            if scale_text:
                try:
                    value *= 10 ** int(scale_text)
                except ValueError:
                    continue
            if str(elem.get("sign", "")).strip() == "-":
                value = -abs(value)
            name = elem.get("name") or elem.get("format") or "unknown"
            context = elem.get("contextRef") or elem.get("contextref") or ""
            unit = elem.get("unitRef") or elem.get("unitref")
            fact = (name, context, value, unit)
            if fact not in seen:
                seen.add(fact)
                facts.append(fact)
        if facts:
            break
    return facts


def acquire_symbol(symbol: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    s = session()
    url = CATALOG.format(symbol=symbol)
    r = s.get(url, timeout=60)
    r.raise_for_status()
    catalog_hash = sha256_bytes(r.content)
    payload = r.json()
    rows = normalize_catalog(payload)
    records: list[dict[str, Any]] = []
    xbrl_count = 0
    for row in rows:
        period_end = first(row, "period", "periodEnd", "periodEnded", "toDate")
        broadcast = first(row, "broadCastDate", "broadcastDate", "broadcastDateTime", "broadcast_date_time")
        filing_url = clean_url(first(row, "xbrl", "xbrlFile", "xbrlFileLink"))
        base = {
            "symbol": symbol,
            "reported_date": str(period_end or ""),
            "available_date": str(broadcast or "")[:10],
            "available_timestamp": str(broadcast or ""),
            "statement_type": str(first(row, "consolidated", "consolidatedNonConsolidated") or ""),
            "filing_type": str(first(row, "audited", "auditedUnaudited") or ""),
            "source": "NSE India Financial Results",
            "source_id": f"nse-financial-results:{symbol}:{sha256_bytes(json.dumps(row, sort_keys=True, default=str).encode())[:16]}",
            "source_url": url,
            "source_sha256": catalog_hash,
            "version": str(first(row, "revision", "revisedDate", "revisedDateTime") or "original"),
            "metric_name": "__FILING_CATALOG__",
            "metric_value": 1.0,
            "currency_units": "filing-record",
            "period_metadata_json": json.dumps(row, sort_keys=True, default=str),
        }
        if not filing_url:
            records.append(base)
            continue
        try:
            xr = s.get(filing_url, timeout=60)
            xr.raise_for_status()
            xhash = sha256_bytes(xr.content)
            facts = parse_xbrl(xr.content)
            if facts:
                xbrl_count += 1
                for metric, context, value, unit in facts:
                    rec = dict(base)
                    rec.update({
                        "metric_name": metric,
                        "metric_value": value,
                        "currency_units": unit or "",
                        "xbrl_url": filing_url,
                        "xbrl_sha256": xhash,
                        "xbrl_context_ref": context,
                        "xbrl_parse_status": "parsed",
                    })
                    records.append(rec)
            else:
                base.update({"xbrl_url": filing_url, "xbrl_sha256": xhash, "xbrl_parse_status": "no_numeric_facts"})
                records.append(base)
        except requests.RequestException as exc:
            base.update({"xbrl_url": filing_url, "xbrl_parse_status": f"blocked:{type(exc).__name__}"})
            records.append(base)
    return records, {"symbol": symbol, "catalog_records": len(rows), "xbrl_documents_with_numeric_facts": xbrl_count}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--max-workers", type=int, default=8)
    args = ap.parse_args()
    symbols = get_symbols(session())
    all_records: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        futures = {pool.submit(acquire_symbol, sym): sym for sym in symbols}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            sym = futures[future]
            try:
                records, stat = future.result()
                all_records.extend(records)
                stats.append(stat)
            except Exception as exc:
                errors.append({"symbol": sym, "error": f"{type(exc).__name__}: {exc}"})
            if i % 25 == 0:
                print(f"processed {i}/{len(symbols)} symbols")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["symbol","reported_date","available_date","available_timestamp","statement_type","filing_type","source","source_id","source_url","source_sha256","version","metric_name","metric_value","currency_units","period_metadata_json","xbrl_url","xbrl_sha256","xbrl_context_ref","xbrl_parse_status"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for rec in sorted(all_records, key=lambda x: (x.get("symbol", ""), x.get("reported_date", ""), x.get("available_timestamp", ""), x.get("metric_name", ""))):
            w.writerow(rec)
    manifest = {
        "status": "complete" if not errors and any(s.get("xbrl_documents_with_numeric_facts", 0) for s in stats) else "blocked",
        "source": "NSE India Financial Results + linked NSE XBRL/iXBRL",
        "nifty500_source_url": NIFTY500_URL,
        "catalog_endpoint": "https://www.nseindia.com/api/corporates-financial-results?index=equities&symbol={symbol}&period=Quarterly",
        "symbols_expected": EXPECTED_COUNT,
        "symbols_processed": len(stats),
        "records": len(all_records),
        "symbols_with_xbrl_numeric_facts": sum(bool(s.get("xbrl_documents_with_numeric_facts")) for s in stats),
        "errors": errors,
        "policy": "Missing publication evidence or unavailable filings remain BLOCKED; no synthetic values are permitted.",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
