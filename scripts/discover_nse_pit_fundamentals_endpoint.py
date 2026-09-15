"""Discover the live NSE Financial Results data endpoint without guessing it.

Phase B deliberately starts from the authoritative NSE page and inspects its
published JavaScript for the endpoint used by the Financial Results UI. The
script records only discovered URLs; it never invents an endpoint or data.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests

PAGE_URL = "https://www.nseindia.com/companies-listing/corporate-filings-financial-results"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def discover(output: Path) -> dict[str, object]:
    session = requests.Session()
    response = session.get(PAGE_URL, headers=HEADERS, timeout=60)
    response.raise_for_status()
    html = response.text

    script_urls = [urljoin(PAGE_URL, src) for src in re.findall(r'<script[^>]+src=["\']([^"\']+)', html, re.I)]
    candidates: set[str] = set()
    inspected_scripts = 0
    errors: list[str] = []

    for script_url in script_urls:
        try:
            js = session.get(script_url, headers={**HEADERS, "Accept": "*/*"}, timeout=60)
            js.raise_for_status()
            inspected_scripts += 1
        except requests.RequestException as exc:
            errors.append(f"{script_url}: {exc}")
            continue
        for match in re.finditer(r"[\"']([^\"']*(?:corporate|financial)[^\"']*(?:api|result)[^\"']*)[\"']", js.text, re.I):
            value = match.group(1)
            if "/api/" in value.lower() or "api.nseindia.com" in value.lower():
                candidates.add(urljoin(PAGE_URL, value))
        for match in re.finditer(r"https?://[^\"'\s]+api[^\"'\s]+", js.text, re.I):
            if "nseindia.com" in match.group(0).lower() and ("financial" in match.group(0).lower() or "result" in match.group(0).lower()):
                candidates.add(match.group(0))

    result = {
        "status": "discovered" if candidates else "blocked",
        "source_page": PAGE_URL,
        "source_page_sha256": __import__("hashlib").sha256(response.content).hexdigest(),
        "script_count": len(script_urls),
        "scripts_inspected": inspected_scripts,
        "candidate_endpoints": sorted(candidates),
        "errors": errors,
        "policy": "No endpoint or filing data is inferred; absence of a discovered authoritative endpoint is BLOCKED.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = discover(args.output)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "discovered" else 2)
