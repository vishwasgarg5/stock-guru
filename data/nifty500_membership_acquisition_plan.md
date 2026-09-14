# NIFTY 500 Historical Membership Acquisition

## Certification rule

Do not treat a reconstructed or inferred membership history as authoritative certification evidence. Historical membership must be backed by primary NSE/NSE Indices evidence or a licensed historical constituent-data product whose provenance can be verified.

## Discovered evidence

1. NSE/Nifty Indices publishes the current NIFTY 500 constituent CSV, but the public index page exposes the current snapshot only.
2. NSE Indices explicitly states that its data products include ongoing and historical index constituent data and that historical constituent data is available through subscription.
3. Official NIFTY 500 press releases provide primary inclusion/exclusion events and are already logged in `data/nifty500_historical_source_log.csv`.
4. A secondary open-source reconstruction (`aditya-jha/nse-historical-membership`) provides NIFTY 500 point-in-time intervals from public NSE publications. It is suitable for discovery and cross-checking, not as sole certification evidence.

## Acquisition target

Obtain a licensed/authoritative historical NIFTY 500 constituent export covering the backtest start date through the current certification date, preferably with one snapshot per effective review date and source identifiers.

Required fields:

- `as_of`
- `symbol`
- `source`
- `source_id`

Required provenance:

- provider/publisher
- retrieval date
- terms/license
- original source identifier or URL
- file SHA-256
- coverage dates

## Acceptance checks

- exactly one row per `(as_of, symbol)`
- expected NIFTY 500 cardinality for each complete snapshot
- no inferred members
- no fabricated dates
- every snapshot has provenance
- snapshot transitions reconcile with the primary event ledger
- gaps are explicitly reported and remain BLOCKED until resolved

## Current status

BLOCKED — no authoritative historical full-membership snapshot dataset has yet been acquired. Existing event evidence and secondary reconstruction must not be promoted to primary certification evidence.
