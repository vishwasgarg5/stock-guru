# Phase B — Historical PIT fundamentals acquisition plan

## Certification objective
Acquire real historical company fundamentals for the NIFTY 500 universe with a defensible point-in-time availability date. The dataset must support survivorship-bias-free training and backtesting.

## Primary evidence source
NSE corporate filings are the preferred primary source. NSE exposes Financial Results with period-ended dates and broadcast date/time, and provides XBRL filing formats for Regulation 33 financial results. The broadcast date/time is the exchange-publication timestamp and must be retained as the PIT availability boundary.

## Required fields
- symbol
- reported_date / period_end
- available_date
- available_timestamp
- statement_type (standalone/consolidated)
- filing_type / audited_status
- metric_name
- metric_value
- currency / units
- source_url
- source_id
- source_sha256 when source bytes are captured
- version / revision identifier when supplied

## PIT rules
1. `available_timestamp` is the earliest timestamp at which the filing was publicly available from the authoritative source.
2. A fundamental observation may only be joined to market data strictly after its availability timestamp.
3. Period-end date is never treated as publication date.
4. Later restatements must not overwrite an earlier observation used by a historical backtest; each filing/version remains immutable and separately identified.
5. Missing publication evidence is BLOCKED, not inferred.
6. Vendor-derived datasets may be used for engineering validation only until provenance, licensing, and timestamp semantics are independently verified.

## Acquisition paths
1. NSE Financial Results / XBRL archives.
2. NSE Annual Report-XBRL archives where annual data is required.
3. A licensed historical fundamentals vendor may be used only as optional cross-check evidence, never as a required dependency for the free build.

## Phase B implementation now active
- Canonical PIT fundamentals validation already gates Step 3.
- `scripts/audit_pit_fundamentals.py` now provides a fail-closed certification audit.
- `tests/test_audit_pit_fundamentals.py` covers missing evidence, missing publication timestamp, valid provenance, and duplicate-version blocking.
- The next acquisition deliverable is actual historical NSE filing evidence; no placeholder fundamentals dataset will be committed.

## Validation gates
- source provenance manifest present
- source bytes or immutable source identifier captured
- publication timestamp present
- no future-availability joins
- no duplicate `(symbol, period_end, statement_type, source_id, version)` keys
- restatement/version lineage preserved
- coverage measured against the PIT universe intervals
- unresolved gaps reported explicitly

## Current status
**IN PROGRESS / BLOCKED FOR CERTIFICATION:** Phase B infrastructure is implemented, but certification remains blocked until real historical NSE filing records with publication timestamps and provenance are acquired. No synthetic, interpolated, or inferred historical fundamentals may be promoted to certification evidence.
