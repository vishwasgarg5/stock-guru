# Historical PIT fundamentals acquisition plan

## Certification objective
Acquire real historical company fundamentals for the NIFTY 500 universe with a defensible point-in-time availability date. The dataset must support survivorship-bias-free training and backtesting.

## Primary evidence source
NSE corporate filings are the preferred primary source. NSE exposes Financial Results with period-ended dates and broadcast date/time, and provides XBRL filing formats for Regulation 33 financial results. The broadcast date/time is the exchange-publication timestamp and must be retained as the PIT availability boundary.

## Required fields
- symbol
- period_end
- available_date
- available_timestamp (when supplied)
- statement_type (standalone/consolidated)
- filing_type / audited_status
- metric_name
- metric_value
- currency / units
- source_url
- source_id
- source_sha256 when the source bytes are captured

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
3. A licensed historical fundamentals vendor may be used if it supplies immutable filing/version identifiers and publication timestamps.

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
BLOCKED: no complete authoritative historical PIT fundamentals dataset has been imported into the repository yet.

No synthetic, interpolated, or inferred historical fundamentals may be promoted to certification evidence.
