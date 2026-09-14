# NIFTY 500 Historical Constituent Data Request

## Purpose

Acquire authoritative point-in-time NIFTY 500 membership snapshots for certification. Do not use reconstructed or inferred membership as the primary certification dataset.

## Primary provider

NSE Indices Limited.

Official data-subscription documentation states that NSE Indices provides ongoing and historical index constituent data, including constituent identifiers and related index data. Historical constituent data is offered as a subscription/licensed product.

Official contact: `indices@nseids.co.in`

## Requested coverage

- NIFTY 500
- Backtest start date through 2026-09-15
- Prefer one complete constituent snapshot per effective review/reconstitution date
- Include exceptional/interim changes where applicable

## Required data fields

- `as_of`
- `symbol`
- `source`
- `source_id`

If available, also retain company name, index weight, effective date, and any vendor/NSE security identifier without replacing the required fields.

## Required provenance

- Provider/publisher
- Original dataset/product name
- Retrieval date
- License/terms
- Original source identifier or URL
- Source file SHA-256
- Coverage start/end
- Snapshot frequency

## Acceptance criteria

1. Exactly one row per `(as_of, symbol)`.
2. Complete snapshots must contain the expected NIFTY 500 constituent count for that effective date.
3. No inferred, reconstructed, or synthetic membership rows.
4. Every snapshot has provenance.
5. Snapshot transitions reconcile against the repository's primary NSE event ledger.
6. Any coverage gap remains explicitly BLOCKED; it is never filled by inference.
7. Source fingerprints must be recorded before certification.

## Current decision

**BLOCKED pending acquisition.** The repository already contains primary NSE/NIFTY Indices event evidence, but event evidence is not equivalent to complete historical membership snapshots. The current public NIFTY 500 page exposes a current constituent download, while NSE Indices documents historical constituent data as a subscription product.

## After receipt

1. Store the licensed source file outside the repository if redistribution is prohibited.
2. Store only the permitted derived snapshot CSV and provenance manifest in the repository.
3. Run `validate_source_bundle(...)` with full-snapshot enforcement.
4. Run snapshot-to-event reconciliation and PIT universe audit.
5. Only then unblock the real Step 3 PIT backtest.
