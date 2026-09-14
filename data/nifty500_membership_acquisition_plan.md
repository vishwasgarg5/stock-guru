# NIFTY 500 Historical Membership — Free Public Evidence Plan

## Objective

Keep Stock Guru independent of paid historical constituent-data subscriptions. The core PIT universe pipeline must work from publicly obtainable primary NSE/NIFTY Indices evidence.

## Certification rule

Do not treat a reconstructed or inferred membership history as authoritative by itself. However, membership intervals may be marked `EVENT_DERIVED` when they are deterministically reconstructed from an authoritative public snapshot plus a complete chain of verified primary NSE/NIFTY Indices membership events.

Secondary reconstructions are `CROSS_CHECK_ONLY`: they may identify discrepancies or missing releases but can never fill a gap.

## Public evidence already available

1. NSE/Nifty Indices publishes the current NIFTY 500 constituent CSV.
2. Official NIFTY 500/NIFTY Indices press releases provide primary inclusion/exclusion events and are logged in `data/nifty500_historical_source_log.csv`.
3. The repository contains event-level evidence files and validation targets for the verified releases.
4. Public secondary reconstructions can be used for cross-checking only.

## Free reconstruction target

Build a point-in-time membership history from:

`public current snapshot -> verified primary events -> deterministic membership intervals`

Required fields:

- `as_of`
- `symbol`
- `source`
- `source_id`
- `evidence_tier`

Required provenance:

- provider/publisher
- retrieval date
- original source identifier or URL
- source SHA-256 where the source bytes are locally captured
- coverage dates

## Acceptance checks

- exactly one row per `(as_of, symbol)`
- deterministic state transitions
- every transition references a verified primary source ID
- no inferred members
- no fabricated dates
- no silent carry-forward across an unsupported evidence gap
- secondary datasets never fill missing primary evidence
- gaps are explicitly reported
- unsupported dates remain `BLOCKED`
- complete public event-derived intervals may be marked `EVENT_DERIVED`

## Cost policy

A paid NSE historical constituent subscription is optional for independent comparison only. It must never be required by the core model, tests, backtests, or certification pipeline.

## Current status

IMPLEMENTATION PATH: FREE_PUBLIC_EVIDENCE. Historical periods remain `BLOCKED` until their public primary event chain is complete; no paid dataset is required to proceed with the reconstruction work.