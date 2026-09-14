# Historical NIFTY 500 event chain

The event ledger stores only primary-source replacement evidence that has been explicitly reviewed. Every row preserves `effective_date`, `symbol`, `action`, `source`, and `source_id`.

## Current verified coverage

- 2021-09-30: revised periodic review
- 2022-03-31: periodic review
- 2022-04-07: ad-hoc replacement
- 2022-09-30: periodic review plus separate scheme-of-arrangement event
- 2023-03-31: semi-annual review
- 2023-09-29: periodic review
- 2023-10-26: permitted-to-trade replacement
- 2024-03-28: semi-annual review
- 2024-09-30: periodic review plus correction preserved by source ID
- 2025-03-28: periodic review
- 2025-09-23: scheme-of-amalgamation replacement
- 2025-09-30: periodic review
- 2025-12-31: demerger eligibility replacement
- 2026-03-30: semi-annual review

## Audit rules

`stock_guru.event_audit.audit_event_chain` fails closed when evidence files are missing, manifest source IDs are duplicated, manifest statuses are invalid, effective dates disagree with evidence, or event rows contain source IDs absent from the manifest. It also relies on the strict event loader, which rejects duplicate event keys and invalid state transitions.

The audit is **not** a historical universe certification. A complete certification still requires an authoritative full constituent snapshot anchor and a reconciled chain from that anchor. Missing snapshots are never inferred.
