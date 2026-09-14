# NIFTY 500 September 2022 reconciliation

## Evidence

- 1 Sep 2022 periodic review: 18 exclusions + 18 inclusions, effective 30 Sep 2022.
- 16 Sep 2022 scheme-of-arrangement release: EQUITAS excluded and SHOPERSTOP included, also effective 30 Sep 2022.

## Rule

These are **two distinct same-effective-date event sources**, not duplicate snapshots. The canonical event ledger must retain both source IDs and apply both event sets in deterministic source order.

## Deterministic ordering

1. `ind_prs01092022` — periodic review.
2. `ind_prs16092022` — scheme-of-arrangement replacement.

The resulting membership state is therefore the state after both releases, rather than the state after either release independently.

## Certification status

**VERIFIED EVENT EVIDENCE / NOT YET A FULL SNAPSHOT CERTIFICATION.** No constituent snapshot is inferred from these events alone.
