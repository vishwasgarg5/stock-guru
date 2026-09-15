# Historical PIT Fundamentals Mapping Policy

Status: BLOCKED until authoritative historical security identifiers are available.

## Rule
Historical NIFTY 500 membership records contain issuer/security names. They must not be converted to NSE symbols by fuzzy matching, current-name inference, guessed ticker changes, or synthetic mappings.

## Accepted evidence
A historical member may enter the PIT fundamentals universe only when an authoritative source establishes the mapping from the historical constituent/security identity to an NSE security identifier/symbol for the relevant period. Accepted evidence must preserve:

- historical constituent/security name
- NSE symbol/security identifier
- effective start/end dates when available
- authoritative source URL or source ID
- retrieval timestamp
- source SHA-256
- mapping version/provenance

## Fail-closed behavior
Unmapped historical constituents remain BLOCKED and are excluded from certification rather than assigned an inferred symbol. PIT fundamentals certification cannot become `complete` while required historical mappings are missing.

## Current implementation boundary
The current NSE acquisition path can collect real XBRL/iXBRL fundamentals for the official current NIFTY 500 symbol list. That is evidence acquisition, not historical PIT certification. Historical membership-to-identifier mapping remains a separate evidence requirement.
