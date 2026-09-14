# Free PIT Universe Schema

The free-public-evidence universe stores reconstructed membership as explicit intervals.

| Field | Meaning |
|---|---|
| `symbol` | Normalized NSE symbol |
| `start_date` | First date supported by the anchor/event chain |
| `end_date` | Exclusion effective date; null means still active in the reconstructed range |
| `evidence_tier` | `EVENT_DERIVED` for deterministic primary-event reconstruction |
| `source_ids` | Primary NSE/NIFTY Indices source identifiers supporting the interval |

## Safety rules

- An exclusion for a non-active symbol is an error.
- An inclusion for an already-active symbol is an error.
- Cardinality mismatches are `BLOCKED`, not repaired.
- Unsupported historical dates are never filled by secondary datasets.
- A paid historical constituent feed is not required.
