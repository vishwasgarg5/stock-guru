# Free Public Evidence Policy

## Objective

Stock Guru is designed to remain usable without paid market-data subscriptions. Paid NSE historical constituent products must not be a required dependency for research, backtesting, or the model's reproducible data pipeline.

## Evidence tiers

- `CERTIFIED`: supported by a complete public primary snapshot or complete chain of verified public primary events.
- `EVENT_DERIVED`: membership reconstructed deterministically from a public authoritative snapshot plus verified NSE/NIFTY Indices membership events.
- `CROSS_CHECK_ONLY`: secondary reconstruction used only to identify discrepancies or missing evidence. It cannot create or repair membership history.
- `BLOCKED`: required public evidence is missing, contradictory, incomplete, or cannot establish the requested point-in-time state.

## Rules

1. Never purchase or require proprietary data to make the core pipeline work.
2. Never fabricate a constituent, effective date, filing date, or fundamental value.
3. Never use a secondary reconstruction as the sole certification source.
4. Public primary NSE/NIFTY Indices releases may be used to derive membership intervals when the transition chain is complete.
5. Every derived interval must retain source identifiers and provenance.
6. Missing transitions create explicit gaps and remain `BLOCKED`; the system must not silently carry membership across an unsupported gap.
7. Historical fundamentals must likewise use publicly obtainable filings/results and point-in-time publication/availability evidence.
8. A `CERTIFIED` production decision still requires all final readiness gates; free data does not lower the evidence standard.

## Intended workflow

`public current snapshot -> verified public events -> deterministic PIT intervals -> audit -> backtest`

Secondary public datasets may be compared against the result for anomaly detection, but never used as an authoritative fill source.

## Cost guarantee

The repository must not require Bloomberg, FactSet, paid NSE historical constituent feeds, or another proprietary data subscription to execute its core validation and research pipeline. Optional paid data may be used by an operator for independent comparison, but the model cannot depend on it.

## Current certification consequence

Historical membership remains `BLOCKED` where the public primary event chain is incomplete. The correct response to an evidence gap is to report the gap, not to substitute inferred history.