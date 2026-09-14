# Stock Guru

Adaptive NIFTY 500 stock-selection and next-day OHLC forecasting pipeline.

## Goal

1. Rank the NIFTY 500 using fundamental and technical features.
2. Select a small top-k candidate set.
3. Predict next-day Open, High, Low and Close as normalized returns.
4. Compare predictions with the next trading day's actual OHLC.
5. Log errors and evaluate walk-forward performance.
6. Retrain only when new labeled observations are available, while preserving an untouched validation/test period.

## Current implementation

The repository contains the model baseline plus guarded infrastructure for point-in-time research, paper execution, feedback settlement, and an optional temporal challenger. The bootstrap data layer fetches the current NIFTY 500 universe and daily NSE OHLCV history. Current constituents are useful for pipeline development, but are not a substitute for historical constituent membership.

Production hardening includes strict market-session calendar validation, paper-trading configuration guards, source-file fingerprints for PIT universe and fundamentals, provenance-pair validation for filing-derived fundamentals, model-artifact integrity checks, fail-closed rollback/recovery decisions, deterministic production smoke checks, and a final readiness/certification evidence gate. These controls improve reproducibility and fail-closed behavior; they do not create missing historical data.

## Roadmap status

- **Core model, ranking, OHLC forecasting, risk, uncertainty, calibration, walk-forward, costs, feedback, retraining, paper trading, and challenger infrastructure:** implemented and regression protected.
- **PIT universe:** interval builder, as-of filtering, provenance-bearing baseline/event reconstruction, coverage-quality reporting, interval integrity checks, source-gap diagnostics, fingerprints, audit reports, and reconciliation controls are implemented. Real historical NIFTY 500 constituent evidence still must be supplied and verified.
- **Historical event evidence:** official NSE Indices releases are captured for verified 2021-2026 replacement events; each event retains its source ID, effective date, source URL, and action. Same-effective-date releases remain provenance-distinct. The event-chain audit blocks missing evidence, invalid manifest statuses, date mismatches, and unknown source IDs.
- **PIT fundamentals:** canonical filing-derived schema validation, availability-date protection, provenance validation, source fingerprinting, and quality reporting are implemented. Real historical filing values still must be supplied and verified.
- **Steps 27-30:** rollback/failure handling, production smoke checks, final readiness reporting, and READY/BLOCKED certification are implemented and regression protected.
- **Production certification:** currently **BLOCKED** until real, independently verifiable historical NIFTY 500 membership and filing-derived fundamentals are supplied, fingerprinted, reconciled, and validated. CI green alone cannot override this evidence gate.

## Production safety commands

Run the event-chain audit:

```bash
PYTHONPATH=src python -m stock_guru.cli event-audit
```

Run the deterministic production smoke checks:

```bash
PYTHONPATH=src python -m stock_guru.cli production-smoke
```

Generate the final readiness report from a JSON evidence object:

```bash
PYTHONPATH=src python -m stock_guru.cli readiness-report --evidence artifacts/readiness_evidence.json
```

Run smoke checks plus the final certification decision:

```bash
PYTHONPATH=src python -m stock_guru.cli certify --evidence artifacts/readiness_evidence.json
```

A `PASS` from the smoke test means the production safety path is functioning; it does **not** certify the historical universe. `certify` emits `READY` only when every required evidence gate passes, otherwise it emits `BLOCKED`.

## Final readiness gate

Use `stock_guru.final_readiness.build_final_readiness_report` after the real PIT datasets and evaluation artifacts have been produced. The report requires verified historical data, completed backtest and walk-forward evaluation, validated paper trading and feedback, and reproducible model artifacts. Missing evidence results in `blocked` rather than an inferred or fabricated success.

```python
from stock_guru.final_readiness import build_final_readiness_report

report = build_final_readiness_report(
    model_exists=True,
    pit_universe_validated=True,
    fundamentals_validated=True,
    monitoring_configured=True,
    tests_green=True,
    historical_data_available=True,
    historical_data_verified=True,
    backtest_completed=True,
    walk_forward_completed=True,
    paper_trading_validated=True,
    feedback_cycle_validated=True,
    artifact_reproducible=True,
)
```

## Important research limitation

The remaining non-code dependency is **real, independently verifiable point-in-time NIFTY 500 membership and filing-derived fundamentals**. Until those sources are populated, fingerprinted, reconciled, and validated, the project must not publish survivorship-bias-free historical performance claims or declare production readiness.
