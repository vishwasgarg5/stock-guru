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

Production hardening includes strict market-session calendar validation, paper-trading configuration guards, source-file fingerprints for PIT universe and fundamentals, provenance-pair validation for filing-derived fundamentals, model-artifact integrity checks, and a final readiness evidence gate. These controls improve reproducibility and fail-closed behavior; they do not create missing historical data.

## Roadmap status

- **Core model, ranking, OHLC forecasting, risk, uncertainty, calibration, walk-forward, costs, feedback, retraining, paper trading, and challenger infrastructure:** implemented and regression protected.
- **PIT universe:** interval builder, as-of filtering, provenance-bearing baseline/event reconstruction, coverage-quality reporting, interval integrity checks, source-gap diagnostics, fingerprints, audit reports, and reconciliation controls are implemented. Real historical NIFTY 500 constituent evidence still must be supplied and verified.
- **PIT fundamentals:** canonical filing-derived schema validation, availability-date protection, provenance validation, source fingerprinting, and quality reporting are implemented. Real historical filing values still must be supplied and verified.
- **Production readiness:** the final gate now requires both engineering gates and real historical-data/evaluation evidence. It cannot report `ready` merely because tests pass.

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
