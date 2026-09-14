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

## Roadmap status

- **Step 24 — Point-in-time universe:** interval builder, as-of filtering, provenance-bearing baseline/event reconstruction, coverage-quality reporting, interval integrity checks, and source-gap diagnostics are implemented. Real historical NIFTY 500 constituent events still need to be populated from a trustworthy historical source.
- **Step 25 — Point-in-time fundamentals:** canonical filing-derived schema validation is implemented. Real filing/history ingestion still needs to be connected; no historical values are fabricated.
- **Step 26 — Paper trading:** next-session execution, position caps, slippage/commission accounting, and idempotent trade persistence are implemented.
- **Step 27 — Feedback/retraining:** prediction settlement and validation-gated adaptive retraining are wired through the existing ledger/retrainer path.
- **Step 28 — Temporal challenger:** an optional PyTorch LSTM challenger is isolated from the production XGBoost path and cannot silently replace it.
- **Steps 29–43 — PIT/data-quality hardening:** strict PIT fundamentals joining, optional PIT universe filtering, numeric/finite fundamental validation, market-input guards, OHLC fail-closed behavior, source manifests, provenance-aware coverage, and regression protection are implemented.
- **Steps 44–53 — PIT research integration:** walk-forward and strategy backtests consume optional PIT fundamentals/universe intervals; CLI train/predict/backtest accept those datasets; empty PIT universes fail closed; interval overlaps are rejected; source snapshot gaps are reported; and PIT regression tests are in CI.
- **Step 54 onward:** import verified historical NIFTY 500 evidence, run source fingerprint/coverage audits, populate dated snapshots/events, connect real filing-derived fundamentals, and only then publish survivorship-bias-free historical performance claims.

## Design

- `src/stock_guru/data.py`: current NIFTY 500 universe + OHLCV ingestion.
- `src/stock_guru/universe_history.py`: point-in-time constituent snapshots, interval validation, and membership filtering.
- `src/stock_guru/universe_events.py`: provenance-bearing inclusion/exclusion events and baseline reconstruction.
- `src/stock_guru/universe_ingest.py`: normalization and validation for PIT universe snapshots/events.
- `src/stock_guru/universe_source.py`: provenance manifest, source validation, and snapshot gap diagnostics.
- `src/stock_guru/universe_coverage.py`: PIT universe coverage and integrity report without inferring historical completeness.
- `src/stock_guru/fundamentals.py`: legacy-compatible PIT fundamentals loading/as-of join.
- `src/stock_guru/fundamentals_ingest.py`: validation/normalization contract for filing-derived PIT fundamentals.
- `src/stock_guru/pit.py`: strict point-in-time fundamentals as-of join used by feature construction.
- `src/stock_guru/features.py`: leakage-safe technical/fundamental feature engineering.
- `src/stock_guru/ranker.py`: XGBoost learning-to-rank stock selector.
- `src/stock_guru/ohlc.py`: four XGBoost regressors for normalized next-day OHLC returns.
- `src/stock_guru/evaluation.py`: error metrics and prediction labeling.
- `src/stock_guru/pipeline.py`: train/predict orchestration with optional PIT fundamentals and universe filtering.
- `src/stock_guru/walk_forward.py`: expanding-window PIT-aware validation.
- `src/stock_guru/retrainer.py`: validation-gated model replacement and labeled prediction storage.
- `src/stock_guru/feedback.py`: next-session prediction settlement and feedback labeling.
- `src/stock_guru/paper_trading.py`: paper execution with costs and position limits.
- `src/stock_guru/temporal_challenger.py`: optional isolated LSTM challenger.
- `src/stock_guru/cli.py`: command-line entry point.
- `tests/`: regression and smoke tests.

## Data contract

The modeling input must contain at least:

- `date`, `symbol`, `open`, `high`, `low`, `close`, `volume`
- optional point-in-time fundamental columns: `roe`, `roce`, `eps_growth`, `revenue_growth`, `pe`, `pb`, `debt_to_equity`, `operating_margin`, `free_cash_flow`

Never use a fundamental value before its public availability date. The strict PIT join in `stock_guru.pit` uses `available_date` as the eligibility boundary and never substitutes a period-end or report date for availability.

Market data must have at most one row per `date`/`symbol`; malformed dates, blank symbols, or missing OHLCV columns are rejected during feature construction.

PIT universe snapshots must contain `as_of` and `symbol`. Event imports must additionally contain `effective_date`, `action`, `source`, and `source_id`. Symbols and actions are normalized, dates are normalized, duplicates are rejected, and event actions are limited to `include`/`exclude`. Generated membership intervals must not overlap for the same symbol.

## Validate a historical source before ingestion

Every real historical snapshot dataset should be accompanied by a provenance manifest containing `dataset`, `source_name`, `source_url`, `retrieved_at`, and `license_or_terms`.

```bash
PYTHONPATH=src python -m stock_guru.cli universe-source-validate \
  --snapshots data/nifty500_snapshots.csv \
  --manifest data/nifty500_source_manifest.json \
  --gap-threshold-days 180
```

The validator normalizes symbols and dates, rejects duplicate observations, reports the observed snapshot span and constituent-count range, and surfaces long gaps without filling them. If a source contract explicitly guarantees a fixed snapshot size, add `--expected-constituents N --require-full-snapshot-size`; otherwise source-specific counts are allowed.

## Check universe coverage

After generating snapshots, produce a machine-readable coverage report:

```bash
PYTHONPATH=src python -m stock_guru.cli universe-quality \
  --snapshots data/nifty500_snapshots.csv \
  --events data/nifty500_events.csv \
  --manifest data/nifty500_source_manifest.json \
  --output artifacts/universe_quality.json
```

The report records the supplied snapshot span, snapshot counts, unique constituents, event counts, event span, event provenance completeness, and source provenance. It deliberately reports `historical_completeness: unknown`; coverage evidence is not treated as proof that every historical rebalance has been captured.

## Train / predict / backtest with PIT data

```bash
PYTHONPATH=src python -m stock_guru.cli train \
  --prices data/prices.csv \
  --fundamentals data/fundamentals.csv \
  --universe data/nifty500_universe_history.csv \
  --model-dir artifacts

PYTHONPATH=src python -m stock_guru.cli backtest \
  --prices data/prices.csv \
  --fundamentals data/fundamentals.csv \
  --universe data/nifty500_universe_history.csv \
  --output-dir artifacts/backtest
```

The walk-forward path applies the same PIT universe and fundamental inputs to training and prediction folds. An empty eligible universe fails closed rather than silently producing an unrestricted result.

## Important research limitation

The remaining data work is **real point-in-time NIFTY 500 membership and filing-derived fundamentals**. Until those sources are populated and validated, historical performance must not be described as survivorship-bias-free.
