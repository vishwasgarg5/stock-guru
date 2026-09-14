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

- **Step 24 — Point-in-time universe:** interval builder, as-of filtering, provenance-bearing baseline/event reconstruction, and coverage-quality reporting are implemented. Coverage validation rejects invalid dates, blank provenance, unsupported actions, and duplicate symbol/date events. Real historical NIFTY 500 constituent events still need to be populated from a trustworthy historical source.
- **Step 25 — Point-in-time fundamentals:** canonical filing-derived schema validation is implemented. Real filing/history ingestion still needs to be connected; no historical values are fabricated.
- **Step 26 — Paper trading:** next-session execution, position caps, slippage/commission accounting, and idempotent trade persistence are implemented.
- **Step 27 — Feedback/retraining:** prediction settlement and validation-gated adaptive retraining are wired through the existing ledger/retrainer path.
- **Step 28 — Temporal challenger:** an optional PyTorch LSTM challenger is isolated from the production XGBoost path and cannot silently replace it.
- **Steps 29–38 — PIT/data-quality hardening:** a reusable point-in-time fundamentals as-of join is available; feature construction and the model pipeline can consume PIT fundamentals and optional PIT universe intervals; fundamental values are validated for numeric/finite content; market feature inputs reject malformed or duplicate symbol/date observations; OHLC training/prediction fails closed on empty usable data; regression tests cover the new guards. These steps improve research safety but do not claim that the underlying historical constituent or filing data are complete.

## Design

- `src/stock_guru/data.py`: current NIFTY 500 universe + OHLCV ingestion.
- `src/stock_guru/universe_history.py`: point-in-time constituent snapshots and membership intervals.
- `src/stock_guru/universe_events.py`: provenance-bearing inclusion/exclusion events and baseline reconstruction.
- `src/stock_guru/universe_coverage.py`: PIT universe coverage and integrity report without inferring historical completeness.
- `src/stock_guru/fundamentals.py`: legacy-compatible PIT fundamentals loading/as-of join.
- `src/stock_guru/fundamentals_ingest.py`: validation/normalization contract for filing-derived PIT fundamentals.
- `src/stock_guru/pit.py`: strict point-in-time fundamentals as-of join used by feature construction.
- `src/stock_guru/features.py`: leakage-safe technical/fundamental feature engineering.
- `src/stock_guru/ranker.py`: XGBoost learning-to-rank stock selector.
- `src/stock_guru/ohlc.py`: four XGBoost regressors for normalized next-day OHLC returns.
- `src/stock_guru/evaluation.py`: error metrics and prediction labeling.
- `src/stock_guru/pipeline.py`: train/predict orchestration with optional PIT fundamentals and universe filtering.
- `src/stock_guru/walk_forward.py`: expanding-window validation.
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

## Install

```bash
pip install -r requirements.txt
```

## Download a bootstrap dataset

```bash
PYTHONPATH=src python -c "from stock_guru.data import download_nifty500_prices; download_nifty500_prices(start='2018-01-01')"
```

This writes `data/prices.csv` and `data/nifty500_universe.csv`. The downloader uses the current constituent list, so this dataset is suitable for pipeline development but **not** a fully unbiased historical NIFTY 500 backtest.

## Build a point-in-time universe history

Supply an authoritative one-date baseline and a provenance-bearing event CSV. Events must contain `effective_date`, `symbol`, `action` (`include` or `exclude`), `source`, and `source_id`.

```bash
PYTHONPATH=src python -m stock_guru.cli universe-history \
  --baseline data/nifty500_baseline.csv \
  --events data/nifty500_events.csv \
  --output data/nifty500_universe_history.csv
```

The reconstruction is deliberately conservative: it does not invent membership before the supplied baseline, rejects duplicate symbol/date events, and rejects impossible include/exclude transitions.

## Check universe coverage

After generating snapshots, produce a machine-readable coverage report:

```bash
PYTHONPATH=src python -m stock_guru.cli universe-quality \
  --snapshots data/nifty500_snapshots.csv \
  --events data/nifty500_events.csv \
  --output artifacts/universe_quality.json
```

The report records the supplied snapshot span, snapshot counts, unique constituents, event counts, event span, and event provenance completeness. Invalid dates, blank symbols/provenance, unsupported actions, and duplicate effective-date/symbol events are rejected. It deliberately reports `historical_completeness: unknown`; coverage evidence is not treated as proof that every historical rebalance has been captured.

## Train

```bash
PYTHONPATH=src python -m stock_guru.cli train --prices data/prices.csv --model-dir artifacts
```

## Predict

```bash
PYTHONPATH=src python -m stock_guru.cli predict --prices data/prices.csv --model-dir artifacts --date 2026-01-02
```

## Walk-forward research

Use `stock_guru.walk_forward.run_walk_forward()` with an expanding training window. Do not use random train/test splits for this time-series problem.

## Paper trading

`stock_guru.paper_trading.execute_signals()` consumes accepted prediction signals and enters at the first subsequent session open, applying configured slippage, commission, and position caps. Use `save_trades()` for idempotent persistence.

## Important research limitation

The remaining data work is **real point-in-time NIFTY 500 membership and filing-derived fundamentals**. Until those sources are populated and validated, historical performance must not be described as survivorship-bias-free.
