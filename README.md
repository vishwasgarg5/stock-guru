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

The repository now contains the model baseline plus a market-data bootstrap layer. The current data layer fetches the current NIFTY 500 universe from NSE Indices and downloads daily NSE OHLCV history through yfinance. NSE Indices describes NIFTY 500 as a 500-company broad-market index and publishes its constituent data through its index resources. For historical research, persist dated constituent snapshots because today's universe is not a valid proxy for every historical NIFTY 500 membership.

## Design

- `src/stock_guru/data.py`: current NIFTY 500 universe + OHLCV ingestion.
- `src/stock_guru/features.py`: leakage-safe technical/fundamental feature engineering.
- `src/stock_guru/ranker.py`: XGBoost learning-to-rank stock selector.
- `src/stock_guru/ohlc.py`: four XGBoost regressors for normalized next-day OHLC returns.
- `src/stock_guru/evaluation.py`: error metrics and prediction labeling.
- `src/stock_guru/pipeline.py`: train/predict orchestration.
- `src/stock_guru/walk_forward.py`: expanding-window validation.
- `src/stock_guru/retrainer.py`: validation-gated model replacement and labeled prediction storage.
- `src/stock_guru/cli.py`: command-line entry point.
- `tests/`: smoke tests.

## Data contract

The modeling input must contain at least:

- `date`, `symbol`, `open`, `high`, `low`, `close`, `volume`
- optional point-in-time fundamental columns: `roe`, `roce`, `eps_growth`, `revenue_growth`, `pe`, `pb`, `debt_to_equity`, `operating_margin`, `free_cash_flow`

Never use a fundamental value before its public availability date. This project treats leakage prevention as a first-class requirement.

## Install

```bash
pip install -r requirements.txt
```

## Download a bootstrap dataset

```bash
PYTHONPATH=src python -c "from stock_guru.data import download_nifty500_prices; download_nifty500_prices(start='2018-01-01')"
```

This writes `data/prices.csv` and `data/nifty500_universe.csv`. The downloader uses the current constituent list, so this dataset is suitable for pipeline development but **not** a fully unbiased historical NIFTY 500 backtest.

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

## Important research limitation

The next major data task is **point-in-time fundamentals and point-in-time NIFTY 500 membership**. Current fundamentals or today's constituent list must not be retroactively applied to historical dates, otherwise the reported model performance will be contaminated by look-ahead/survivorship bias.
