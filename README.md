# Stock Guru

Adaptive NIFTY 500 stock-selection and next-day OHLC forecasting pipeline.

## Goal

1. Rank the NIFTY 500 using fundamental and technical features.
2. Select a small top-k candidate set.
3. Predict next-day Open, High, Low and Close as normalized returns.
4. Compare predictions with the next trading day's actual OHLC.
5. Log errors and evaluate walk-forward performance.
6. Retrain only when new labeled observations are available, while preserving an untouched test period.

## Initial scope

This repository provides the modeling skeleton. It deliberately does **not** bundle proprietary market/fundamental data or claim live predictions. Supply point-in-time data through CSV files and run the pipeline.

## Design

- `src/stock_guru/features.py`: technical/fundamental feature engineering.
- `src/stock_guru/ranker.py`: XGBoost learning-to-rank stock selector.
- `src/stock_guru/ohlc.py`: four XGBoost regressors for normalized next-day OHLC returns.
- `src/stock_guru/evaluation.py`: error metrics and prediction logging.
- `src/stock_guru/pipeline.py`: train/predict/evaluate orchestration.
- `src/stock_guru/cli.py`: command-line entry point.
- `tests/`: smoke tests.

XGBoost's dedicated `XGBRanker` supports learning-to-rank objectives such as NDCG, which is appropriate for ranking stocks within each trading date. See the official documentation: https://xgboost.readthedocs.io/en/latest/python/examples/learning_to_rank.html

## Data contract

The input price/fundamental table should contain at least:

- `date`, `symbol`, `open`, `high`, `low`, `close`, `volume`
- optional point-in-time fundamental columns such as `roe`, `roce`, `eps_growth`, `revenue_growth`, `pe`, `pb`, `debt_to_equity`, `operating_margin`, `free_cash_flow`

Never use a fundamental value before its public availability date. This project treats leakage prevention as a first-class requirement.

## Install

```bash
pip install -r requirements.txt
```

## Example

```bash
python -m stock_guru.cli train --prices data/prices.csv --model-dir artifacts
python -m stock_guru.cli predict --prices data/prices.csv --model-dir artifacts --date 2026-01-02
python -m stock_guru.cli evaluate --predictions artifacts/predictions.csv
```

The commands are intentionally data-source agnostic so a later connector can be added without rewriting the models.
