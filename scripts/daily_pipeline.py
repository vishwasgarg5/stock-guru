from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd

from stock_guru.data import download_nifty500_prices
from stock_guru.daily import predict_daily
from stock_guru.pipeline import Pipeline


def save_model(pipe: Pipeline, model_dir: Path, model_version: str = "") -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    pipe.ranker.save(str(model_dir / "ranker.joblib"))
    pipe.forecaster.save(str(model_dir / "ohlc.joblib"))
    pd.Series(pipe.features).to_csv(model_dir / "features.csv", index=False, header=False)
    if model_version:
        (model_dir / "model_metadata.json").write_text(json.dumps({"model_version": model_version}, indent=2), encoding="utf-8")


def train_before_prediction_date(market: pd.DataFrame, prediction_date: pd.Timestamp) -> Pipeline:
    normalized = pd.to_datetime(market["date"]).dt.normalize()
    train = market.loc[normalized < prediction_date].copy()
    if train.empty:
        raise RuntimeError(f"No historical sessions exist before prediction date {prediction_date.date()}")
    return Pipeline().train(train)


def main() -> None:
    p = argparse.ArgumentParser(description="Run the daily Stock Guru data/prediction cycle")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--prices", default="data/prices.csv")
    p.add_argument("--fundamentals", default=None)
    p.add_argument("--universe-snapshots", default="data/universe_snapshots.csv")
    p.add_argument("--model-dir", default="artifacts")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--prediction-date", default=None)
    p.add_argument("--output", default="artifacts/daily_predictions.csv")
    p.add_argument("--prediction-store", default="artifacts/prediction_ledger.csv")
    args = p.parse_args()

    download_nifty500_prices(args.start, output=args.prices)
    market = pd.read_csv(args.prices, parse_dates=["date"])
    if market.empty:
        raise RuntimeError("No market data was downloaded")
    available_dates = pd.to_datetime(market["date"]).dt.normalize().drop_duplicates().sort_values()
    latest_date = available_dates.iloc[-1].date().isoformat()
    prediction_date = args.prediction_date or latest_date
    prediction_ts = pd.Timestamp(prediction_date).normalize()
    if prediction_ts not in set(available_dates):
        raise ValueError(f"Prediction date {prediction_date} is not an available market session")
    model_dir = Path(args.model_dir)
    if not (model_dir / "ranker.joblib").exists() or not (model_dir / "ohlc.joblib").exists():
        print(f"No trained model found; training only on history before {prediction_ts.date()}.")
        save_model(train_before_prediction_date(market, prediction_ts), model_dir, model_version=f"bootstrap-{prediction_ts.date()}")
    predictions = predict_daily(args.prices, args.model_dir, prediction_date, top_k=args.top_k,
                                fundamentals_path=args.fundamentals,
                                universe_snapshots_path=args.universe_snapshots,
                                prediction_store=args.prediction_store)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(predictions.to_string(index=False))
    print(f"Saved {len(predictions)} predictions for {prediction_date} to {args.output}")


if __name__ == "__main__":
    main()
