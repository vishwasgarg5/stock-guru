from __future__ import annotations

import argparse
from pathlib import Path
from datetime import date, timedelta

import pandas as pd

from stock_guru.data import download_nifty500_prices
from stock_guru.daily import predict_daily
from stock_guru.pipeline import Pipeline


def main() -> None:
    p = argparse.ArgumentParser(description="Run the daily Stock Guru data/prediction cycle")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--prices", default="data/prices.csv")
    p.add_argument("--fundamentals", default=None)
    p.add_argument("--model-dir", default="artifacts")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--prediction-date", default=None)
    p.add_argument("--output", default="artifacts/daily_predictions.csv")
    args = p.parse_args()

    yesterday = date.today() - timedelta(days=1)
    download_nifty500_prices(args.start, yesterday.isoformat(), args.prices)

    market = pd.read_csv(args.prices, parse_dates=["date"])
    latest_date = market["date"].max().date().isoformat()
    prediction_date = args.prediction_date or latest_date

    model_dir = Path(args.model_dir)
    if not (model_dir / "ranker.joblib").exists() or not (model_dir / "ohlc.joblib").exists():
        print("No trained model found; training on the refreshed market history.")
        pipe = Pipeline(top_k=args.top_k).train(market)
        model_dir.mkdir(parents=True, exist_ok=True)
        pipe.ranker.save(str(model_dir / "ranker.joblib"))
        pipe.forecaster.save(str(model_dir / "ohlc.joblib"))
        pd.Series(pipe.features).to_csv(model_dir / "features.csv", index=False, header=False)

    predictions = predict_daily(
        args.prices,
        args.model_dir,
        prediction_date,
        top_k=args.top_k,
        fundamentals_path=args.fundamentals,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(predictions.to_string(index=False))
    print(f"Saved {len(predictions)} predictions for {prediction_date} to {args.output}")


if __name__ == "__main__":
    main()
