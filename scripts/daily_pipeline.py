from __future__ import annotations

import argparse
from pathlib import Path
from datetime import date, timedelta

from stock_guru.data import download_nifty500_prices
from stock_guru.daily import predict_daily


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

    # Refresh the market data before prediction. The downloader also persists the
    # current NIFTY 500 membership snapshot for later point-in-time universe work.
    yesterday = date.today() - timedelta(days=1)
    download_nifty500_prices(args.start, yesterday.isoformat(), args.prices)

    prediction_date = args.prediction_date or yesterday.isoformat()
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
    print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
