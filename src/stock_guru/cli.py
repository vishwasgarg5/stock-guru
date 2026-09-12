from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from .pipeline import Pipeline
from .evaluation import evaluate, attach_actuals


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df.sort_values(["date", "symbol"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock Guru adaptive NIFTY 500 baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train"); tr.add_argument("--prices", required=True); tr.add_argument("--model-dir", default="artifacts")
    pr = sub.add_parser("predict"); pr.add_argument("--prices", required=True); pr.add_argument("--model-dir", default="artifacts"); pr.add_argument("--date", required=True)
    ev = sub.add_parser("evaluate"); ev.add_argument("--predictions", required=True)
    args = parser.parse_args()

    if args.command == "train":
        data = load(args.prices)
        pipe = Pipeline().train(data)
        out = Path(args.model_dir); out.mkdir(parents=True, exist_ok=True)
        pipe.ranker.save(str(out / "ranker.joblib"))
        pipe.forecaster.save(str(out / "ohlc.joblib"))
        pd.Series(pipe.features).to_csv(out / "features.csv", index=False, header=False)
        print(f"trained on {len(data):,} rows; features={len(pipe.features)}")
    elif args.command == "predict":
        # Baseline loader; model objects can be loaded directly with joblib in a production service.
        import joblib
        data = load(args.prices)
        from .features import build_features
        from .ranker import StockRanker
        from .ohlc import OHLCForecaster
        feat_data, features = build_features(data)
        ranker = StockRanker.load(str(Path(args.model_dir) / "ranker.joblib"))
        forecaster = OHLCForecaster.load(str(Path(args.model_dir) / "ohlc.joblib"))
        day = feat_data[feat_data.date.astype(str) == args.date].dropna(subset=features)
        ranked = ranker.score(day).head(10)
        pred = forecaster.predict(ranked)
        pred["rank"] = range(1, len(pred) + 1)
        print(pred.to_string(index=False))
    elif args.command == "evaluate":
        p = pd.read_csv(args.predictions)
        print(evaluate(p))

if __name__ == "__main__":
    main()
