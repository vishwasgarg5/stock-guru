from __future__ import annotations

import argparse
from pathlib import Path
import json
import pandas as pd
from .pipeline import Pipeline
from .evaluation import evaluate


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df.sort_values(["date", "symbol"])


def save_model(pipe: Pipeline, model_dir: str) -> None:
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipe.ranker.save(str(out / "ranker.joblib"))
    pipe.forecaster.save(str(out / "ohlc.joblib"))
    pd.Series(pipe.features).to_csv(out / "features.csv", index=False, header=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock Guru adaptive NIFTY 500 pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download"); dl.add_argument("--start", required=True); dl.add_argument("--end"); dl.add_argument("--output", default="data/prices.csv")
    tr = sub.add_parser("train"); tr.add_argument("--prices", required=True); tr.add_argument("--model-dir", default="artifacts")
    pr = sub.add_parser("predict"); pr.add_argument("--prices", required=True); pr.add_argument("--model-dir", default="artifacts"); pr.add_argument("--date", required=True); pr.add_argument("--top-k", type=int, default=10); pr.add_argument("--output", default="artifacts/predictions.csv")
    fb = sub.add_parser("feedback"); fb.add_argument("--prices", required=True); fb.add_argument("--predictions", required=True); fb.add_argument("--output", default="artifacts/feedback.csv")
    rt = sub.add_parser("retrain"); rt.add_argument("--prices", required=True); rt.add_argument("--model-dir", default="artifacts"); rt.add_argument("--min-train-days", type=int, default=252); rt.add_argument("--step-days", type=int, default=20); rt.add_argument("--top-k", type=int, default=10)
    ev = sub.add_parser("evaluate"); ev.add_argument("--predictions", required=True)
    args = parser.parse_args()

    if args.command == "download":
        from .data import download_nifty500_prices
        path = download_nifty500_prices(start=args.start, end=args.end, output=args.output)
        print(f"saved {path}")
    elif args.command == "train":
        pipe = Pipeline().train(load(args.prices))
        save_model(pipe, args.model_dir)
        print(f"trained; features={len(pipe.features)}")
    elif args.command == "predict":
        data = load(args.prices)
        from .features import build_features
        from .ranker import StockRanker
        from .ohlc import OHLCForecaster
        feat_data, features = build_features(data)
        ranker = StockRanker.load(str(Path(args.model_dir) / "ranker.joblib"))
        forecaster = OHLCForecaster.load(str(Path(args.model_dir) / "ohlc.joblib"))
        day = feat_data[feat_data.date.astype(str).str[:10] == args.date].dropna(subset=features)
        ranked = ranker.score(day).head(args.top_k)
        pred = forecaster.predict(ranked)
        pred["rank"] = range(1, len(pred) + 1)
        pred["model_version"] = "adaptive-v1"
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(args.output, index=False)
        print(pred.to_string(index=False))
    elif args.command == "feedback":
        from .feedback import label_predictions, append_feedback
        predictions = pd.read_csv(args.predictions, parse_dates=["date"])
        labeled = label_predictions(predictions, load(args.prices))
        if labeled.empty:
            raise RuntimeError("No next-day actuals matched the stored predictions")
        append_feedback(args.output, labeled)
        print(evaluate(labeled))
    elif args.command == "retrain":
        from .model_selection import evaluate_candidate, should_promote, save_metrics
        data = load(args.prices)
        model_dir = Path(args.model_dir)
        candidate_metrics = evaluate_candidate(
            data,
            min_train_days=args.min_train_days,
            step_days=args.step_days,
            top_k=args.top_k,
        )
        metrics_path = model_dir / "walk_forward_metrics.json"
        old_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None
        accepted = should_promote(old_metrics, candidate_metrics)
        print({"accepted": accepted, "candidate": candidate_metrics, "previous": old_metrics})
        if accepted:
            # Validation decides whether to promote; the promoted model is then
            # retrained on all currently available history.
            final_pipe = Pipeline(top_k=args.top_k).train(data)
            save_model(final_pipe, args.model_dir)
            save_metrics(args.model_dir, candidate_metrics)
        else:
            print("candidate rejected; existing model retained")
    elif args.command == "evaluate":
        print(evaluate(pd.read_csv(args.predictions)))


if __name__ == "__main__":
    main()
