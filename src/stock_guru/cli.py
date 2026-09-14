from __future__ import annotations

import argparse
from pathlib import Path
import json
import os
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


def load_optional_csv(path: str | None) -> pd.DataFrame | None:
    return pd.read_csv(path) if path else None


def load_universe(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    df = pd.read_csv(path)
    required = {"symbol", "start_date", "end_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing universe interval columns: {sorted(missing)}")
    for col in ("start_date", "end_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    if df["start_date"].isna().any():
        raise ValueError("Universe intervals contain invalid start dates")
    if df["end_date"].notna().any() and (df["start_date"] >= df["end_date"].fillna(pd.Timestamp.max)).any():
        raise ValueError("Universe intervals must have end_date after start_date")
    return df


def save_model(pipe: Pipeline, model_dir: str, *, trained_through=None, validation_metrics=None, model_version=None, pit_context=None) -> None:
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipe.ranker.save(str(out / "ranker.joblib"))
    pipe.forecaster.save(str(out / "ohlc.joblib"))
    pd.Series(pipe.features).to_csv(out / "features.csv", index=False, header=False)
    metadata = {
        "model_version": model_version or os.environ.get("GITHUB_SHA", "local")[:12],
        "trained_through": str(pd.Timestamp(trained_through).date()) if trained_through is not None else None,
        "validation_metrics": validation_metrics or {},
        "features": list(pipe.features),
        "pit_context": pit_context or {"fundamentals_supplied": False, "universe_intervals_supplied": False},
    }
    (out / "model_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stock Guru adaptive NIFTY 500 pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download"); dl.add_argument("--start", required=True); dl.add_argument("--end"); dl.add_argument("--output", default="data/prices.csv")
    tr = sub.add_parser("train"); tr.add_argument("--prices", required=True); tr.add_argument("--fundamentals", default=None); tr.add_argument("--universe", default=None); tr.add_argument("--model-dir", default="artifacts")
    pr = sub.add_parser("predict"); pr.add_argument("--prices", required=True); pr.add_argument("--fundamentals", default=None); pr.add_argument("--universe", default=None); pr.add_argument("--model-dir", default="artifacts"); pr.add_argument("--date", required=True); pr.add_argument("--top-k", type=int, default=10); pr.add_argument("--output", default="artifacts/predictions.csv")
    fb = sub.add_parser("feedback"); fb.add_argument("--prices", required=True); fb.add_argument("--predictions", required=True); fb.add_argument("--output", default="artifacts/feedback.csv"); fb.add_argument("--metrics-output", default="artifacts/feedback_metrics.json")
    rt = sub.add_parser("retrain"); rt.add_argument("--prices", required=True); rt.add_argument("--model-dir", default="artifacts"); rt.add_argument("--min-train-days", type=int, default=252); rt.add_argument("--step-days", type=int, default=20); rt.add_argument("--top-k", type=int, default=10); rt.add_argument("--prediction-date", default=None); rt.add_argument("--feedback", default=None); rt.add_argument("--decision-output", default=None)
    bt = sub.add_parser("backtest"); bt.add_argument("--prices", required=True); bt.add_argument("--fundamentals", default=None); bt.add_argument("--universe", default=None); bt.add_argument("--output-dir", default="artifacts/backtest"); bt.add_argument("--min-train-days", type=int, default=252); bt.add_argument("--step-days", type=int, default=20); bt.add_argument("--top-k", type=int, default=10); bt.add_argument("--transaction-cost-bps", type=float, default=10.0)
    ev = sub.add_parser("evaluate"); ev.add_argument("--predictions", required=True)
    uh = sub.add_parser("universe-history", help="Build PIT universe intervals from an authoritative baseline and event ledger")
    uh.add_argument("--baseline", required=True); uh.add_argument("--events", required=True); uh.add_argument("--output", default="data/nifty500_universe_history.csv")
    uq = sub.add_parser("universe-quality", help="Report PIT universe coverage without assuming historical completeness")
    uq.add_argument("--snapshots", required=True); uq.add_argument("--events", default=None); uq.add_argument("--manifest", default=None); uq.add_argument("--output", default="artifacts/universe_quality.json")
    us = sub.add_parser("universe-source-validate", help="Validate historical universe snapshots and provenance manifest")
    us.add_argument("--snapshots", required=True); us.add_argument("--manifest", required=True); us.add_argument("--expected-constituents", type=int, default=None); us.add_argument("--require-full-snapshot-size", action="store_true"); us.add_argument("--gap-threshold-days", type=int, default=None)
    ua = sub.add_parser("universe-audit", help="Fingerprint and audit a validated historical universe source")
    ua.add_argument("--snapshots", required=True); ua.add_argument("--manifest", required=True); ua.add_argument("--expected-constituents", type=int, default=None); ua.add_argument("--require-full-snapshot-size", action="store_true"); ua.add_argument("--gap-threshold-days", type=int, default=None); ua.add_argument("--output", default="artifacts/universe_source_audit.json")
    ea = sub.add_parser("event-audit", help="Audit provenance, completeness gates, and duplicate protection for the historical event chain")
    ea.add_argument("--manifest", default="data/nifty500_event_chain_manifest.csv")
    ea.add_argument("--output", default="artifacts/event_chain_audit.json")
    args = parser.parse_args()

    if args.command == "download":
        from .data import download_nifty500_prices
        path = download_nifty500_prices(start=args.start, end=args.end, output=args.output); print(f"saved {path}")
    elif args.command == "train":
        data = load(args.prices); fundamentals = load_optional_csv(args.fundamentals); universe = load_universe(args.universe); pipe = Pipeline().train(data, fundamentals=fundamentals, universe_intervals=universe); save_model(pipe, args.model_dir, trained_through=data["date"].max(), pit_context={"fundamentals_supplied": fundamentals is not None, "universe_intervals_supplied": universe is not None}); print(f"trained; features={len(pipe.features)}")
    elif args.command == "predict":
        from .features import build_features
        from .ranker import StockRanker
        from .ohlc import OHLCForecaster
        data = load(args.prices); fundamentals = load_optional_csv(args.fundamentals); universe = load_universe(args.universe); prepared = Pipeline._prepare(data, universe); feat_data, features = build_features(prepared, fundamentals); ranker = StockRanker.load(str(Path(args.model_dir) / "ranker.joblib")); forecaster = OHLCForecaster.load(str(Path(args.model_dir) / "ohlc.joblib")); metadata_path = Path(args.model_dir) / "model_metadata.json"; model_version = "unknown" if not metadata_path.exists() else json.loads(metadata_path.read_text(encoding="utf-8")).get("model_version", "unknown"); day = feat_data[feat_data.date.astype(str).str[:10] == args.date].dropna(subset=features); ranked = ranker.score(day).head(args.top_k); pred = forecaster.predict(ranked); pred["rank"] = range(1, len(pred) + 1); pred["model_version"] = model_version; Path(args.output).parent.mkdir(parents=True, exist_ok=True); pred.to_csv(args.output, index=False); print(pred.to_string(index=False))
    elif args.command == "feedback":
        from .feedback import label_predictions, append_feedback
        from .model_selection import summarize_feedback
        predictions = pd.read_csv(args.predictions); predictions["date"] = pd.to_datetime(predictions["date"] if "date" in predictions.columns else predictions["prediction_date"]); market = load(args.prices); labeled = label_predictions(predictions, market); 
        if labeled.empty: raise RuntimeError("No next-day actuals matched the stored predictions")
        append_feedback(args.output, labeled); metrics = {"current_batch": evaluate(labeled), "cumulative": summarize_feedback(pd.read_csv(args.output), min_rows=1)}; Path(args.metrics_output).parent.mkdir(parents=True, exist_ok=True); Path(args.metrics_output).write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8"); print(metrics)
    elif args.command == "retrain":
        from .model_selection import evaluate_candidate, should_promote, save_metrics, summarize_feedback
        data = load(args.prices); model_dir = Path(args.model_dir); candidate_metrics = evaluate_candidate(data, min_train_days=args.min_train_days, step_days=args.step_days, top_k=args.top_k); feedback_summary = summarize_feedback(pd.read_csv(args.feedback)) if args.feedback and Path(args.feedback).exists() else None; metrics_path = model_dir / "walk_forward_metrics.json"; old_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None; accepted = should_promote(old_metrics, candidate_metrics, feedback=feedback_summary); decision = {"accepted": accepted, "candidate": candidate_metrics, "previous": old_metrics, "feedback": feedback_summary};
        if accepted:
            cutoff = pd.Timestamp(args.prediction_date).normalize() if args.prediction_date else pd.to_datetime(data["date"]).dt.normalize().max(); training = data[pd.to_datetime(data["date"]).dt.normalize() < cutoff].copy();
            if training.empty: raise ValueError("No historical sessions remain before the retraining prediction date")
            final_pipe = Pipeline(top_k=args.top_k).train(training); save_model(final_pipe, args.model_dir, trained_through=training["date"].max(), validation_metrics=candidate_metrics); save_metrics(args.model_dir, candidate_metrics); decision["trained_through"] = str(training["date"].max().date())
        else: decision["reason"] = "candidate rejected; existing model retained"
        if args.decision_output: output = Path(args.decision_output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
        else: print(json.dumps(decision, indent=2, default=str))
    elif args.command == "backtest":
        from .walk_forward_backtest import run_strategy_walk_forward
        from .reporting import save_backtest_report
        data = load(args.prices); fundamentals = load_optional_csv(args.fundamentals); universe = load_universe(args.universe); result = run_strategy_walk_forward(data, min_train_days=args.min_train_days, step_days=args.step_days, top_k=args.top_k, transaction_cost_bps=args.transaction_cost_bps, fundamentals=fundamentals, universe_intervals=universe); print(json.dumps({"portfolio": result["portfolio"], "files": save_backtest_report(result, args.output_dir), "pit_context": result.get("pit_context", {})}, indent=2, default=str))
    elif args.command == "evaluate": print(evaluate(pd.read_csv(args.predictions)))
    elif args.command == "event-audit":
        from .event_audit import audit_event_chain, save_event_audit
        report = audit_event_chain(args.manifest); save_event_audit(args.manifest, args.output); print(json.dumps(report, indent=2))
    elif args.command == "universe-history":
        from .universe_events import build_universe_history_from_events
        print(f"saved {build_universe_history_from_events(args.baseline, args.events, args.output)}")
    elif args.command == "universe-quality":
        from .universe_coverage import save_coverage_report
        from .universe_source import load_source_manifest
        snapshots = pd.read_csv(args.snapshots); events = pd.read_csv(args.events) if args.events else None; manifest = load_source_manifest(args.manifest) if args.manifest else None; output = save_coverage_report(snapshots, args.output, events, source_manifest=manifest); print(output.read_text(encoding="utf-8"))
    elif args.command == "universe-source-validate":
        from .universe_source import validate_source_bundle
        print(json.dumps(validate_source_bundle(args.snapshots, args.manifest, expected_constituents=args.expected_constituents, require_full_snapshot_size=args.require_full_snapshot_size, gap_threshold_days=args.gap_threshold_days), indent=2))
    elif args.command == "universe-audit":
        from .universe_audit import build_source_audit, save_source_audit
        report = build_source_audit(args.snapshots, args.manifest, expected_constituents=args.expected_constituents, require_full_snapshot_size=args.require_full_snapshot_size, gap_threshold_days=args.gap_threshold_days); output = save_source_audit(report, args.output); print(output.read_text(encoding="utf-8"))


if __name__ == "__main__": main()
