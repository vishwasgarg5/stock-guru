from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def _aggregate_fold_regimes(folds) -> dict:
    """Aggregate regime diagnostics across folds, weighted by forecast samples."""
    buckets: dict[str, list[dict]] = {}
    for fold in folds:
        for regime, metrics in (getattr(fold, "regime_metrics", None) or {}).items():
            buckets.setdefault(str(regime), []).append(metrics)

    result = {}
    for regime, rows in buckets.items():
        sample_total = sum(int(row.get("samples", 0)) for row in rows)
        if sample_total <= 0:
            continue
        aggregated = {"samples": sample_total}
        keys = {key for row in rows for key in row if key != "samples"}
        for key in sorted(keys):
            values = [(float(row[key]), int(row.get("samples", 0))) for row in rows if key in row]
            if values:
                denominator = sum(weight for _, weight in values)
                aggregated[key] = sum(value * weight for value, weight in values) / denominator if denominator else 0.0
        result[regime] = aggregated
    return result


def _forecast_confidence_metrics(predictions: pd.DataFrame) -> dict:
    """Summarize the confidence/uncertainty distribution without treating confidence as probability."""
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        return {}
    cols = {"forecast_confidence", "pred_close_uncertainty_pct"}
    if not cols.intersection(predictions.columns):
        return {}
    result = {"samples": int(len(predictions))}
    if "forecast_confidence" in predictions:
        values = pd.to_numeric(predictions["forecast_confidence"], errors="coerce").dropna()
        if not values.empty:
            result["mean"] = float(values.mean())
            result["median"] = float(values.median())
            result["p10"] = float(values.quantile(0.10))
            result["p90"] = float(values.quantile(0.90))
    if "pred_close_uncertainty_pct" in predictions:
        values = pd.to_numeric(predictions["pred_close_uncertainty_pct"], errors="coerce").dropna()
        if not values.empty:
            result["uncertainty_median"] = float(values.median())
            result["uncertainty_p90"] = float(values.quantile(0.90))
    return result


def save_backtest_report(result: dict, output_dir: str = "artifacts/backtest") -> dict:
    """Persist portfolio metrics, cost sensitivity, fold metrics, and forecast diagnostics."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    portfolio = result.get("portfolio", {})
    (out / "metrics.json").write_text(json.dumps(portfolio, indent=2, default=str), encoding="utf-8")

    cost_sensitivity = result.get("cost_sensitivity")
    if cost_sensitivity:
        (out / "cost_sensitivity.json").write_text(json.dumps(cost_sensitivity, indent=2, default=str), encoding="utf-8")

    folds = result.get("fold_results", [])
    regime_metrics = _aggregate_fold_regimes(folds)
    (out / "regime_metrics.json").write_text(json.dumps(regime_metrics, indent=2), encoding="utf-8")

    predictions = result.get("predictions")
    confidence_metrics = _forecast_confidence_metrics(predictions)
    (out / "forecast_confidence.json").write_text(json.dumps(confidence_metrics, indent=2), encoding="utf-8")

    if folds:
        pd.DataFrame([{
            "train_end": f.train_end,
            "prediction_date": f.prediction_date,
            **f.metrics,
        } for f in folds]).to_csv(out / "fold_metrics.csv", index=False)

    if isinstance(predictions, pd.DataFrame) and not predictions.empty:
        predictions.to_csv(out / "trades.csv", index=False)

    return {
        "metrics": str(out / "metrics.json"),
        "cost_sensitivity": str(out / "cost_sensitivity.json") if cost_sensitivity else None,
        "regime_metrics": str(out / "regime_metrics.json"),
        "forecast_confidence": str(out / "forecast_confidence.json"),
        "fold_metrics": str(out / "fold_metrics.csv"),
        "trades": str(out / "trades.csv"),
    }
