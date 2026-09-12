from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def save_backtest_report(result: dict, output_dir: str = "artifacts/backtest") -> dict:
    """Persist portfolio metrics, fold metrics, and equity/trade data."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    portfolio = result.get("portfolio", {})
    (out / "metrics.json").write_text(json.dumps(portfolio, indent=2, default=str), encoding="utf-8")

    folds = result.get("fold_results", [])
    if folds:
        pd.DataFrame([{
            "train_end": f.train_end,
            "prediction_date": f.prediction_date,
            **f.metrics,
        } for f in folds]).to_csv(out / "fold_metrics.csv", index=False)

    predictions = result.get("predictions")
    if isinstance(predictions, pd.DataFrame) and not predictions.empty:
        predictions.to_csv(out / "trades.csv", index=False)

    return {
        "metrics": str(out / "metrics.json"),
        "fold_metrics": str(out / "fold_metrics.csv"),
        "trades": str(out / "trades.csv"),
    }
