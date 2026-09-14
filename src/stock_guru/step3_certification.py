from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import pandas as pd

from .backtest import backtest, cost_sensitivity
from .fundamentals_ingest import validate_pit_fundamentals
from .pit_quality import validate_pit_join_output
from .walk_forward import run_walk_forward_with_predictions

REQUIRED_PRICE_COLUMNS = {"date", "symbol", "close"}
REQUIRED_UNIVERSE_COLUMNS = {"symbol", "start_date", "end_date"}


def _load_csv(path: str | Path, required: set[str], label: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{label} file not found: {p}")
    frame = pd.read_csv(p)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")
    return frame


def validate_step3_inputs(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    universe_intervals: pd.DataFrame,
) -> dict[str, Any]:
    """Validate the real-data contract required before Step 3 may run."""
    missing = REQUIRED_PRICE_COLUMNS - set(prices.columns)
    if missing:
        raise ValueError(f"Prices missing columns: {sorted(missing)}")
    if "source" not in fundamentals.columns or "source_id" not in fundamentals.columns:
        raise ValueError("Certification fundamentals require source and source_id provenance")
    if "available_timestamp" not in fundamentals.columns:
        raise ValueError("Certification fundamentals require available_timestamp")
    missing = REQUIRED_UNIVERSE_COLUMNS - set(universe_intervals.columns)
    if missing:
        raise ValueError(f"Universe intervals missing columns: {sorted(missing)}")

    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.normalize()
    if p["date"].isna().any() or p["symbol"].astype(str).str.strip().eq("").any():
        raise ValueError("Prices contain invalid dates or symbols")
    if p.duplicated(["date", "symbol"]).any():
        raise ValueError("Prices contain duplicate date/symbol rows")

    f = validate_pit_fundamentals(fundamentals)
    timestamps = pd.to_datetime(f["available_timestamp"], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise ValueError("Fundamentals contain invalid availability timestamps")

    u = universe_intervals.copy()
    for column in ("start_date", "end_date"):
        u[column] = pd.to_datetime(u[column], errors="coerce").dt.normalize()
    if u[["start_date", "end_date"]].isna().any().any():
        raise ValueError("Universe intervals contain invalid dates")
    if (u["end_date"] < u["start_date"]).any():
        raise ValueError("Universe intervals contain end dates before start dates")
    if u.duplicated(["symbol", "start_date", "end_date"]).any():
        raise ValueError("Universe intervals contain duplicate intervals")

    return {
        "status": "validated",
        "price_rows": int(len(p)),
        "price_symbols": int(p["symbol"].nunique()),
        "price_start": p["date"].min().date().isoformat(),
        "price_end": p["date"].max().date().isoformat(),
        "fundamental_rows": int(len(f)),
        "fundamental_symbols": int(f["symbol"].nunique()),
        "fundamental_start": f["available_date"].min().date().isoformat(),
        "fundamental_end": f["available_date"].max().date().isoformat(),
        "universe_intervals": int(len(u)),
        "universe_symbols": int(u["symbol"].nunique()),
    }


def run_step3_certification(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    universe_intervals: pd.DataFrame,
    *,
    min_train_days: int = 252,
    step_days: int = 20,
    top_k: int = 10,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
    cost_scenarios_bps: tuple[float, ...] = (0.0, 10.0, 25.0, 50.0),
) -> dict[str, Any]:
    """Run the real PIT Step 3 evaluation, or fail closed before modeling.

    This function never creates substitute history. It requires real prices,
    PIT fundamentals with publication timestamps, and historical universe
    intervals. If those inputs are absent or invalid, the returned status is
    ``blocked`` and no performance claim is produced.
    """
    validation = validate_step3_inputs(prices, fundamentals, universe_intervals)
    folds, prediction_frames = run_walk_forward_with_predictions(
        prices,
        min_train_days=min_train_days,
        step_days=step_days,
        top_k=top_k,
        fundamentals=fundamentals,
        universe_intervals=universe_intervals,
    )
    if not folds or not prediction_frames:
        return {
            "status": "blocked",
            "reason": "No out-of-sample folds produced from the supplied real PIT data",
            "input_validation": validation,
            "folds": 0,
            "portfolio": None,
            "cost_sensitivity": {},
        }

    predictions = pd.concat(prediction_frames, ignore_index=True)
    join_audit = validate_pit_join_output(predictions, date_column="prediction_date")
    portfolio = backtest(
        predictions,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
    sensitivity = cost_sensitivity(
        predictions,
        cost_scenarios_bps=cost_scenarios_bps,
        slippage_bps=slippage_bps,
    )
    return {
        "status": "complete",
        "input_validation": validation,
        "pit_join_audit": join_audit,
        "folds": len(folds),
        "walk_forward": [
            {
                "train_end": fold.train_end,
                "prediction_date": fold.prediction_date,
                "metrics": fold.metrics,
                "regime_metrics": fold.regime_metrics,
            }
            for fold in folds
        ],
        "portfolio": portfolio,
        "cost_sensitivity": sensitivity,
    }


def run_step3_from_files(
    prices_path: str | Path,
    fundamentals_path: str | Path,
    universe_intervals_path: str | Path,
    output_path: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load certification inputs from files and persist the Step 3 evidence report."""
    try:
        prices = _load_csv(prices_path, REQUIRED_PRICE_COLUMNS, "prices")
        fundamentals = _load_csv(fundamentals_path, {"symbol", "reported_date", "available_date", "available_timestamp", "source", "source_id"}, "fundamentals")
        universe = _load_csv(universe_intervals_path, REQUIRED_UNIVERSE_COLUMNS, "universe intervals")
        report = run_step3_certification(prices, fundamentals, universe, **kwargs)
    except (FileNotFoundError, ValueError) as exc:
        report = {
            "status": "blocked",
            "reason": str(exc),
            "input_validation": None,
            "folds": 0,
            "portfolio": None,
            "cost_sensitivity": {},
        }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return report
