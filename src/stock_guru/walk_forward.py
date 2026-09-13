from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from .evaluation import ranking_metrics
from .feedback import label_predictions, score_labeled
from .features import build_features
from .pipeline import Pipeline
from .regime import confidence_from_rank, regime_label


@dataclass
class FoldResult:
    train_end: str
    prediction_date: str
    metrics: dict
    regime_metrics: dict | None = None


def run_walk_forward_with_predictions(
    raw: pd.DataFrame,
    min_train_days: int = 252,
    step_days: int = 20,
    top_k: int = 10,
) -> tuple[list[FoldResult], list[pd.DataFrame]]:
    """Run leakage-safe folds once and return metrics plus labeled predictions."""
    dates = sorted(pd.to_datetime(raw["date"]).dt.normalize().unique())
    if len(dates) <= min_train_days + 1:
        raise ValueError("Not enough dates for walk-forward validation")

    results: list[FoldResult] = []
    predictions: list[pd.DataFrame] = []
    full_features, _ = build_features(raw)
    normalized = pd.to_datetime(full_features["date"]).dt.normalize()
    raw_dates = pd.to_datetime(raw["date"]).dt.normalize()

    actuals = raw[["date", "symbol", "close"]].copy()
    actuals["date"] = pd.to_datetime(actuals["date"]).dt.normalize()
    actuals = actuals.sort_values(["symbol", "date"])
    actuals["next_close"] = actuals.groupby("symbol")["close"].shift(-1)
    actuals["base_close"] = actuals["close"]

    for idx in range(min_train_days, len(dates) - 1, step_days):
        train_end = dates[idx - 1]
        prediction_date = dates[idx]
        train = raw[raw_dates <= train_end].copy()
        pipe = Pipeline(top_k=top_k).train(train)

        day = full_features[normalized == prediction_date].copy().dropna(subset=pipe.features)
        if day.empty:
            continue

        scored = pipe.ranker.score(day)
        market_regime = regime_label(day.iloc[0])
        ranking_base = scored[["date", "symbol", "rank_score"]].copy()
        ranking_base["prediction_date"] = prediction_date
        ranking_base["market_regime"] = market_regime
        ranking_base = ranking_base.merge(
            actuals[["date", "symbol", "base_close", "next_close"]],
            on=["date", "symbol"], how="left"
        ).dropna(subset=["base_close", "next_close"])
        ranking_base["actual_close"] = ranking_base["next_close"]

        ranked = scored.head(top_k).copy()
        pred = pipe.forecaster.predict(ranked)
        if pred.empty:
            continue

        pred["rank"] = range(1, len(pred) + 1)
        pred["rank_confidence"] = confidence_from_rank(pred["rank_score"])
        pred["market_regime"] = market_regime

        # Preserve risk inputs from the prediction-time feature row. The
        # previous implementation dropped these columns when converting the
        # ranked frame into OHLC forecasts, causing every backtest trade to
        # fail closed on missing ATR/volatility/confidence fields.
        risk_columns = ["atr_pct_14", "volatility_20"]
        risk_frame = ranked[["symbol", *risk_columns]].copy()
        pred = pred.merge(risk_frame, on="symbol", how="left", validate="one_to_one")

        labeled = label_predictions(pred, raw)
        labeled = labeled[labeled["prediction_date"] == prediction_date].copy()
        if labeled.empty:
            continue

        metrics = score_labeled(labeled)
        metrics.update(ranking_metrics(ranking_base, k=top_k))
        regime_metrics = metrics.pop("regime_metrics", None)
        results.append(FoldResult(str(train_end.date()), str(prediction_date.date()), metrics, regime_metrics))
        predictions.append(labeled)

    return results, predictions


def run_walk_forward(raw: pd.DataFrame, min_train_days: int = 252, step_days: int = 20, top_k: int = 10) -> list[FoldResult]:
    """Evaluate next-session forecasts with expanding, leakage-safe training windows."""
    results, _ = run_walk_forward_with_predictions(
        raw, min_train_days=min_train_days, step_days=step_days, top_k=top_k
    )
    return results
