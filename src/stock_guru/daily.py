from __future__ import annotations

from pathlib import Path
import pandas as pd
from .features import build_features
from .fundamentals import load_fundamentals, asof_join
from .ranker import StockRanker
from .ohlc import OHLCForecaster


def prepare_market(prices_path: str, fundamentals_path: str | None = None) -> pd.DataFrame:
    prices = pd.read_csv(prices_path, parse_dates=["date"])
    if fundamentals_path:
        prices = asof_join(prices, load_fundamentals(fundamentals_path))
    return prices.sort_values(["date", "symbol"])


def predict_daily(prices_path: str, model_dir: str, prediction_date: str,
                  top_k: int = 10, fundamentals_path: str | None = None) -> pd.DataFrame:
    data = prepare_market(prices_path, fundamentals_path)
    feat_data, features = build_features(data)
    day = feat_data[feat_data["date"].astype(str).str[:10] == prediction_date].dropna(subset=features)
    if day.empty:
        raise ValueError(f"No usable rows for prediction date {prediction_date}")
    model_path = Path(model_dir)
    ranker = StockRanker.load(str(model_path / "ranker.joblib"))
    forecaster = OHLCForecaster.load(str(model_path / "ohlc.joblib"))
    ranked = ranker.score(day).head(top_k)
    pred = forecaster.predict(ranked)
    pred["rank"] = range(1, len(pred) + 1)
    pred.insert(0, "prediction_date", prediction_date)
    pred["model_version"] = "adaptive-v1"
    if "score" in pred:
        pred["confidence"] = pred["score"].rank(pct=True)
    return pred
