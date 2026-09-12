from __future__ import annotations

from pathlib import Path
import pandas as pd
from .features import build_features
from .fundamentals import load_fundamentals, asof_join
from .pipeline import Pipeline


def prepare_market(prices_path: str, fundamentals_path: str | None = None) -> pd.DataFrame:
    prices = pd.read_csv(prices_path, parse_dates=["date"])
    if fundamentals_path:
        fundamentals = load_fundamentals(fundamentals_path)
        prices = asof_join(prices, fundamentals)
    return prices.sort_values(["date", "symbol"])


def predict_daily(prices_path: str, model_dir: str, prediction_date: str,
                  top_k: int = 10, fundamentals_path: str | None = None) -> pd.DataFrame:
    data = prepare_market(prices_path, fundamentals_path)
    feat_data, features = build_features(data)
    day = feat_data[feat_data["date"].dt.strftime("%Y-%m-%d") == prediction_date].dropna(subset=features)
    if day.empty:
        raise ValueError(f"No usable rows for prediction date {prediction_date}")
    pipe = Pipeline.load(model_dir, features=features, top_k=top_k)
    pred = pipe.predict(day)
    pred.insert(0, "prediction_date", prediction_date)
    pred["model_version"] = Path(model_dir).name
    pred["confidence"] = pred.get("score", pd.Series(index=pred.index, dtype=float)).rank(pct=True)
    return pred
