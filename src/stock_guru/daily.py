from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
from .features import build_features
from .fundamentals import load_fundamentals, asof_join
from .ranker import StockRanker
from .ohlc import OHLCForecaster
from .regime import add_market_regime_features, confidence_from_rank, regime_label
from .risk import RiskConfig, final_trade_decision


def prepare_market(prices_path: str, fundamentals_path: str | None = None) -> pd.DataFrame:
    prices = pd.read_csv(prices_path, parse_dates=["date"])
    if fundamentals_path:
        prices = asof_join(prices, load_fundamentals(fundamentals_path))
    return prices.sort_values(["date", "symbol"])


def predict_daily(prices_path: str, model_dir: str, prediction_date: str,
                  top_k: int = 10, fundamentals_path: str | None = None,
                  risk_config: RiskConfig | None = None) -> pd.DataFrame:
    data = prepare_market(prices_path, fundamentals_path)
    feat_data, features = build_features(data)
    feat_data = add_market_regime_features(feat_data)
    prediction_date = pd.Timestamp(prediction_date).normalize()
    day = feat_data[feat_data["date"].dt.normalize() == prediction_date].dropna(subset=features)
    if day.empty:
        raise ValueError(f"No usable rows for prediction date {prediction_date.date()}")

    model_path = Path(model_dir)
    ranker = StockRanker.load(str(model_path / "ranker.joblib"))
    forecaster = OHLCForecaster.load(str(model_path / "ohlc.joblib"))
    model_version = "unknown"
    metadata_path = model_path / "model_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        model_version = str(metadata.get("model_version") or model_version)

    ranked = ranker.score(day).head(top_k)
    pred = forecaster.predict(ranked)
    pred["rank"] = range(1, len(pred) + 1)
    pred["rank_confidence"] = confidence_from_rank(pred["rank_score"])

    regime_columns = ["market_ret_20d", "market_volatility_20", "market_breadth"]
    risk_columns = ["symbol", "atr_pct_14", "volatility_20", "sector", *regime_columns]
    available = [c for c in risk_columns if c in ranked.columns]
    pred = pred.merge(ranked[available].drop_duplicates("symbol"), on="symbol", how="left")
    pred.insert(0, "prediction_date", prediction_date.date().isoformat())

    if all(c in pred.columns for c in regime_columns):
        pred["market_regime"] = pred.apply(regime_label, axis=1)
    else:
        pred["market_regime"] = "unknown"

    # The forecast is for the next trading session after prediction_date.
    pred["forecast_horizon"] = "next_session"
    pred["model_version"] = model_version
    return final_trade_decision(pred, risk_config)
