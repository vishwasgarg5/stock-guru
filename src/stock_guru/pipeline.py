from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from .features import build_features
from .ranker import StockRanker
from .ohlc import OHLCForecaster
from .regime import add_market_regime_features, confidence_from_rank, regime_label


@dataclass
class Pipeline:
    top_k: int = 10
    ranker: StockRanker | None = None
    forecaster: OHLCForecaster | None = None
    features: list[str] | None = None

    def train(self, raw: pd.DataFrame) -> "Pipeline":
        data, features = build_features(raw)
        self.features = features
        self.ranker = StockRanker().fit(data, features)
        self.forecaster = OHLCForecaster().fit(data, features)
        return self

    def predict_date(self, raw: pd.DataFrame, date: str) -> pd.DataFrame:
        if not self.ranker or not self.forecaster or self.features is None:
            raise RuntimeError("Train the pipeline before prediction.")
        data, _ = build_features(raw)
        data = add_market_regime_features(data)
        day = data[data["date"].astype(str) == str(date)].copy()
        day = day.dropna(subset=self.features)
        ranked = self.ranker.score(day).head(self.top_k)
        pred = self.forecaster.predict(ranked)
        pred["rank"] = range(1, len(pred) + 1)
        pred["rank_confidence"] = confidence_from_rank(pred["rank_score"])
        if not day.empty:
            regime = regime_label(day.iloc[0])
            pred["market_regime"] = regime
        else:
            pred["market_regime"] = "unknown"
        pred["model_version"] = "adaptive-v1"
        return pred
