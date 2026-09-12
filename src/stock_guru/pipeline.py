from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from .features import build_features
from .ranker import StockRanker
from .ohlc import OHLCForecaster

@dataclass
class Pipeline:
    top_k: int = 10
    ranker: StockRanker | None = None
    forecaster: OHLCForecaster | None = None
    features: list[str] | None = None

    def train(self, raw: pd.DataFrame) -> "Pipeline":
        data, features = build_features(raw)
        # Require at least a full target row; ranking learns relative next-day return.
        self.features = features
        self.ranker = StockRanker().fit(data, features)
        self.forecaster = OHLCForecaster().fit(data, features)
        return self

    def predict_date(self, raw: pd.DataFrame, date: str) -> pd.DataFrame:
        if not self.ranker or not self.forecaster or self.features is None:
            raise RuntimeError("Train the pipeline before prediction.")
        data, _ = build_features(raw)
        day = data[data["date"].astype(str) == str(date)].copy()
        day = day.dropna(subset=self.features)
        ranked = self.ranker.score(day).head(self.top_k)
        pred = self.forecaster.predict(ranked)
        pred["rank"] = range(1, len(pred) + 1)
        pred["model_version"] = "initial-v1"
        return pred
