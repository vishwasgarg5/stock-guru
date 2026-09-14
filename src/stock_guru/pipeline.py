from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from .features import build_features
from .ranker import StockRanker
from .ohlc import OHLCForecaster
from .regime import confidence_from_rank, regime_label
from .universe_history import apply_point_in_time_universe


@dataclass
class Pipeline:
    top_k: int = 10
    ranker: StockRanker | None = None
    forecaster: OHLCForecaster | None = None
    features: list[str] | None = None

    @staticmethod
    def _prepare(
        raw: pd.DataFrame,
        universe_intervals: pd.DataFrame | None,
    ) -> pd.DataFrame:
        if universe_intervals is None:
            return raw.copy()
        return apply_point_in_time_universe(raw, universe_intervals)

    def train(
        self,
        raw: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        universe_intervals: pd.DataFrame | None = None,
    ) -> "Pipeline":
        prepared = self._prepare(raw, universe_intervals)
        data, features = build_features(prepared, fundamentals)
        self.features = features
        self.ranker = StockRanker().fit(data, features)
        self.forecaster = OHLCForecaster().fit(data, features)
        return self

    def predict_date(
        self,
        raw: pd.DataFrame,
        date: str,
        fundamentals: pd.DataFrame | None = None,
        universe_intervals: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if not self.ranker or not self.forecaster or self.features is None:
            raise RuntimeError("Train the pipeline before prediction.")
        prepared = self._prepare(raw, universe_intervals)
        data, _ = build_features(prepared, fundamentals)
        day = data[data["date"].astype(str) == str(date)].copy().dropna(subset=self.features)
        ranked = self.ranker.score(day).head(self.top_k)
        pred = self.forecaster.predict(ranked)
        pred["rank"] = range(1, len(pred) + 1)
        pred["rank_confidence"] = confidence_from_rank(pred["rank_score"])
        pred["market_regime"] = regime_label(day.iloc[0]) if not day.empty else "unknown"
        risk_columns = ["atr_pct_14", "volatility_20", "downside_volatility_20", "volume_ratio_20"]
        risk_frame = ranked[["symbol", *risk_columns]].copy()
        pred = pred.merge(risk_frame, on="symbol", how="left", validate="one_to_one")
        pred["model_version"] = "adaptive-v2"
        return pred
