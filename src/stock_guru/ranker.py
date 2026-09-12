from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRanker

class StockRanker:
    """Ranks stocks within each date using future next-day return as relevance."""
    def __init__(self, params: dict | None = None):
        self.model = XGBRanker(
            objective="rank:ndcg",
            eval_metric="ndcg@10",
            n_estimators=300,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            random_state=42,
            **(params or {}),
        )
        self.features: list[str] = []

    @staticmethod
    def relevance(values: pd.Series) -> pd.Series:
        # Cross-sectional percentile is stable across different price scales.
        return values.rank(pct=True, method="average")

    def fit(self, df: pd.DataFrame, features: list[str]) -> "StockRanker":
        train = df.dropna(subset=features + ["target_return"]).sort_values(["date", "symbol"])
        y = train.groupby("date")["target_return"].transform(self.relevance).astype(float)
        qid = train["date"].factorize(sort=True)[0]
        self.features = features
        self.model.fit(train[features], y, qid=qid)
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        x["rank_score"] = self.model.predict(x[self.features])
        return x.sort_values(["date", "rank_score"], ascending=[True, False])

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "StockRanker":
        return joblib.load(path)
