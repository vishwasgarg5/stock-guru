from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRanker


class StockRanker:
    """Ranks stocks within each date using risk-adjusted next-day return."""

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
        self.ranking_target = "risk_adjusted_return"

    @staticmethod
    def relevance(values: pd.Series) -> pd.Series:
        """Map each cross-sectional rank to the valid 0..31 NDCG range."""
        ranks = values.rank(method="average", ascending=True)
        if len(ranks) <= 1:
            return pd.Series(0, index=values.index, dtype=int)
        scaled = ((ranks - 1.0) * 31.0 / (len(ranks) - 1.0)).round()
        return scaled.clip(0, 31).astype(int)

    @staticmethod
    def build_ranking_target(df: pd.DataFrame) -> pd.Series:
        """Prefer return per unit of current risk, with a safe volatility floor."""
        if "target_return" not in df.columns:
            raise ValueError("target_return is required for ranking")
        if "downside_volatility_20" in df.columns:
            risk = df["downside_volatility_20"].clip(lower=0.005)
        elif "volatility_20" in df.columns:
            risk = df["volatility_20"].clip(lower=0.005)
        else:
            return df["target_return"].astype(float)
        return df["target_return"].astype(float) / risk

    def fit(self, df: pd.DataFrame, features: list[str]) -> "StockRanker":
        required = features + ["target_return"]
        train = df.dropna(subset=required).sort_values(["date", "symbol"])
        ranking_target = self.build_ranking_target(train)
        y = train.groupby("date")[ranking_target.name if ranking_target.name else "target_return"].transform(self.relevance) if ranking_target.name in train else self.relevance(ranking_target)
        # Compute relevance independently inside each daily query group.
        y = ranking_target.groupby(train["date"]).transform(self.relevance).astype(int)
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
