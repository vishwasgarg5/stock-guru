from __future__ import annotations

import joblib
import pandas as pd
from xgboost import XGBRanker


class StockRanker:
    """Rank stocks within each date using risk-adjusted next-day return."""

    def __init__(self, params: dict | None = None):
        self.model = XGBRanker(
            objective="rank:ndcg", eval_metric="ndcg@10", n_estimators=300,
            max_depth=5, learning_rate=0.04, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=2.0, random_state=42, **(params or {}),
        )
        self.features: list[str] = []
        self.ranking_target = "risk_adjusted_return"

    @staticmethod
    def relevance(values: pd.Series) -> pd.Series:
        ranks = values.rank(method="average", ascending=True)
        if len(ranks) <= 1:
            return pd.Series(0, index=values.index, dtype=int)
        scaled = ((ranks - 1.0) * 31.0 / (len(ranks) - 1.0)).round()
        return scaled.clip(0, 31).astype(int)

    @staticmethod
    def build_ranking_target(df: pd.DataFrame) -> pd.Series:
        """Return per unit of current downside risk, with a safe risk floor."""
        if "target_return" not in df.columns:
            raise ValueError("target_return is required for ranking")
        risk_col = "downside_volatility_20" if "downside_volatility_20" in df.columns else "volatility_20"
        if risk_col not in df.columns:
            return df["target_return"].astype(float)
        risk = df[risk_col].astype(float).clip(lower=0.005)
        return df["target_return"].astype(float) / risk

    def fit(self, df: pd.DataFrame, features: list[str]) -> "StockRanker":
        required = features + ["target_return"]
        train = df.dropna(subset=required).sort_values(["date", "symbol"])
        ranking_target = self.build_ranking_target(train)
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
