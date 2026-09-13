from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .regime import regime_label

TARGETS = ["target_open", "target_high", "target_low", "target_close"]


class OHLCForecaster:
    def __init__(self, params: dict | None = None):
        base = dict(
            objective="reg:squarederror", n_estimators=400, max_depth=5,
            learning_rate=0.03, subsample=0.8, colsample_bytree=0.85,
            reg_lambda=2.0, random_state=42,
        )
        if params:
            base.update(params)
        self.models = {target: XGBRegressor(**base) for target in TARGETS}
        self.features: list[str] = []
        self.regime_adjustments: dict[str, dict[str, float]] = {}

    def _fit_regime_adjustments(self, train: pd.DataFrame) -> None:
        """Learn small additive residual corrections from training data only."""
        labels = train.apply(regime_label, axis=1)
        residuals = pd.DataFrame(index=train.index)
        for target, model in self.models.items():
            residuals[target] = train[target] - model.predict(train[self.features])

        adjustments: dict[str, dict[str, float]] = {}
        for label in ("bear", "high_vol_bear"):
            mask = labels.eq(label)
            if mask.sum() < 10:
                continue
            adjustments[label] = {
                target: float(residuals.loc[mask, target].median())
                for target in TARGETS
            }
        self.regime_adjustments = adjustments

    def fit(self, df: pd.DataFrame, features: list[str]) -> "OHLCForecaster":
        self.features = features
        train = df.dropna(subset=features + TARGETS).copy()
        weights = pd.Series(1.0, index=train.index)
        for target, model in self.models.items():
            model.fit(train[features], train[target], sample_weight=weights)
        self._fit_regime_adjustments(train)
        return self

    @staticmethod
    def enforce_ohlc_constraints(out: pd.DataFrame) -> pd.DataFrame:
        """Ensure predicted OHLC forms a physically valid daily candle."""
        out = out.copy()
        price_cols = ["pred_open", "pred_high", "pred_low", "pred_close"]
        for col in price_cols:
            out[col] = out[col].clip(lower=0.0)
        out["pred_high"] = out[["pred_open", "pred_high", "pred_close"]].max(axis=1)
        out["pred_low"] = out[["pred_open", "pred_low", "pred_close"]].min(axis=1)
        return out

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        metadata = [c for c in ["date", "symbol", "close", "rank_score"] if c in df.columns]
        out = df[metadata].copy()
        for target, model in self.models.items():
            out[target.replace("target_", "pred_")] = model.predict(df[self.features])

        if self.regime_adjustments:
            labels = df.apply(regime_label, axis=1)
            for label, corrections in self.regime_adjustments.items():
                mask = labels.eq(label)
                for target, correction in corrections.items():
                    out.loc[mask, target.replace("target_", "pred_")] += correction

        base = out["close"]
        out["pred_open"] = base * (1 + out["pred_open"])
        out["pred_high"] = base * (1 + out["pred_high"])
        out["pred_low"] = base * (1 + out["pred_low"])
        out["pred_close"] = base * (1 + out["pred_close"])
        return self.enforce_ohlc_constraints(out)

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "OHLCForecaster":
        return joblib.load(path)
