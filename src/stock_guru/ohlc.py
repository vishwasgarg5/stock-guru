from __future__ import annotations

import joblib
import pandas as pd
from xgboost import XGBRegressor

TARGETS = ["target_open", "target_high", "target_low", "target_close"]

class OHLCForecaster:
    def __init__(self, params: dict | None = None):
        base = dict(
            objective="reg:squarederror", n_estimators=400, max_depth=5,
            learning_rate=0.03, subsample=0.8, colsample_bytree=0.85,
            reg_lambda=2.0, random_state=42,
        )
        if params: base.update(params)
        self.models = {target: XGBRegressor(**base) for target in TARGETS}
        self.features: list[str] = []

    def fit(self, df: pd.DataFrame, features: list[str]) -> "OHLCForecaster":
        self.features = features
        train = df.dropna(subset=features + TARGETS)
        for target, model in self.models.items():
            model.fit(train[features], train[target])
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[["date", "symbol", "close"]].copy()
        for target, model in self.models.items():
            out[target.replace("target_", "pred_")] = model.predict(df[self.features])
        base = out["close"]
        out["pred_open"] = base * (1 + out["pred_open"])
        out["pred_high"] = base * (1 + out["pred_high"])
        out["pred_low"] = base * (1 + out["pred_low"])
        out["pred_close"] = base * (1 + out["pred_close"])
        return out

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "OHLCForecaster":
        return joblib.load(path)
