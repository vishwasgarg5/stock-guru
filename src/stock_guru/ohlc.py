from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit
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
        self.regime_adjustment_shrinkage = 0.5
        self.regime_models: dict[str, dict[str, XGBRegressor]] = {}
        self.regime_blend = {"bear": 0.25, "high_vol_bear": 0.25}

    def _fit_regime_adjustments(self, train: pd.DataFrame) -> None:
        """Learn conservative regime corrections from time-series OOF residuals."""
        labels = train.apply(regime_label, axis=1)
        train = train.copy().sort_values("date") if "date" in train else train.copy()
        residuals = pd.DataFrame(index=train.index, columns=TARGETS, dtype=float)
        n_splits = min(3, max(0, len(train) // 20))
        if n_splits >= 2:
            splitter = TimeSeriesSplit(n_splits=n_splits)
            for fit_idx, valid_idx in splitter.split(train):
                fit = train.iloc[fit_idx]
                valid = train.iloc[valid_idx]
                for target, model in self.models.items():
                    oof_model = clone(model)
                    oof_model.fit(fit[self.features], fit[target])
                    residuals.loc[valid.index, target] = valid[target].to_numpy() - oof_model.predict(valid[self.features])

        adjustments: dict[str, dict[str, float]] = {}
        for label in ("bear", "high_vol_bear"):
            mask = labels.reindex(train.index).eq(label) & residuals.notna().all(axis=1)
            if mask.sum() < 10:
                continue
            adjustments[label] = {
                target: float(residuals.loc[mask, target].median() * self.regime_adjustment_shrinkage)
                for target in TARGETS
            }
        self.regime_adjustments = adjustments

    def _fit_regime_models(self, train: pd.DataFrame) -> None:
        """Fit small specialist models only on adverse-regime training rows."""
        self.regime_models = {}
        labels = train.apply(regime_label, axis=1)
        for label in ("bear", "high_vol_bear"):
            mask = labels.eq(label)
            if mask.sum() < 30:
                continue
            specialists: dict[str, XGBRegressor] = {}
            for target, base_model in self.models.items():
                specialist = clone(base_model)
                specialist.set_params(n_estimators=max(100, int(base_model.get_params()["n_estimators"] * 0.5)))
                specialist.fit(train.loc[mask, self.features], train.loc[mask, target])
                specialists[target] = specialist
            self.regime_models[label] = specialists

    def fit(self, df: pd.DataFrame, features: list[str]) -> "OHLCForecaster":
        self.features = features
        train = df.dropna(subset=features + TARGETS).copy()
        weights = pd.Series(1.0, index=train.index)
        for target, model in self.models.items():
            model.fit(train[features], train[target], sample_weight=weights)
        self._fit_regime_adjustments(train)
        self._fit_regime_models(train)
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
        labels = df.apply(regime_label, axis=1)
        for target, model in self.models.items():
            base_pred = model.predict(df[self.features])
            pred = base_pred.copy()
            target_col = target.replace("target_", "pred_")
            for label, specialists in self.regime_models.items():
                mask = labels.eq(label)
                if mask.any():
                    blend = self.regime_blend.get(label, 0.0)
                    specialist_pred = specialists[target].predict(df.loc[mask, self.features])
                    pred[mask.to_numpy()] = (1.0 - blend) * pred[mask.to_numpy()] + blend * specialist_pred
            out[target_col] = pred

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
