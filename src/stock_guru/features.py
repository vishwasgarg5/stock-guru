from __future__ import annotations

import numpy as np
import pandas as pd

from .regime import REGIME_FEATURES, add_market_regime_features

BASE_FUNDAMENTALS = [
    "roe", "roce", "eps_growth", "revenue_growth", "pe", "pb",
    "debt_to_equity", "operating_margin", "free_cash_flow",
]


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe features using information available through each row's date."""
    out = df.sort_values(["symbol", "date"]).copy()
    g = out.groupby("symbol", group_keys=False)
    close = g["close"]
    out["ret_1d"] = close.pct_change()
    out["ret_5d"] = close.pct_change(5)
    out["ret_20d"] = close.pct_change(20)
    out["sma_10"] = close.transform(lambda s: s.rolling(10).mean())
    out["sma_20"] = close.transform(lambda s: s.rolling(20).mean())
    out["sma_50"] = close.transform(lambda s: s.rolling(50).mean())
    out["ema_20"] = close.transform(lambda s: s.ewm(span=20, adjust=False).mean())
    out["ema_50"] = close.transform(lambda s: s.ewm(span=50, adjust=False).mean())
    delta = g["close"].diff()
    gain = delta.clip(lower=0).groupby(out["symbol"]).transform(lambda s: s.rolling(14).mean())
    loss = (-delta.clip(upper=0)).groupby(out["symbol"]).transform(lambda s: s.rolling(14).mean())
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))
    out["volatility_20"] = g["close"].pct_change().groupby(out["symbol"]).transform(lambda s: s.rolling(20).std())
    out["volume_ratio_20"] = out["volume"] / out.groupby("symbol")["volume"].transform(lambda s: s.rolling(20).mean())
    out["atr_pct_14"] = ((out["high"] - out["low"]) / out["close"]).groupby(out["symbol"]).transform(lambda s: s.rolling(14).mean())
    return out


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create next-day normalized OHLC targets from today's close."""
    out = df.sort_values(["symbol", "date"]).copy()
    g = out.groupby("symbol")
    next_open = g["open"].shift(-1)
    next_high = g["high"].shift(-1)
    next_low = g["low"].shift(-1)
    next_close = g["close"].shift(-1)
    base = out["close"]
    out["target_open"] = next_open / base - 1
    out["target_high"] = next_high / base - 1
    out["target_low"] = next_low / base - 1
    out["target_close"] = next_close / base - 1
    out["target_return"] = out["target_close"]
    return out


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = add_targets(add_technical_features(df.copy()))
    out = add_market_regime_features(out)
    fundamentals = [c for c in BASE_FUNDAMENTALS if c in out.columns]
    technical = [
        "ret_1d", "ret_5d", "ret_20d", "sma_10", "sma_20", "sma_50",
        "ema_20", "ema_50", "rsi_14", "volatility_20", "volume_ratio_20", "atr_pct_14",
    ]
    for c in ["sma_10", "sma_20", "sma_50", "ema_20", "ema_50"]:
        out[f"{c}_ratio"] = out[c] / out["close"] - 1
    technical = [c for c in technical if c not in {"sma_10", "sma_20", "sma_50", "ema_20", "ema_50"}]
    technical += ["sma_10_ratio", "sma_20_ratio", "sma_50_ratio", "ema_20_ratio", "ema_50_ratio"]
    features = fundamentals + technical + REGIME_FEATURES
    out[features] = out[features].replace([np.inf, -np.inf], np.nan)
    return out, features
