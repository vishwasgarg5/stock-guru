from __future__ import annotations

import numpy as np
import pandas as pd


REGIME_FEATURES = [
    "market_ret_1d",
    "market_ret_5d",
    "market_ret_20d",
    "market_volatility_20",
    "market_breadth",
    "market_above_sma20",
]


def add_market_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add equal-weight market-regime features known at prediction time."""
    out = df.sort_values(["symbol", "date"]).copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()

    # Build an equal-weight market return series from stock returns rather than
    # averaging raw share prices, which would overweight high-priced stocks.
    out["_stock_ret_1d"] = out.groupby("symbol")["close"].pct_change()
    daily = out.groupby("date")[["_stock_ret_1d"]].mean().rename(columns={"_stock_ret_1d": "market_ret_1d"})
    daily["market_ret_5d"] = (1.0 + daily["market_ret_1d"]).rolling(5).apply(np.prod, raw=True) - 1.0
    daily["market_ret_20d"] = (1.0 + daily["market_ret_1d"]).rolling(20).apply(np.prod, raw=True) - 1.0
    daily["market_volatility_20"] = daily["market_ret_1d"].rolling(20).std()

    # Breadth uses today's close relative to each stock's own 20-session SMA.
    out["_sma20"] = out.groupby("symbol")["close"].transform(lambda s: s.rolling(20).mean())
    valid = out["_sma20"].notna()
    breadth = out.loc[valid].assign(_above=out.loc[valid, "close"] > out.loc[valid, "_sma20"]).groupby("date")["_above"].mean()
    daily["market_breadth"] = breadth
    daily["market_above_sma20"] = daily["market_breadth"]

    result = out.merge(daily[REGIME_FEATURES], left_on="date", right_index=True, how="left")
    return result.drop(columns=["_stock_ret_1d", "_sma20"])


def confidence_from_rank(scores: pd.Series) -> pd.Series:
    """Convert within-day rank scores to a relative 0-1 confidence proxy.

    This is not a probability calibration; it measures relative conviction
    among the stocks scored on the same day.
    """
    if scores.empty:
        return scores.astype(float)
    ranks = scores.rank(method="average", pct=True)
    return ranks


def regime_label(row: pd.Series) -> str:
    """Simple deterministic regime label based only on known market features."""
    ret20 = row.get("market_ret_20d", np.nan)
    vol = row.get("market_volatility_20", np.nan)
    breadth = row.get("market_breadth", np.nan)
    if pd.isna(ret20) or pd.isna(vol) or pd.isna(breadth):
        return "unknown"
    if vol >= 0.02 and ret20 < 0:
        return "high_vol_bear"
    if ret20 >= 0.03 and breadth >= 0.60:
        return "bull"
    if ret20 <= -0.03 and breadth < 0.45:
        return "bear"
    if vol >= 0.02:
        return "high_volatility"
    return "neutral"
