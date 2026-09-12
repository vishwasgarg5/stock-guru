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
    """Add cross-sectional market-regime features known at prediction time."""
    out = df.sort_values(["date", "symbol"]).copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()

    market = out.groupby("date").agg(
        market_close=("close", "mean"),
        market_sma20=("close", lambda s: s.mean()),
    ).sort_index()
    market["market_ret_1d"] = market["market_close"].pct_change()
    market["market_ret_5d"] = market["market_close"].pct_change(5)
    market["market_ret_20d"] = market["market_close"].pct_change(20)
    market["market_volatility_20"] = market["market_ret_1d"].rolling(20).std()

    # Breadth uses today's already-known close relative to the 20-session SMA.
    tmp = out.copy()
    tmp["sma20"] = tmp.groupby("symbol")["close"].transform(lambda s: s.rolling(20).mean())
    breadth = tmp.groupby("date")["sma20"].agg(
        market_breadth=lambda s: float(np.mean(tmp.loc[s.index, "close"] > s))
    )
    market = market.join(breadth)
    market["market_above_sma20"] = market["market_breadth"]

    cols = REGIME_FEATURES
    return out.merge(market[cols], left_on="date", right_index=True, how="left")


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
