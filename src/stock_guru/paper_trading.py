from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class PaperConfig:
    initial_capital: float = 1_000_000.0
    max_position_pct: float = 0.10
    commission_bps: float = 10.0
    slippage_bps: float = 5.0


def _load(path: str | Path, required: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df


def execute_signals(signals: pd.DataFrame, prices: pd.DataFrame, capital: float,
                    config: PaperConfig | None = None) -> pd.DataFrame:
    """Simulate next-session entries using predicted signals and actual opens.

    Signals must contain prediction_date, symbol, and a positive position_weight.
    Prices must contain date, symbol and open. Allocation is capped per position.
    """
    cfg = config or PaperConfig()
    required = {"prediction_date", "symbol", "position_weight"}
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"Missing signal columns: {sorted(missing)}")
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    s = signals.copy()
    s["prediction_date"] = pd.to_datetime(s["prediction_date"]).dt.normalize()
    s["symbol"] = s["symbol"].astype(str).str.strip()
    p["symbol"] = p["symbol"].astype(str).str.strip()
    next_prices = p[["date", "symbol", "open"]].sort_values(["symbol", "date"])
    s = s.sort_values(["symbol", "prediction_date"])
    next_prices["entry_date"] = next_prices["date"]
    merged = pd.merge_asof(
        s.sort_values(["symbol", "prediction_date"]),
        next_prices.sort_values(["symbol", "entry_date"]),
        left_on="prediction_date", right_on="entry_date", by="symbol",
        direction="forward", allow_exact_matches=False,
    )
    merged["allocation_pct"] = merged["position_weight"].clip(lower=0, upper=cfg.max_position_pct)
    merged["gross_allocation"] = capital * merged["allocation_pct"]
    cost_rate = (cfg.commission_bps + cfg.slippage_bps) / 10_000.0
    merged["entry_price"] = merged["open"] * (1.0 + cfg.slippage_bps / 10_000.0)
    merged["quantity"] = (merged["gross_allocation"] / merged["entry_price"]).fillna(0).astype(int)
    merged["entry_value"] = merged["quantity"] * merged["entry_price"]
    merged["entry_cost"] = merged["entry_value"] * cost_rate
    return merged.dropna(subset=["entry_date", "entry_price"])


def mark_to_market(trades: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "entry_date", "quantity", "entry_value", "entry_cost"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"Missing trade columns: {sorted(missing)}")
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    p = p[["date", "symbol", "close"]].copy()
    t = trades.copy()
    t["symbol"] = t["symbol"].astype(str).str.strip()
    t["entry_date"] = pd.to_datetime(t["entry_date"]).dt.normalize()
    t = t.merge(p, left_on=["symbol", "entry_date"], right_on=["symbol", "date"], how="left")
    t["exit_price"] = t["close"]
    t["exit_value"] = t["quantity"] * t["exit_price"]
    t["exit_cost"] = t["exit_value"] * (trades.attrs.get("exit_cost_rate", 0.0))
    t["net_pnl"] = t["exit_value"] - t["entry_value"] - t["entry_cost"] - t["exit_cost"]
    return t.drop(columns=["date"], errors="ignore")
