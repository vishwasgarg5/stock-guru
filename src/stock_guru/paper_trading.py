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


def execute_signals(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    capital: float | None = None,
    config: PaperConfig | None = None,
) -> pd.DataFrame:
    """Execute accepted signals at the next session open and close that day.

    The engine is cash-aware: each prediction date shares a common cash budget,
    rejects allocations that exceed available cash, and records end-of-day equity.
    """
    cfg = config or PaperConfig()
    starting_cash = cfg.initial_capital if capital is None else float(capital)
    if starting_cash <= 0:
        raise ValueError("capital must be positive")
    if not 0 < cfg.max_position_pct <= 1:
        raise ValueError("max_position_pct must be in (0, 1]")
    if cfg.commission_bps < 0 or cfg.slippage_bps < 0:
        raise ValueError("cost assumptions cannot be negative")

    required = {"prediction_date", "symbol"}
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"Missing signal columns: {sorted(missing)}")
    weight_col = "position_weight" if "position_weight" in signals.columns else "allocation_pct"
    if weight_col not in signals.columns:
        raise ValueError("Signals must contain position_weight or allocation_pct")

    p = prices.copy()
    missing = {"date", "symbol", "open", "close"} - set(p.columns)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    p["date"] = pd.to_datetime(p["date"], errors="coerce").dt.normalize()
    p["symbol"] = p["symbol"].astype(str).str.strip()
    if p["date"].isna().any() or p["symbol"].eq("").any():
        raise ValueError("Prices contain invalid dates or blank symbols")
    if p.duplicated(["date", "symbol"]).any():
        raise ValueError("Prices contain duplicate date/symbol rows")
    for col in ["open", "close"]:
        p[col] = pd.to_numeric(p[col], errors="coerce")
    if (p[["open", "close"]] <= 0).any().any():
        raise ValueError("Prices must be positive")

    s = signals.copy()
    s["prediction_date"] = pd.to_datetime(s["prediction_date"], errors="coerce").dt.normalize()
    s["symbol"] = s["symbol"].astype(str).str.strip()
    if s["prediction_date"].isna().any() or s["symbol"].eq("").any():
        raise ValueError("Signals contain invalid dates or blank symbols")
    s[weight_col] = pd.to_numeric(s[weight_col], errors="coerce")
    if s[weight_col].isna().any() or (s[weight_col] < 0).any():
        raise ValueError("Signal weights must be non-negative numbers")

    future = p[["date", "symbol", "open", "close"]].rename(columns={"date": "entry_date"})
    out = pd.merge_asof(
        s.sort_values(["symbol", "prediction_date"]),
        future.sort_values(["symbol", "entry_date"]),
        left_on="prediction_date", right_on="entry_date", by="symbol",
        direction="forward", allow_exact_matches=False,
    )
    out["requested_allocation_pct"] = out[weight_col].clip(lower=0, upper=cfg.max_position_pct)
    out["allocation_pct"] = 0.0
    out["allocated_capital"] = 0.0
    out["entry_price"] = out["open"] * (1 + cfg.slippage_bps / 10_000)
    out["exit_price"] = out["close"] * (1 - cfg.slippage_bps / 10_000)
    out["quantity"] = 0
    out["entry_value"] = 0.0
    out["entry_cost"] = 0.0
    out["exit_value"] = 0.0
    out["exit_cost"] = 0.0
    out["net_pnl"] = 0.0
    out["cash_after"] = starting_cash
    out["equity_after"] = starting_cash

    cash = starting_cash
    for prediction_date, idx in out.groupby("prediction_date", sort=True).groups.items():
        day = out.loc[idx].copy()
        # Budget is shared across all signals generated for the same prediction date.
        requested = (day["requested_allocation_pct"] * cash).clip(lower=0)
        total_requested = float(requested.sum())
        scale = min(1.0, cash / total_requested) if total_requested > 0 else 0.0
        allocated = requested * scale
        for row_idx, budget in allocated.items():
            entry = float(out.at[row_idx, "entry_price"])
            qty = int(budget // entry) if pd.notna(entry) and entry > 0 else 0
            entry_value = qty * entry
            entry_cost = entry_value * cfg.commission_bps / 10_000
            exit_price = float(out.at[row_idx, "exit_price"])
            exit_value = qty * exit_price
            exit_cost = exit_value * cfg.commission_bps / 10_000
            total_debit = entry_value + entry_cost
            if total_debit > cash and qty > 0:
                qty = int(cash // (entry * (1 + cfg.commission_bps / 10_000)))
                entry_value = qty * entry
                entry_cost = entry_value * cfg.commission_bps / 10_000
                exit_value = qty * exit_price
                exit_cost = exit_value * cfg.commission_bps / 10_000
                total_debit = entry_value + entry_cost
            cash -= total_debit
            net_pnl = exit_value - entry_value - entry_cost - exit_cost
            cash += exit_value - exit_cost
            out.at[row_idx, "allocation_pct"] = budget / starting_cash
            out.at[row_idx, "allocated_capital"] = budget
            out.at[row_idx, "quantity"] = qty
            out.at[row_idx, "entry_value"] = entry_value
            out.at[row_idx, "entry_cost"] = entry_cost
            out.at[row_idx, "exit_value"] = exit_value
            out.at[row_idx, "exit_cost"] = exit_cost
            out.at[row_idx, "net_pnl"] = net_pnl
        out.loc[idx, "cash_after"] = cash
        out.loc[idx, "equity_after"] = cash

    out["return_on_allocated"] = out["net_pnl"] / out["entry_value"].replace(0, pd.NA)
    return out.dropna(subset=["entry_date", "entry_price", "exit_price"])


def save_trades(trades: pd.DataFrame, path: str | Path) -> None:
    """Persist paper trades idempotently by prediction date/symbol/model."""
    required = {"prediction_date", "symbol"}
    if not required.issubset(trades.columns):
        raise ValueError(f"Missing trade columns: {sorted(required - set(trades.columns))}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = trades.copy()
    if path.exists():
        incoming = pd.concat([pd.read_csv(path), incoming], ignore_index=True, sort=False)
    keys = [c for c in ["prediction_date", "symbol", "model_version"] if c in incoming.columns]
    incoming.drop_duplicates(keys or ["prediction_date", "symbol"], keep="last").to_csv(path, index=False)
