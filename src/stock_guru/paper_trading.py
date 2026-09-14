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


def execute_signals(signals: pd.DataFrame, prices: pd.DataFrame, capital: float,
                    config: PaperConfig | None = None) -> pd.DataFrame:
    """Execute accepted signals at the next session open and close them that day."""
    cfg = config or PaperConfig()
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
    p["date"] = pd.to_datetime(p["date"]).dt.normalize()
    p["symbol"] = p["symbol"].astype(str).str.strip()
    s = signals.copy()
    s["prediction_date"] = pd.to_datetime(s["prediction_date"]).dt.normalize()
    s["symbol"] = s["symbol"].astype(str).str.strip()
    future = p[["date", "symbol", "open", "close"]].rename(columns={"date": "entry_date"})
    out = pd.merge_asof(s.sort_values(["symbol", "prediction_date"]),
                        future.sort_values(["symbol", "entry_date"]),
                        left_on="prediction_date", right_on="entry_date", by="symbol",
                        direction="forward", allow_exact_matches=False)
    out["allocation_pct"] = pd.to_numeric(out[weight_col], errors="coerce").clip(lower=0, upper=cfg.max_position_pct)
    out["allocated_capital"] = capital * out["allocation_pct"]
    out["entry_price"] = out["open"] * (1 + cfg.slippage_bps / 10_000)
    out["quantity"] = (out["allocated_capital"] / out["entry_price"]).fillna(0).astype(int)
    out["entry_value"] = out["quantity"] * out["entry_price"]
    out["entry_cost"] = out["entry_value"] * cfg.commission_bps / 10_000
    out["exit_price"] = out["close"] * (1 - cfg.slippage_bps / 10_000)
    out["exit_value"] = out["quantity"] * out["exit_price"]
    out["exit_cost"] = out["exit_value"] * cfg.commission_bps / 10_000
    out["net_pnl"] = out["exit_value"] - out["entry_value"] - out["entry_cost"] - out["exit_cost"]
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
