from __future__ import annotations

from dataclasses import dataclass
import math
import pandas as pd


@dataclass(frozen=True)
class RiskConfig:
    max_position_pct: float = 0.15
    max_sector_pct: float = 0.30
    target_risk_pct: float = 0.01
    min_confidence: float = 0.60
    max_atr_pct: float = 0.08
    max_volatility_20: float = 0.06
    min_expected_return: float = 0.002
    min_price: float = 20.0


def _safe_float(value, default=float("nan")) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def apply_risk_filters(predictions: pd.DataFrame, config: RiskConfig | None = None) -> pd.DataFrame:
    """Filter predictions using volatility, ATR, confidence and return thresholds."""
    cfg = config or RiskConfig()
    out = predictions.copy()
    out["expected_return"] = out["pred_close"] / out["close"] - 1

    checks = pd.DataFrame(index=out.index)
    checks["price_ok"] = out["close"] >= cfg.min_price
    checks["confidence_ok"] = out.get("rank_confidence", pd.Series(0.0, index=out.index)) >= cfg.min_confidence
    checks["atr_ok"] = out.get("atr_pct_14", pd.Series(float("nan"), index=out.index)) <= cfg.max_atr_pct
    checks["volatility_ok"] = out.get("volatility_20", pd.Series(float("nan"), index=out.index)) <= cfg.max_volatility_20
    checks["return_ok"] = out["expected_return"] >= cfg.min_expected_return

    # Missing risk inputs fail closed rather than silently bypassing protection.
    for col in ["atr_pct_14", "volatility_20"]:
        if col not in out:
            checks[col.replace("_14", "").replace("volatility_20", "volatility") + "_ok"] = False

    out["risk_pass"] = checks.all(axis=1)
    out["risk_reason"] = checks.apply(
        lambda row: "approved" if row.all() else ";".join(name.replace("_ok", "") for name, passed in row.items() if not passed),
        axis=1,
    )
    return out


def size_positions(predictions: pd.DataFrame, config: RiskConfig | None = None) -> pd.DataFrame:
    """Assign volatility-adjusted weights, then enforce per-position and sector caps."""
    cfg = config or RiskConfig()
    out = predictions.copy()
    out["position_weight"] = 0.0
    eligible = out["risk_pass"].fillna(False)
    if not eligible.any():
        return out

    risk = out.loc[eligible, "atr_pct_14"].clip(lower=0.005)
    confidence = out.loc[eligible, "rank_confidence"].clip(lower=0.0, upper=1.0)
    raw = cfg.target_risk_pct / risk * (0.5 + 0.5 * confidence)
    raw = raw.clip(upper=cfg.max_position_pct)
    total = raw.sum()
    if total > 1.0:
        raw = raw / total
    out.loc[eligible, "position_weight"] = raw

    if "sector" in out.columns:
        for sector, idx in out.loc[eligible].groupby("sector").groups.items():
            sector_total = out.loc[idx, "position_weight"].sum()
            if sector_total > cfg.max_sector_pct:
                out.loc[idx, "position_weight"] *= cfg.max_sector_pct / sector_total

    out["position_weight_pct"] = out["position_weight"] * 100.0
    return out


def final_trade_decision(predictions: pd.DataFrame, config: RiskConfig | None = None) -> pd.DataFrame:
    """Produce final trade/no-trade decisions after all risk controls."""
    cfg = config or RiskConfig()
    out = apply_risk_filters(predictions, cfg)
    out = size_positions(out, cfg)
    out["trade"] = out["risk_pass"] & (out["position_weight"] > 0)
    out["decision"] = out["trade"].map({True: "TRADE", False: "NO_TRADE"})
    return out
