from __future__ import annotations

from dataclasses import dataclass, replace
import math
import pandas as pd


@dataclass(frozen=True)
class RiskConfig:
    max_position_pct: float = 0.15
    max_sector_pct: float = 0.30
    max_total_exposure_pct: float = 1.00
    target_risk_pct: float = 0.01
    min_confidence: float = 0.60
    max_atr_pct: float = 0.08
    max_volatility_20: float = 0.06
    max_downside_volatility_20: float = 0.045
    min_volume_ratio_20: float = 0.50
    min_expected_return: float = 0.002
    min_price: float = 20.0


def _safe_float(value, default=float("nan")) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _regime_config(config: RiskConfig, regime: str) -> RiskConfig:
    """Tighten portfolio risk limits for adverse market regimes."""
    adjustments = {
        "high_vol_bear": dict(max_position_pct=0.075, max_sector_pct=0.20, max_total_exposure_pct=0.50, min_confidence=0.70, max_atr_pct=0.06, max_volatility_20=0.045, max_downside_volatility_20=0.032, min_volume_ratio_20=0.75, min_expected_return=0.004),
        "high_volatility": dict(max_position_pct=0.10, max_sector_pct=0.25, max_total_exposure_pct=0.70, min_confidence=0.65, max_atr_pct=0.07, max_volatility_20=0.05, max_downside_volatility_20=0.038, min_volume_ratio_20=0.65, min_expected_return=0.003),
        "bear": dict(max_position_pct=0.10, max_sector_pct=0.25, max_total_exposure_pct=0.50, min_confidence=0.70, max_atr_pct=0.07, max_volatility_20=0.05, max_downside_volatility_20=0.035, min_volume_ratio_20=0.65, min_expected_return=0.004),
        "bull": dict(max_total_exposure_pct=1.00), "neutral": dict(), "unknown": dict(),
    }
    return replace(config, **adjustments.get(str(regime), {}))


def _effective_configs(predictions: pd.DataFrame, config: RiskConfig) -> pd.Series:
    regimes = predictions.get("market_regime", pd.Series("unknown", index=predictions.index))
    return regimes.fillna("unknown").map(lambda regime: _regime_config(config, str(regime)))


def apply_risk_filters(predictions: pd.DataFrame, config: RiskConfig | None = None) -> pd.DataFrame:
    """Filter predictions using regime-aware volatility, liquidity and return thresholds."""
    cfg = config or RiskConfig()
    out = predictions.copy()
    out["expected_return"] = out["pred_close"] / out["close"] - 1
    effective = _effective_configs(out, cfg)
    checks = pd.DataFrame(index=out.index)
    checks["price_ok"] = out["close"] >= cfg.min_price
    checks["confidence_ok"] = [_safe_float(out.get("rank_confidence", pd.Series(0.0, index=out.index)).loc[idx], 0.0) >= effective.loc[idx].min_confidence for idx in out.index]
    checks["atr_ok"] = [_safe_float(out.get("atr_pct_14", pd.Series(float("nan"), index=out.index)).loc[idx]) <= effective.loc[idx].max_atr_pct for idx in out.index]
    checks["volatility_ok"] = [_safe_float(out.get("volatility_20", pd.Series(float("nan"), index=out.index)).loc[idx]) <= effective.loc[idx].max_volatility_20 for idx in out.index]
    checks["downside_volatility_ok"] = [_safe_float(out.get("downside_volatility_20", pd.Series(float("nan"), index=out.index)).loc[idx]) <= effective.loc[idx].max_downside_volatility_20 for idx in out.index]
    checks["liquidity_ok"] = [_safe_float(out.get("volume_ratio_20", pd.Series(float("nan"), index=out.index)).loc[idx]) >= effective.loc[idx].min_volume_ratio_20 for idx in out.index]
    checks["return_ok"] = out["expected_return"] >= effective.map(lambda c: c.min_expected_return)
    for col, check_name in [("atr_pct_14", "atr_ok"), ("volatility_20", "volatility_ok"), ("downside_volatility_20", "downside_volatility_ok"), ("volume_ratio_20", "liquidity_ok")]:
        if col not in out:
            checks[check_name] = False
    out["risk_pass"] = checks.all(axis=1)
    out["risk_reason"] = checks.apply(lambda row: "approved" if row.all() else ";".join(name.replace("_ok", "") for name, passed in row.items() if not passed), axis=1)
    out["risk_regime"] = out.get("market_regime", "unknown")
    return out


def size_positions(predictions: pd.DataFrame, config: RiskConfig | None = None) -> pd.DataFrame:
    """Assign volatility-adjusted weights, then enforce regime, position and sector caps."""
    cfg = config or RiskConfig()
    out = predictions.copy()
    out["position_weight"] = 0.0
    eligible = out["risk_pass"].fillna(False)
    if not eligible.any():
        out["position_weight_pct"] = 0.0
        return out
    effective = _effective_configs(out, cfg)
    risk = out.loc[eligible, "atr_pct_14"].clip(lower=0.005)
    confidence = out.loc[eligible, "rank_confidence"].clip(lower=0.0, upper=1.0)
    raw = cfg.target_risk_pct / risk * (0.5 + 0.5 * confidence)
    per_position_cap = pd.Series([effective.loc[idx].max_position_pct for idx in raw.index], index=raw.index, dtype=float)
    raw = pd.Series([min(value, cap) for value, cap in zip(raw, per_position_cap)], index=raw.index)
    total_cap = min(effective.loc[idx].max_total_exposure_pct for idx in raw.index)
    total = raw.sum()
    if total > total_cap:
        raw = raw * (total_cap / total)
    out.loc[eligible, "position_weight"] = raw
    if "sector" in out.columns:
        for sector, idx in out.loc[eligible].groupby("sector").groups.items():
            sector_total = out.loc[idx, "position_weight"].sum()
            sector_cap = min(effective.loc[i].max_sector_pct for i in idx)
            if sector_total > sector_cap:
                out.loc[idx, "position_weight"] *= sector_cap / sector_total
    out["position_weight_pct"] = out["position_weight"] * 100.0
    return out


def final_trade_decision(predictions: pd.DataFrame, config: RiskConfig | None = None) -> pd.DataFrame:
    cfg = config or RiskConfig()
    out = apply_risk_filters(predictions, cfg)
    out = size_positions(out, cfg)
    out["trade"] = out["risk_pass"] & (out["position_weight"] > 0)
    out["decision"] = out["trade"].map({True: "TRADE", False: "NO_TRADE"})
    return out
