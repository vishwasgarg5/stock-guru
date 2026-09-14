import pandas as pd

from stock_guru.risk import RiskConfig, final_trade_decision


def _base(**extra):
    data = {"close": 100.0, "pred_close": 105.0, "rank_confidence": 0.9, "forecast_confidence": 0.9, "pred_close_uncertainty_pct": 0.02, "atr_pct_14": 0.02, "volatility_20": 0.03, "downside_volatility_20": 0.02, "volume_ratio_20": 1.0}
    data.update(extra)
    return data


def test_high_volatility_is_rejected():
    out = final_trade_decision(pd.DataFrame([_base(symbol="AAA", atr_pct_14=0.10)]))
    assert out.iloc[0]["decision"] == "NO_TRADE"
    assert "atr" in out.iloc[0]["risk_reason"]


def test_position_and_sector_caps():
    rows = [_base(symbol="A", sector="BANK"), _base(symbol="B", sector="BANK", pred_close=104.0), _base(symbol="C", sector="IT", pred_close=103.0)]
    out = final_trade_decision(pd.DataFrame(rows), RiskConfig(max_position_pct=0.20, max_sector_pct=0.30))
    assert (out["position_weight"] <= 0.20 + 1e-12).all()
    assert out.loc[out["sector"] == "BANK", "position_weight"].sum() <= 0.30 + 1e-12
    assert (out["decision"] == "TRADE").any()


def test_low_expected_return_is_no_trade():
    out = final_trade_decision(pd.DataFrame([_base(symbol="AAA", pred_close=100.1)]))
    assert out.iloc[0]["decision"] == "NO_TRADE"


def test_high_vol_bear_tightens_confidence_filter():
    rows = [_base(symbol="AAA", rank_confidence=0.65, atr_pct_14=0.05, volatility_20=0.04, downside_volatility_20=0.025, market_regime="high_vol_bear"), _base(symbol="BBB", rank_confidence=0.75, atr_pct_14=0.05, volatility_20=0.04, downside_volatility_20=0.025, market_regime="high_vol_bear")]
    out = final_trade_decision(pd.DataFrame(rows))
    assert out.loc[out["symbol"] == "AAA", "decision"].iloc[0] == "NO_TRADE"
    assert out.loc[out["symbol"] == "BBB", "decision"].iloc[0] == "TRADE"


def test_high_vol_bear_caps_total_exposure():
    out = final_trade_decision(pd.DataFrame([_base(symbol=s, rank_confidence=1.0, atr_pct_14=0.01, volatility_20=0.04, downside_volatility_20=0.025, market_regime="high_vol_bear") for s in "ABCDEFGH"]))
    assert out["position_weight"].sum() <= 0.50 + 1e-12
    assert (out["position_weight"] <= 0.075 + 1e-12).all()


def test_bear_regime_requires_higher_confidence_and_return():
    rows = [_base(symbol="LOW", rank_confidence=0.69, pred_close=100.5, atr_pct_14=0.04, volatility_20=0.04, downside_volatility_20=0.025, market_regime="bear"), _base(symbol="HIGH", rank_confidence=0.70, pred_close=100.5, atr_pct_14=0.04, volatility_20=0.04, downside_volatility_20=0.025, market_regime="bear")]
    out = final_trade_decision(pd.DataFrame(rows))
    assert out.loc[out["symbol"] == "LOW", "decision"].iloc[0] == "NO_TRADE"
    assert out.loc[out["symbol"] == "HIGH", "decision"].iloc[0] == "TRADE"


def test_bear_regime_exposure_is_capped():
    out = final_trade_decision(pd.DataFrame([_base(symbol=s, rank_confidence=1.0, atr_pct_14=0.01, volatility_20=0.04, downside_volatility_20=0.025, market_regime="bear") for s in "ABCDEFG"]))
    assert out["position_weight"].sum() <= 0.50 + 1e-12
    assert (out["position_weight"] <= 0.10 + 1e-12).all()


def test_excessive_downside_volatility_is_rejected():
    out = final_trade_decision(pd.DataFrame([_base(symbol="AAA", downside_volatility_20=0.06)]))
    assert out.iloc[0]["decision"] == "NO_TRADE"
    assert "downside_volatility" in out.iloc[0]["risk_reason"]


def test_thin_volume_is_rejected():
    out = final_trade_decision(pd.DataFrame([_base(symbol="AAA", volume_ratio_20=0.40)]))
    assert out.iloc[0]["decision"] == "NO_TRADE"
    assert "liquidity" in out.iloc[0]["risk_reason"]


def test_high_uncertainty_is_rejected():
    out = final_trade_decision(pd.DataFrame([_base(symbol="AAA", pred_close_uncertainty_pct=0.10)]))
    assert out.iloc[0]["decision"] == "NO_TRADE"
    assert "uncertainty" in out.iloc[0]["risk_reason"]


def test_low_forecast_confidence_is_rejected():
    out = final_trade_decision(pd.DataFrame([_base(symbol="AAA", forecast_confidence=0.40)]))
    assert out.iloc[0]["decision"] == "NO_TRADE"
    assert "forecast_confidence" in out.iloc[0]["risk_reason"]
