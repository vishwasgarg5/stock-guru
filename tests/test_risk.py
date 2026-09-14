import pandas as pd

from stock_guru.risk import RiskConfig, final_trade_decision


def test_high_volatility_is_rejected():
    df = pd.DataFrame({
        "symbol": ["AAA"], "close": [100.0], "pred_close": [105.0],
        "rank_confidence": [0.9], "atr_pct_14": [0.10], "volatility_20": [0.03],
        "downside_volatility_20": [0.02],
    })
    out = final_trade_decision(df)
    assert out.iloc[0]["decision"] == "NO_TRADE"
    assert "atr" in out.iloc[0]["risk_reason"]


def test_position_and_sector_caps():
    df = pd.DataFrame({
        "symbol": ["A", "B", "C"], "sector": ["BANK", "BANK", "IT"],
        "close": [100.0, 100.0, 100.0], "pred_close": [105.0, 104.0, 103.0],
        "rank_confidence": [1.0, 0.9, 0.8], "atr_pct_14": [0.01, 0.01, 0.01],
        "volatility_20": [0.02, 0.02, 0.02], "downside_volatility_20": [0.015, 0.015, 0.015],
    })
    cfg = RiskConfig(max_position_pct=0.20, max_sector_pct=0.30)
    out = final_trade_decision(df, cfg)
    assert (out["position_weight"] <= 0.20 + 1e-12).all()
    assert out.loc[out["sector"] == "BANK", "position_weight"].sum() <= 0.30 + 1e-12
    assert (out["decision"] == "TRADE").any()


def test_low_expected_return_is_no_trade():
    df = pd.DataFrame({
        "symbol": ["AAA"], "close": [100.0], "pred_close": [100.1],
        "rank_confidence": [0.9], "atr_pct_14": [0.02], "volatility_20": [0.02],
        "downside_volatility_20": [0.015],
    })
    out = final_trade_decision(df)
    assert out.iloc[0]["decision"] == "NO_TRADE"


def test_high_vol_bear_tightens_confidence_filter():
    df = pd.DataFrame({
        "symbol": ["AAA", "BBB"], "close": [100.0, 100.0], "pred_close": [105.0, 105.0],
        "rank_confidence": [0.65, 0.75], "atr_pct_14": [0.05, 0.05],
        "volatility_20": [0.04, 0.04], "downside_volatility_20": [0.025, 0.025],
        "market_regime": ["high_vol_bear", "high_vol_bear"],
    })
    out = final_trade_decision(df)
    assert out.loc[out["symbol"] == "AAA", "decision"].iloc[0] == "NO_TRADE"
    assert out.loc[out["symbol"] == "BBB", "decision"].iloc[0] == "TRADE"


def test_high_vol_bear_caps_total_exposure():
    df = pd.DataFrame({
        "symbol": list("ABCDEFGH"), "close": [100.0] * 8, "pred_close": [105.0] * 8,
        "rank_confidence": [1.0] * 8, "atr_pct_14": [0.01] * 8,
        "volatility_20": [0.04] * 8, "downside_volatility_20": [0.025] * 8,
        "market_regime": ["high_vol_bear"] * 8,
    })
    out = final_trade_decision(df)
    assert out["position_weight"].sum() <= 0.50 + 1e-12
    assert (out["position_weight"] <= 0.075 + 1e-12).all()


def test_bear_regime_requires_higher_confidence_and_return():
    df = pd.DataFrame({
        "symbol": ["LOW", "HIGH"], "close": [100.0, 100.0], "pred_close": [100.3, 100.5],
        "rank_confidence": [0.69, 0.70], "atr_pct_14": [0.04, 0.04],
        "volatility_20": [0.04, 0.04], "downside_volatility_20": [0.025, 0.025],
        "market_regime": ["bear", "bear"],
    })
    out = final_trade_decision(df)
    assert out.loc[out["symbol"] == "LOW", "decision"].iloc[0] == "NO_TRADE"
    assert out.loc[out["symbol"] == "HIGH", "decision"].iloc[0] == "TRADE"


def test_bear_regime_exposure_is_capped():
    df = pd.DataFrame({
        "symbol": list("ABCDEFG"), "close": [100.0] * 7, "pred_close": [105.0] * 7,
        "rank_confidence": [1.0] * 7, "atr_pct_14": [0.01] * 7,
        "volatility_20": [0.04] * 7, "downside_volatility_20": [0.025] * 7,
        "market_regime": ["bear"] * 7,
    })
    out = final_trade_decision(df)
    assert out["position_weight"].sum() <= 0.50 + 1e-12
    assert (out["position_weight"] <= 0.10 + 1e-12).all()


def test_excessive_downside_volatility_is_rejected():
    df = pd.DataFrame({
        "symbol": ["AAA"], "close": [100.0], "pred_close": [105.0],
        "rank_confidence": [0.9], "atr_pct_14": [0.02], "volatility_20": [0.03],
        "downside_volatility_20": [0.06],
    })
    out = final_trade_decision(df)
    assert out.iloc[0]["decision"] == "NO_TRADE"
    assert "downside_volatility" in out.iloc[0]["risk_reason"]
