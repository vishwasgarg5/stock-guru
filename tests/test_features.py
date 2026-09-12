import numpy as np
import pandas as pd

from stock_guru.features import build_features


def test_feature_pipeline_is_grouped_by_symbol():
    dates = pd.date_range("2024-01-01", periods=70, freq="D")
    rows = []
    for symbol, base in [("AAA", 100.0), ("BBB", 200.0)]:
        for i, date in enumerate(dates):
            close = base + i
            rows.append({
                "date": date,
                "symbol": symbol,
                "open": close - 1,
                "high": close + 2,
                "low": close - 2,
                "close": close,
                "volume": 1000 + i,
            })
    df = pd.DataFrame(rows)
    out, features = build_features(df)
    assert features
    assert len(out) == 140
    assert np.isfinite(out["ret_5d"].dropna()).all()
    assert out["target_close"].notna().sum() == 138
