import pandas as pd
import pytest
from stock_guru.fundamentals import asof_join, load_fundamentals


def test_asof_join_never_uses_future_fundamentals():
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-10", "2026-02-10"]),
        "symbol": ["ABC", "ABC"], "close": [100, 110],
    })
    fundamentals = pd.DataFrame({
        "symbol": ["ABC", "ABC"],
        "available_date": pd.to_datetime(["2026-01-01", "2026-02-15"]),
        "roe": [10.0, 20.0],
    })
    out = asof_join(prices, fundamentals)
    assert out.loc[out.date == pd.Timestamp("2026-01-10"), "roe"].iloc[0] == 10.0
    assert out.loc[out.date == pd.Timestamp("2026-02-10"), "roe"].iloc[0] == 10.0


def test_load_fundamentals_rejects_duplicate_release_rows(tmp_path):
    path = tmp_path / "fundamentals.csv"
    pd.DataFrame({"symbol": ["ABC", "ABC"], "available_date": ["2026-01-01", "2026-01-01"], "roe": [10, 11]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_fundamentals(path)


def test_load_fundamentals_rejects_release_before_report(tmp_path):
    path = tmp_path / "fundamentals.csv"
    pd.DataFrame({
        "symbol": ["ABC"],
        "period_end": ["2025-12-31"],
        "reported_date": ["2026-02-15"],
        "available_date": ["2026-02-10"],
        "roe": [10],
    }).to_csv(path, index=False)
    with pytest.raises(ValueError, match="available_date earlier than reported_date"):
        load_fundamentals(path)


def test_asof_join_uses_available_date_not_period_end():
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-15", "2026-02-20"]),
        "symbol": ["ABC", "ABC"],
    })
    fundamentals = pd.DataFrame({
        "symbol": ["ABC"],
        "period_end": pd.to_datetime(["2025-12-31"]),
        "reported_date": pd.to_datetime(["2026-02-10"]),
        "available_date": pd.to_datetime(["2026-02-12"]),
        "roe": [12.0],
    })
    out = asof_join(prices, fundamentals)
    assert pd.isna(out.loc[out.date == pd.Timestamp("2026-01-15"), "roe"].iloc[0])
    assert out.loc[out.date == pd.Timestamp("2026-02-20"), "roe"].iloc[0] == 12.0


def test_load_fundamentals_rejects_blank_source_when_present(tmp_path):
    path = tmp_path / "fundamentals.csv"
    pd.DataFrame({
        "symbol": ["ABC"],
        "available_date": ["2026-02-12"],
        "source": [""],
    }).to_csv(path, index=False)
    with pytest.raises(ValueError, match="blank source"):
        load_fundamentals(path)
