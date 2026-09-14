import pandas as pd
import pytest

from stock_guru.step3_certification import run_step3_certification, run_step3_from_files, validate_step3_inputs


def _inputs():
    prices = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=3),
        "symbol": ["TCS", "TCS", "TCS"],
        "close": [100.0, 101.0, 102.0],
    })
    fundamentals = pd.DataFrame({
        "symbol": ["TCS"],
        "reported_date": ["2025-12-31"],
        "available_date": ["2026-01-01"],
        "available_timestamp": ["2026-01-01T09:00:00Z"],
        "source": ["NSE"],
        "source_id": ["filing-1"],
        "version": ["original"],
        "eps": [3.0],
    })
    universe = pd.DataFrame({
        "symbol": ["TCS"],
        "start_date": ["2025-01-01"],
        "end_date": ["2026-12-31"],
    })
    return prices, fundamentals, universe


def test_step3_input_contract_accepts_real_pit_shape():
    prices, fundamentals, universe = _inputs()
    report = validate_step3_inputs(prices, fundamentals, universe)
    assert report["status"] == "validated"
    assert report["fundamental_rows"] == 1


def test_step3_requires_publication_timestamp():
    prices, fundamentals, universe = _inputs()
    fundamentals = fundamentals.drop(columns=["available_timestamp"])
    with pytest.raises(ValueError, match="available_timestamp"):
        validate_step3_inputs(prices, fundamentals, universe)


def test_step3_rejects_duplicate_price_rows():
    prices, fundamentals, universe = _inputs()
    prices = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_step3_inputs(prices, fundamentals, universe)


def test_step3_file_runner_blocks_missing_real_history(tmp_path):
    report = run_step3_from_files(
        tmp_path / "prices.csv",
        tmp_path / "fundamentals.csv",
        tmp_path / "universe.csv",
    )
    assert report["status"] == "blocked"
    assert report["folds"] == 0
    assert report["portfolio"] is None


def test_step3_runner_does_not_claim_complete_without_folds():
    prices, fundamentals, universe = _inputs()
    report = run_step3_certification(
        prices,
        fundamentals,
        universe,
        min_train_days=252,
    )
    assert report["status"] == "blocked"
    assert report["portfolio"] is None
