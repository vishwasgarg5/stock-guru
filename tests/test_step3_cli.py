import pytest

from stock_guru import cli


def test_step3_cli_parser_is_registered(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "stock-guru",
            "step3",
            "--prices", "prices.csv",
            "--fundamentals", "fundamentals.csv",
            "--universe", "universe.csv",
        ],
    )
    # main() will reach the file runner; intercept it so this test checks only
    # parser wiring and argument names, not external data availability.
    captured = {}

    def fake_runner(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"status": "blocked", "folds": 0}

    monkeypatch.setattr("stock_guru.step3_certification.run_step3_from_files", fake_runner, raising=False)
    # The import is intentionally lazy in cli.main(), so patch the module it imports.
    import stock_guru.step3_certification as step3
    monkeypatch.setattr(step3, "run_step3_from_files", fake_runner)
    cli.main()
    assert captured["args"][:3] == ("prices.csv", "fundamentals.csv", "universe.csv")
    assert captured["kwargs"]["min_train_days"] == 252
    assert captured["kwargs"]["slippage_bps"] == 5.0
