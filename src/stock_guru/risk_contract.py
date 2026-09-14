from __future__ import annotations


def validate_position_limits(weights: dict[str, float], *, max_position_pct: float = 0.10) -> dict:
    if not 0 < max_position_pct <= 1:
        raise ValueError("max_position_pct must be in (0, 1]")
    for symbol, weight in weights.items():
        if not str(symbol).strip():
            raise ValueError("Position contains blank symbol")
        if weight < 0 or weight > max_position_pct:
            raise ValueError(f"Position limit exceeded for {symbol}")
    return {"positions": len(weights), "gross_weight": float(sum(weights.values())), "validated": True}
