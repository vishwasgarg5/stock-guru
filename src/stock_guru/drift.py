from __future__ import annotations

import math
import pandas as pd


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame, features: list[str],
                  psi_threshold: float = 0.20, bins: int = 10) -> dict:
    """Measure population stability index (PSI) for model-input drift.

    Reference distributions define the bins; current observations are scored
    against those fixed bins so the diagnostic is comparable over time.
    """
    if bins < 2:
        raise ValueError("bins must be at least 2")
    rows = {}
    for feature in features:
        if feature not in reference or feature not in current:
            continue
        ref = pd.to_numeric(reference[feature], errors="coerce").dropna()
        cur = pd.to_numeric(current[feature], errors="coerce").dropna()
        if len(ref) < bins or cur.empty:
            continue
        quantiles = ref.quantile([i / bins for i in range(1, bins)]).to_numpy()
        edges = [-math.inf, *quantiles.tolist(), math.inf]
        ref_counts = pd.cut(ref, bins=edges, include_lowest=True).value_counts(sort=False).to_numpy(dtype=float)
        cur_counts = pd.cut(cur, bins=edges, include_lowest=True).value_counts(sort=False).to_numpy(dtype=float)
        ref_pct = (ref_counts + 1e-6) / (ref_counts.sum() + 1e-6 * bins)
        cur_pct = (cur_counts + 1e-6) / (cur_counts.sum() + 1e-6 * bins)
        psi = float(((cur_pct - ref_pct) * (cur_pct / ref_pct).map(lambda x: math.log(x))).sum())
        rows[feature] = {"psi": psi, "drifted": psi >= psi_threshold,
                         "reference_samples": int(len(ref)), "current_samples": int(len(cur))}
    return rows


def drift_summary(diagnostics: dict) -> dict:
    """Convert feature diagnostics into a compact monitoring summary."""
    if not diagnostics:
        return {"features": 0, "drifted_features": 0, "max_psi": 0.0}
    values = [float(v["psi"]) for v in diagnostics.values() if "psi" in v]
    return {"features": len(diagnostics),
            "drifted_features": sum(bool(v.get("drifted")) for v in diagnostics.values()),
            "max_psi": max(values) if values else 0.0}
