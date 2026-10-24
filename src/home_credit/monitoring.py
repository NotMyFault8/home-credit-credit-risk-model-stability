"""Replay monitoring with explicit simulated label availability."""

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import evaluate


def psi(reference, current, bins=10):
    reference, current = np.asarray(reference), np.asarray(current)
    reference, current = (
        reference[np.isfinite(reference)],
        current[np.isfinite(current)],
    )
    if not len(reference) or not len(current):
        return None
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        edges = np.array([-np.inf, float(reference[0]), np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    a, b = np.histogram(reference, edges)[0], np.histogram(current, edges)[0]
    a, b = (a + 0.5) / (a.sum() + 0.5 * len(a)), (b + 0.5) / (b.sum() + 0.5 * len(b))
    return float(np.sum((b - a) * np.log(b / a)))


def report(gold, predictions, reference, output, as_of_week=None, label_delay_weeks=2):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    as_of_week = int(predictions.WEEK_NUM.max()) if as_of_week is None else int(as_of_week)
    risk, performance, quality, calibration = [], [], [], []
    for (version, mode, week), group in predictions.groupby(["model_version", "mode", "WEEK_NUM"]):
        mature = int(week) + label_delay_weeks <= as_of_week
        common = {
            "WEEK_NUM": int(week),
            "model_version": version,
            "mode": mode,
            "as_of_week": as_of_week,
            "label_mature": mature,
            "sample_scope": "final_holdout",
        }
        risk.append(
            {
                **common,
                "applications": len(group),
                "mean_score": float(group.score.mean()),
                "low_risk": int((group.score < 0.2).sum()),
                "medium_risk": int(group.score.between(0.2, 0.5, inclusive="left").sum()),
                "high_risk": int((group.score >= 0.5).sum()),
                "default_rate": float(group.target.mean()) if mature else None,
            }
        )
        if mature:
            _, weekly = evaluate(group)
            performance.append({**common, **weekly.iloc[0].to_dict()})
            buckets = pd.cut(group.score, np.linspace(0, 1, 11), include_lowest=True)
            for bucket, part in group.groupby(buckets, observed=True):
                calibration.append(
                    {
                        **common,
                        "bin": str(bucket),
                        "applications": len(part),
                        "mean_score": float(part.score.mean()),
                        "default_rate": float(part.target.mean()),
                    }
                )
        else:
            performance.append({**common, "applications": len(group), "status": "labels_pending"})
    for week, group in gold[gold.WEEK_NUM.isin(predictions.WEEK_NUM.unique())].groupby("WEEK_NUM"):
        for column in gold:
            if column in {"case_id", "target", "date_decision", "WEEK_NUM", "MONTH"}:
                continue
            numeric = pd.api.types.is_numeric_dtype(gold[column])
            drift = (
                psi(
                    reference[column].to_numpy(dtype=float),
                    group[column].to_numpy(dtype=float),
                )
                if numeric
                else None
            )
            quality.append(
                {
                    "WEEK_NUM": int(week),
                    "feature": column,
                    "missing_rate": float(group[column].isna().mean()),
                    "reference_missing_rate": float(reference[column].isna().mean()),
                    "coverage": float(group[column].mean())
                    if column.endswith("__available")
                    else None,
                    "psi": drift,
                    "alert": bool(drift is not None and drift > 0.2),
                    "reference_scope": "development",
                    "sample_scope": "final_holdout",
                }
            )
    tables = {
        "risk_summary_weekly": pd.DataFrame(risk),
        "model_performance_weekly": pd.DataFrame(performance),
        "data_quality_weekly": pd.DataFrame(quality),
        "calibration_weekly": pd.DataFrame(calibration),
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    return tables
