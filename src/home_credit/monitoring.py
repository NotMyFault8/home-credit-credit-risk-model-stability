"""Replay monitoring with explicit simulated label availability and drift evidence."""

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import calibration_error, evaluate, fixed_quantile_edges


def psi(reference, current, bins=10):
    reference, current = np.asarray(reference), np.asarray(current)
    reference, current = reference[np.isfinite(reference)], current[np.isfinite(current)]
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


def categorical_psi(reference, current):
    """PSI over the union of categories, including an explicit missing value."""
    a = pd.Series(reference).astype(object).where(pd.notna(reference), "__missing__").astype(str)
    b = pd.Series(current).astype(object).where(pd.notna(current), "__missing__").astype(str)
    if not len(a) or not len(b):
        return None
    categories = sorted(set(a) | set(b))
    ac = a.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
    bc = b.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float)
    ac = (ac + 0.5) / (ac.sum() + 0.5 * len(ac))
    bc = (bc + 0.5) / (bc.sum() + 0.5 * len(bc))
    return float(np.sum((bc - ac) * np.log(bc / ac)))


def report(
    gold,
    predictions,
    reference,
    output,
    as_of_week=None,
    label_delay_weeks=2,
    score_bin_edges=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    predictions = predictions.copy()
    if "raw_score" not in predictions:
        predictions["raw_score"] = predictions.score
    if "calibrated_score" not in predictions:
        predictions["calibrated_score"] = predictions.score
    as_of_week = int(predictions.WEEK_NUM.max()) if as_of_week is None else int(as_of_week)
    risk, performance, quality, calibration, score_stability, alerts = [], [], [], [], [], []
    excluded = {"case_id", "target", "date_decision", "WEEK_NUM", "MONTH"}
    reference_min, reference_max = int(reference.WEEK_NUM.min()), int(reference.WEEK_NUM.max())

    for (version, mode), model_predictions in predictions.groupby(["model_version", "mode"]):
        first_week = int(model_predictions.WEEK_NUM.min())
        score_reference = model_predictions[model_predictions.WEEK_NUM == first_week]
        score_edges = (
            score_bin_edges.get(mode)
            if score_bin_edges and mode in score_bin_edges
            else {
                column: fixed_quantile_edges(score_reference[column])
                for column in ("raw_score", "calibrated_score")
            }
        )
        for week, group in model_predictions.groupby("WEEK_NUM"):
            mature = int(week) + label_delay_weeks <= as_of_week
            common = {
                "WEEK_NUM": int(week),
                "model_version": version,
                "mode": mode,
                "as_of_week": as_of_week,
                "label_mature": mature,
                "sample_scope": "retrospective_final_holdout",
            }
            risk.append(
                {
                    **common,
                    "applications": len(group),
                    "mean_score": float(group.score.mean()),
                    "mean_raw_score": float(group.raw_score.mean()),
                    "mean_calibrated_score": float(group.calibrated_score.mean()),
                    "low_risk": int((group.score < 0.2).sum()),
                    "medium_risk": int(group.score.between(0.2, 0.5, inclusive="left").sum()),
                    "high_risk": int((group.score >= 0.5).sum()),
                    "default_rate": float(group.target.mean()) if mature else None,
                }
            )
            if mature:
                _, raw_weekly = evaluate(group, score_column="raw_score")
                calibrated_metrics, _ = evaluate(group, score_column="calibrated_score")
                performance.append(
                    {
                        **common,
                        **raw_weekly.iloc[0].to_dict(),
                        **{f"calibrated_{key}": value for key, value in calibrated_metrics.items()},
                    }
                )
                for score_type in ("raw_score", "calibrated_score"):
                    _, curve = calibration_error(
                        group.target, group[score_type], score_edges[score_type]
                    )
                    for row in curve.to_dict(orient="records"):
                        calibration.append({**common, "score_type": score_type, **row})
            else:
                performance.append({**common, "applications": len(group), "status": "labels_pending"})
            for score_type in ("raw_score", "calibrated_score"):
                values = group[score_type]
                drift = psi(score_reference[score_type], values)
                row = {
                    **common,
                    "score_type": score_type,
                    "reference_week": first_week,
                    "mean": float(values.mean()),
                    "p10": float(values.quantile(0.1)),
                    "p50": float(values.quantile(0.5)),
                    "p90": float(values.quantile(0.9)),
                    "psi": drift,
                    "alert": bool(drift is not None and drift > 0.2),
                }
                score_stability.append(row)
                if row["alert"]:
                    alerts.append(
                        {
                            **common,
                            "metric": "score_psi",
                            "feature": score_type,
                            "value": drift,
                            "threshold": 0.2,
                            "reference_scope": f"holdout_week_{first_week}",
                            "interpretation": "diagnostic example; not an automatic retraining rule",
                        }
                    )

    version_text = ",".join(sorted(predictions.model_version.unique()))
    for week, group in gold[gold.WEEK_NUM.isin(predictions.WEEK_NUM.unique())].groupby("WEEK_NUM"):
        for column in gold:
            if column in excluded:
                continue
            numeric = pd.api.types.is_numeric_dtype(gold[column])
            drift = (
                psi(reference[column].to_numpy(dtype=float), group[column].to_numpy(dtype=float))
                if numeric
                else categorical_psi(reference[column], group[column])
            )
            row = {
                "WEEK_NUM": int(week),
                "feature": column,
                "feature_type": "numeric" if numeric else "categorical",
                "model_version": version_text,
                "missing_rate": float(group[column].isna().mean()),
                "reference_missing_rate": float(reference[column].isna().mean()),
                "coverage": float(group[column].mean()) if column.endswith("__available") else None,
                "psi": drift,
                "alert": bool(drift is not None and drift > 0.2),
                "reference_scope": f"development_weeks_{reference_min}_{reference_max}",
                "sample_scope": "retrospective_final_holdout",
            }
            quality.append(row)
            if row["alert"]:
                alerts.append(
                    {
                        "WEEK_NUM": int(week),
                        "model_version": version_text,
                        "mode": "all",
                        "metric": "feature_psi",
                        "feature": column,
                        "value": drift,
                        "threshold": 0.2,
                        "reference_scope": row["reference_scope"],
                        "interpretation": "diagnostic example; not an automatic approval or retraining rule",
                    }
                )
    tables = {
        "risk_summary_weekly": pd.DataFrame(risk),
        "model_performance_weekly": pd.DataFrame(performance),
        "data_quality_weekly": pd.DataFrame(quality),
        "calibration_weekly": pd.DataFrame(calibration),
        "score_stability_weekly": pd.DataFrame(score_stability),
        "stability_alerts": pd.DataFrame(alerts),
    }
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    return tables
