"""Development-frozen segment definitions and robustness evaluation."""

import json

import numpy as np
import pandas as pd

from .evaluation import point_metrics

RECOMMENDED = {
    "credit_amount": "credamount_770A",
    "credit_type": "credtype_322L",
    "occupation_income_history": "mainoccupationinc_384A",
    "delinquency_history": "dpd_550P",
    "external_bureau_coverage": "bureau__available",
}


def _resolve(frame, token):
    exact = [column for column in frame if column == token]
    matches = [column for column in frame if token in column]
    return (exact or matches or [None])[0]


def fit_segment_definitions(development, requested=None, quantiles=4):
    """Learn numeric boundaries/categories using development rows only."""
    requested = requested or RECOMMENDED
    definitions = []
    for name, token in requested.items():
        column = _resolve(development, token)
        if column is None:
            continue
        series = development[column]
        if pd.api.types.is_numeric_dtype(series) and not column.endswith("__available"):
            values = series.dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            edges = np.unique(np.quantile(values, np.linspace(0, 1, quantiles + 1)))
            if len(edges) < 2:
                edges = np.array([-np.inf, np.inf])
            else:
                edges[0], edges[-1] = -np.inf, np.inf
            definition = {
                "name": name,
                "column": column,
                "kind": "numeric",
                "edges": [float(value) if np.isfinite(value) else None for value in edges],
                "source": "development_quantiles",
            }
        else:
            categories = sorted(str(value) for value in series.dropna().unique())
            definition = {
                "name": name,
                "column": column,
                "kind": "categorical",
                "categories": categories,
                "source": "development_categories",
            }
        definitions.append(definition)
    return definitions


def risk_segment_definition(edges, mode, score_column="raw_score"):
    """Create a model-specific risk tier from development-frozen score edges."""
    return {
        "name": "development_risk_tier",
        "column": score_column,
        "kind": "numeric",
        "edges": [float(value) if np.isfinite(value) else None for value in edges],
        "source": "development_temporal_oof_quantiles",
        "mode": mode,
    }


def _assign(frame, definition):
    series = frame[definition["column"]]
    if definition["kind"] == "numeric":
        edges = [
            -np.inf if index == 0 and value is None else np.inf if value is None else value
            for index, value in enumerate(definition["edges"])
        ]
        assigned = pd.cut(series, edges, include_lowest=True, duplicates="drop")
        return assigned.astype(object).where(series.notna(), "missing").astype(str)
    text = series.astype(object).where(series.notna(), "missing").astype(str)
    known = set(definition["categories"]) | {"missing"}
    return text.where(text.isin(known), "unknown")


def _slice_metrics(part, score_column, min_samples, min_events):
    events = int(part.target.sum())
    estimable = len(part) >= min_samples and min_events <= events <= len(part) - min_events
    metrics = point_metrics(part, score_column, min_samples=1)
    if not estimable:
        for key in (
            "auc",
            "auc_ci_low",
            "auc_ci_high",
            "gini",
            "calibration_intercept",
            "calibration_slope",
        ):
            metrics[key] = None
        metrics["status"] = "not_estimable_low_sample_or_event_count"
    return metrics


def evaluate_segments(
    gold,
    predictions,
    definitions,
    score_column="calibrated_score",
    min_samples=50,
    min_events=5,
    include_weekly=True,
):
    """Compare overall and frozen segments for each model mode and time slice."""
    feature_columns = sorted(
        {definition["column"] for definition in definitions} - set(predictions.columns)
    )
    joined = predictions.merge(
        gold[["case_id"] + feature_columns], on="case_id", how="left", validate="many_to_one"
    )
    rows = []
    periods = [("overall", None)]
    if include_weekly:
        periods.extend(("week", int(week)) for week in sorted(joined.WEEK_NUM.unique()))
    for mode, mode_group in joined.groupby("mode"):
        for period, week in periods:
            population = mode_group if week is None else mode_group[mode_group.WEEK_NUM == week]
            common = {
                "mode": mode,
                "model_version": str(population.model_version.iloc[0]),
                "period": period,
                "WEEK_NUM": week,
                "score_type": score_column,
            }
            rows.append(
                {
                    **common,
                    "segment": "overall",
                    "definition": "all applications",
                    "group": "all",
                    **_slice_metrics(population, score_column, min_samples, min_events),
                }
            )
            for definition in definitions:
                if definition.get("mode") not in {None, mode}:
                    continue
                labels = _assign(population, definition)
                for label, part in population.groupby(labels, dropna=False):
                    rows.append(
                        {
                            **common,
                            "segment": definition["name"],
                            "definition": json.dumps(definition, sort_keys=True),
                            "group": str(label),
                            **_slice_metrics(part, score_column, min_samples, min_events),
                        }
                    )
    return pd.DataFrame(rows)
