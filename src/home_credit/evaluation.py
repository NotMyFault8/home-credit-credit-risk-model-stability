"""Time-aware evaluation with explicit handling for non-estimable metrics."""

import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

EPSILON = 1e-7


def fixed_quantile_edges(reference, bins=10):
    """Return frozen score-bin edges learned from a reference population."""
    values = np.asarray(reference, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.array([-np.inf, np.inf])
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return np.array([-np.inf, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def calibration_error(target, score, edges=None, bins=10):
    """Expected calibration error, plus auditable per-bin counts."""
    target = np.asarray(target, dtype=float)
    score = np.asarray(score, dtype=float)
    edges = fixed_quantile_edges(score, bins) if edges is None else np.asarray(edges)
    labels = pd.cut(score, edges, include_lowest=True, duplicates="drop")
    table = pd.DataFrame({"target": target, "score": score, "bin": labels})
    rows, weighted_error = [], 0.0
    for label, part in table.groupby("bin", observed=True, sort=False):
        observed, predicted = float(part.target.mean()), float(part.score.mean())
        weighted_error += len(part) * abs(observed - predicted)
        rows.append(
            {
                "bin": str(label),
                "applications": int(len(part)),
                "events": int(part.target.sum()),
                "mean_score": predicted,
                "default_rate": observed,
            }
        )
    return (weighted_error / len(table) if len(table) else None), pd.DataFrame(rows)


def _auc_interval(target, auc, confidence=0.95):
    """Hanley-McNeil large-sample AUC interval; unavailable for a single class."""
    positives, negatives = int(np.sum(target)), int(len(target) - np.sum(target))
    if auc is None or not positives or not negatives:
        return None, None
    q1, q2 = auc / (2 - auc), 2 * auc**2 / (1 + auc)
    variance = (
        auc * (1 - auc)
        + (positives - 1) * (q1 - auc**2)
        + (negatives - 1) * (q2 - auc**2)
    ) / (positives * negatives)
    z = 1.959963984540054 if confidence == 0.95 else 1.96
    margin = z * math.sqrt(max(variance, 0))
    return max(0.0, auc - margin), min(1.0, auc + margin)


def _calibration_regression(target, score):
    if len(np.unique(target)) != 2:
        return None, None
    clipped = np.clip(np.asarray(score, dtype=float), EPSILON, 1 - EPSILON)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    try:
        fitted = LogisticRegression(C=1e6, solver="lbfgs").fit(logit, target)
    except (ValueError, FloatingPointError):
        return None, None
    return float(fitted.intercept_[0]), float(fitted.coef_[0, 0])


def point_metrics(frame, score_column="score", min_samples=20, bin_edges=None):
    """Metrics for one evaluation slice without dropping negative results."""
    target = frame.target.to_numpy()
    scores = np.clip(frame[score_column].to_numpy(dtype=float), EPSILON, 1 - EPSILON)
    estimable = len(frame) >= min_samples and len(np.unique(target)) == 2
    auc = float(roc_auc_score(target, scores)) if estimable else None
    auc_low, auc_high = _auc_interval(target, auc)
    ece, _ = calibration_error(target, scores, bin_edges)
    predicted_events = float(scores.sum())
    intercept, slope = _calibration_regression(target, scores) if estimable else (None, None)
    return {
        "applications": int(len(frame)),
        "events": int(target.sum()),
        "default_rate": float(np.mean(target)) if len(target) else None,
        "auc": auc,
        "auc_ci_low": auc_low,
        "auc_ci_high": auc_high,
        "gini": 2 * auc - 1 if auc is not None else None,
        "brier": float(brier_score_loss(target, scores)) if len(target) else None,
        "log_loss": float(log_loss(target, scores, labels=[0, 1])) if len(target) else None,
        "ece": ece,
        "oe_ratio": float(target.sum() / predicted_events) if predicted_events > 0 else None,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "mean_score": float(np.mean(scores)) if len(scores) else None,
        "status": "ok" if estimable else "insufficient_samples_or_single_class",
    }


def evaluate(frame, min_week_samples=20, score_column="score", bin_edges=None):
    rows = []
    for week, group in frame.groupby("WEEK_NUM", sort=True):
        rows.append(
            {
                "WEEK_NUM": int(week),
                **point_metrics(group, score_column, min_week_samples, bin_edges),
            }
        )
    weekly = pd.DataFrame(rows)
    score, slope, residual_std = None, None, None
    if len(weekly) >= 2 and weekly.gini.notna().all():
        # Competition reference uses ordinal positions of sorted observed weeks.
        x, y = np.arange(len(weekly)), weekly.gini.to_numpy()
        slope, intercept = np.polyfit(x, y, 1)
        residual_std = float(np.std(y - (slope * x + intercept), ddof=0))
        score = float(y.mean() + 88 * min(0, slope) - 0.5 * residual_std)
        slope = float(slope)
    overall = point_metrics(frame, score_column, min_samples=1, bin_edges=bin_edges)
    return {
        "auc": overall["auc"],
        "auc_ci_low": overall["auc_ci_low"],
        "auc_ci_high": overall["auc_ci_high"],
        "gini": overall["gini"],
        "brier": overall["brier"],
        "log_loss": overall["log_loss"],
        "ece": overall["ece"],
        "oe_ratio": overall["oe_ratio"],
        "calibration_intercept": overall["calibration_intercept"],
        "calibration_slope": overall["calibration_slope"],
        "applications": overall["applications"],
        "events": overall["events"],
        "stability": score,
        "slope": slope,
        "residual_std": residual_std,
        "weeks": len(weekly),
        "undefined_weeks": int(weekly.gini.isna().sum()),
    }, weekly


def temporal_splits(frame, test_weeks=4, validation_weeks=4, folds=3):
    weeks = np.sort(frame.WEEK_NUM.unique())
    if len(weeks) < test_weeks + folds * validation_weeks + 4:
        raise ValueError("Insufficient weeks for rolling validation and untouched test")
    test = weeks[-test_weeks:]
    development = weeks[:-test_weeks]
    splits = []
    for offset in range(folds, 0, -1):
        start = len(development) - offset * validation_weeks
        train = development[:start]
        valid = development[start : start + validation_weeks]
        splits.append((frame.WEEK_NUM.isin(train), frame.WEEK_NUM.isin(valid)))
    return splits, ~frame.WEEK_NUM.isin(test), frame.WEEK_NUM.isin(test)
