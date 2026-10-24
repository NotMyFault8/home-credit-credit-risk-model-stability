"""Time-aware evaluation. Undefined weeks make the stability score undefined."""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def evaluate(frame, min_week_samples=20):
    rows = []
    for week, group in frame.groupby("WEEK_NUM", sort=True):
        valid = len(group) >= min_week_samples and group.target.nunique() == 2
        auc = float(roc_auc_score(group.target, group.score)) if valid else None
        rows.append(
            {
                "WEEK_NUM": int(week),
                "applications": len(group),
                "default_rate": float(group.target.mean()),
                "auc": auc,
                "gini": 2 * auc - 1 if auc is not None else None,
                "brier": float(brier_score_loss(group.target, group.score)),
                "status": "ok" if valid else "insufficient_samples_or_single_class",
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
    overall_auc = (
        float(roc_auc_score(frame.target, frame.score)) if frame.target.nunique() == 2 else None
    )
    return {
        "auc": overall_auc,
        "brier": float(brier_score_loss(frame.target, frame.score)),
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
