"""Leakage-resistant probability calibration for temporal model validation."""

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .evaluation import evaluate, fixed_quantile_edges


class ProbabilityCalibrator:
    """Small serializable wrapper around raw, Platt and isotonic transforms."""

    def __init__(self, method="none"):
        self.method = method
        self.estimator = None

    def fit(self, scores, target):
        scores = np.asarray(scores, dtype=float)
        target = np.asarray(target, dtype=int)
        if self.method == "sigmoid":
            clipped = np.clip(scores, 1e-7, 1 - 1e-7)
            logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
            self.estimator = LogisticRegression(C=1e6, solver="lbfgs").fit(logits, target)
        elif self.method == "isotonic":
            self.estimator = IsotonicRegression(out_of_bounds="clip").fit(scores, target)
        elif self.method != "none":
            raise ValueError(f"Unknown calibration method: {self.method}")
        return self

    def predict(self, scores):
        scores = np.asarray(scores, dtype=float)
        if self.method == "none":
            return np.clip(scores, 0, 1)
        if self.estimator is None:
            raise ValueError("Calibrator has not been fitted")
        if self.method == "sigmoid":
            clipped = np.clip(scores, 1e-7, 1 - 1e-7)
            logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
            return self.estimator.predict_proba(logits)[:, 1]
        return self.estimator.predict(scores)


def select_calibrator(oof, model_version, data_version, mode, seed=42):
    """Choose on the last OOF fold, then refit on all temporal OOF predictions."""
    required = {"target", "raw_score", "WEEK_NUM", "train_max_week"}
    if not required <= set(oof):
        raise ValueError(f"Missing calibration protocol fields: {sorted(required - set(oof))}")
    if (oof.train_max_week >= oof.WEEK_NUM).any():
        raise ValueError("Calibration leakage: predictions are not strictly out-of-time")
    weeks = np.sort(oof.WEEK_NUM.unique())
    if len(weeks) < 2:
        raise ValueError("Calibration selection needs at least two validation periods")
    selection_weeks = weeks[-max(1, len(weeks) // 3) :]
    fit = oof[~oof.WEEK_NUM.isin(selection_weeks)]
    selection = oof[oof.WEEK_NUM.isin(selection_weeks)]
    if fit.target.nunique() != 2 or selection.target.nunique() != 2:
        raise ValueError("Calibration fit and selection windows both need two target classes")
    edges = fixed_quantile_edges(fit.raw_score)
    comparison = []
    for method in ("none", "sigmoid", "isotonic"):
        calibrator = ProbabilityCalibrator(method).fit(fit.raw_score, fit.target)
        candidate = selection[["WEEK_NUM", "target"]].copy()
        candidate["score"] = calibrator.predict(selection.raw_score)
        metrics, _ = evaluate(candidate, min_week_samples=1, bin_edges=edges)
        comparison.append({"method": method, **metrics})
    chosen = min(comparison, key=lambda row: (row["brier"], row["log_loss"], row["method"]))[
        "method"
    ]
    final = ProbabilityCalibrator(chosen).fit(oof.raw_score, oof.target)
    raw_edges = fixed_quantile_edges(oof.raw_score)
    calibrated_edges = fixed_quantile_edges(final.predict(oof.raw_score))
    protocol = {
        "method": chosen,
        "model_version": model_version,
        "mode": mode,
        "data_version": data_version,
        "seed": seed,
        "fit_min_week": int(oof.WEEK_NUM.min()),
        "fit_max_week": int(oof.WEEK_NUM.max()),
        "selection_weeks": [int(w) for w in selection_weeks],
        "samples": int(len(oof)),
        "events": int(oof.target.sum()),
        "source": "development_temporal_out_of_fold",
        "comparison": comparison,
        "score_bin_edges": {
            "raw_score": [float(x) if np.isfinite(x) else None for x in raw_edges],
            "calibrated_score": [
                float(x) if np.isfinite(x) else None for x in calibrated_edges
            ],
        },
    }
    protocol["version"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()[:16]
    return final, protocol


def calibration_comparison(frame, bin_edges=None):
    """Quantify raw and calibrated behavior on the same evaluation rows."""
    rows, weekly_rows = [], []
    for column in ("raw_score", "calibrated_score"):
        edges = bin_edges.get(column) if isinstance(bin_edges, dict) else bin_edges
        metrics, weekly = evaluate(frame, score_column=column, bin_edges=edges)
        rows.append({"score_type": column, **metrics})
        weekly.insert(0, "score_type", column)
        weekly_rows.append(weekly)
    return pd.DataFrame(rows), pd.concat(weekly_rows, ignore_index=True)
