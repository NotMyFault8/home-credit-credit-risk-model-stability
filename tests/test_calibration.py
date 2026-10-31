import numpy as np
import pandas as pd
import pytest

from home_credit.calibration import calibration_comparison, select_calibrator


def temporal_oof():
    rows = []
    for week in range(10, 16):
        for index in range(40):
            target = int(index % 5 == 0)
            raw = 0.6 if target else 0.2
            rows.append(
                {
                    "WEEK_NUM": week,
                    "train_max_week": week - 1,
                    "target": target,
                    "raw_score": raw,
                }
            )
    return pd.DataFrame(rows)


def test_calibrator_uses_only_strictly_out_of_time_predictions():
    oof = temporal_oof()
    calibrator, protocol = select_calibrator(oof, "model-1", "data-1", "full")
    assert protocol["model_version"] == "model-1"
    assert protocol["source"] == "development_temporal_out_of_fold"
    assert protocol["fit_max_week"] == 15
    calibrated = calibrator.predict(oof.raw_score)
    assert np.isfinite(calibrated).all()
    assert ((0 <= calibrated) & (calibrated <= 1)).all()

    leaked = oof.copy()
    leaked.loc[0, "train_max_week"] = leaked.loc[0, "WEEK_NUM"]
    with pytest.raises(ValueError, match="leakage"):
        select_calibrator(leaked, "model-1", "data-1", "full")


def test_calibration_comparison_quantifies_both_score_versions():
    frame = temporal_oof().rename(columns={"raw_score": "score"})
    frame["raw_score"] = frame.score
    frame["calibrated_score"] = np.where(frame.target == 1, 0.3, 0.175)
    overall, weekly = calibration_comparison(frame)
    assert set(overall.score_type) == {"raw_score", "calibrated_score"}
    assert {"brier", "log_loss", "ece", "oe_ratio", "calibration_slope"} <= set(overall)
    assert set(weekly.score_type) == {"raw_score", "calibrated_score"}
