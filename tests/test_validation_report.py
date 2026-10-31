import json

import pandas as pd

from home_credit.validation_report import generate_validation_report


def test_validation_report_is_generated_from_persisted_evidence(tmp_path):
    (tmp_path / "bi").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "evaluation.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "holdout": {
                    "full": {
                        "auc": 0.75,
                        "gini": 0.5,
                        "brier": 0.1,
                        "log_loss": 0.3,
                        "ece": 0.02,
                        "oe_ratio": 1.1,
                        "stability": 0.4,
                    }
                },
            }
        )
    )
    (tmp_path / "data/manifest.json").write_text(
        json.dumps({"synthetic": True, "data_version": "data-v1", "feature_version": "f-v1"})
    )
    pd.DataFrame([{"mode": "full", "score_type": "raw_score"}]).to_csv(
        tmp_path / "bi/calibration_comparison.csv", index=False
    )
    pd.DataFrame([{"mode": "full", "segment": "overall"}]).to_csv(
        tmp_path / "bi/segment_performance.csv", index=False
    )
    pd.DataFrame([{"metric": "feature_psi"}]).to_csv(
        tmp_path / "bi/stability_alerts.csv", index=False
    )
    report = generate_validation_report(tmp_path)
    text = report.read_text()
    assert "SYNTHETIC ENGINEERING DEMO" in text
    assert "0.750000" in text
    assert "data-v1" in text
    assert "not fairness certification" in text
