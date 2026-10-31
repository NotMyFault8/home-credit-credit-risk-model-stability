# Model Validation Report

> Evidence classification: **RETROSPECTIVE REAL-DATA RUN**. This report is generated from run artifacts.

## 1. Executive Summary

The validation framework covers discrimination, calibration, temporal behavior, segment robustness, data/score drift, and operational score parity; unavailable evidence is identified below. It is a research and engineering review, not approval for lending, a production-readiness statement, or a regulatory certification.

## 2. Data Provenance & Quality

- Data version: `1d2ae54ea09d1723`
- Feature version: `aggregate-v1`
- Source type: historical Kaggle competition data
- Point-in-time feature availability and real label maturity: **not assessed / unavailable**.

## 3. Model & Feature Methodology

Logistic regression and LightGBM candidates use train-only preprocessing. Full and internal fallback models are selected and versioned separately. The legacy `score` remains the raw model probability; `calibrated_score` is an additional, reversible output.

## 4. Validation Design

Three expanding temporal folds are used for model selection and out-of-time calibration predictions. Calibration method selection stays within development data. The final four weeks are a retrospective, previously inspected evaluation period, not a newly untouched independent test.

## 5. Discrimination & Calibration

| Mode | AUC | Gini | Brier | Log loss | ECE | O/E | Stability |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 0.767139 | 0.534277 | 0.021428 | 0.097868 | 0.005514 | 0.820808 | 0.521721 |
| internal | 0.764634 | 0.529267 | 0.021461 | 0.098195 | 0.005409 | 0.824359 | 0.516981 |

Calibration comparison is recorded in `bi/calibration_comparison.csv`; improvements must be judged by the numeric Brier/log-loss/ECE comparison, not chart appearance.

| Mode | Score | AUC | Brier | Log loss | ECE | O/E | Intercept | Slope |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full | raw_score | 0.767139 | 0.021428 | 0.097868 | 0.005481 | 0.820808 | 0.613774 | 1.257168 |
| full | calibrated_score | 0.767139 | 0.021428 | 0.097084 | 0.003104 | 1.142024 | 0.170518 | 1.008225 |
| internal | raw_score | 0.764634 | 0.021461 | 0.098195 | 0.005690 | 0.824359 | 0.724698 | 1.288337 |
| internal | calibrated_score | 0.764634 | 0.021449 | 0.097388 | 0.003388 | 1.140522 | 0.189152 | 1.014140 |

![Reliability diagram](validation-calibration.svg)

## 6. Segment Performance

`bi/segment_performance.csv` contains 260 overall/segment/time rows with explicit non-estimable states. This is segment robustness analysis, not fairness certification.

## 7. Stability & Drift

Weekly performance, feature PSI (numeric and categorical), missingness, source coverage, raw/calibrated score distributions, and score PSI are written under `bi/`.
PSI > 0.2 is an explanatory example threshold, not an automatic approval or retraining rule.
The run produced 4 traceable alert rows.

![Weekly AUC trend](validation-weekly-auc.svg)

## 8. Operational Verification

Batch and API use the same bound model/calibrator transformation. Model metadata records data, code, feature, seed, mode, base-model and calibrator versions. Activation/rollback continues to operate on the paired Full/Internal bundles.

## 9. Limitations & Risks

- Historical feature arrival times cannot be verified from `WEEK_NUM` alone.
- Label delay is simulated and must not be interpreted as observed production maturity.
- The curated feature set is limited; protected attributes and fairness mechanisms are absent.
- The final weeks have already been inspected and reported; a new independent test requires later data.
- Scores are not validated 12-month PDs and are not suitable for real loan approval.

## 10. Validation Conclusion & Action Items

Engineering checks are reproducible from the cited artifacts. Statistical results remain conditional on this historical/synthetic scope. Before real-world use, obtain a genuinely new time period, point-in-time lineage, real label-maturity evidence, independent governance review, and an appropriate fairness assessment.

## Evidence Index

- `evaluation.json`: versions, temporal experiments, retrospective holdout metrics
- `data/manifest.json`: data hashes, schemas and provenance (when retained with the run)
- `bi/calibration_comparison.csv`: raw versus calibrated metrics
- `bi/calibration_weekly.csv`: fixed-bin reliability evidence and event counts
- `bi/segment_performance.csv`: frozen segment definitions and metrics
- `bi/model_performance_weekly.csv`: weekly performance and calibration
- `bi/data_quality_weekly.csv`: feature drift, missingness and coverage
- `bi/score_stability_weekly.csv`: score distribution drift
- `bi/stability_alerts.csv`: threshold, week, model and reference lineage
