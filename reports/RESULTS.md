# Measured results

**Run date: October 24, 2026.** These are local, retrospective results on the supplied competition training data, not leaderboard scores or production lending outcomes. Weeks 88–91 have already been inspected and reported; extension analyses on them are not a newly untouched independent test.

## Dataset and split

- 1,526,659 applications across competition weeks 0–91; no application-row sampling.
- 52,611 applications in the untouched final holdout, weeks 88–91.
- Selected static features plus person, external bureau and depth-two prior-application histories; 35 engineered features in the full model.
- Source rows: 1,526,659 static; 2,973,991 person; 85,791 bureau; 14,075,487 depth-two previous records.
- Three expanding validation folds: train through weeks 75/79/83, validate on 76–79/80–83/84–87. Final fits use weeks 0–87.

## Final holdout

| Model mode | AUC | Brier ↓ | Stability ↑ | Weekly Gini slope |
|---|---:|---:|---:|---:|
| Full sources | 0.767139 | 0.021428 | 0.521721 | 0.006552 |
| Internal fallback | 0.764634 | 0.021461 | 0.516981 | 0.009264 |

All four holdout weeks have defined Gini. Source-outage fallback reduces AUC by approximately 0.0025 on this holdout. That is an observed difference, not a claim of statistical significance.

![Measured holdout behavior](model-stability.png)

## Development-only feature experiments

These exploratory comparisons use the three development-validation folds only. Their scores are means of fold metrics, not overall holdout metrics. They do not change the already-frozen final candidate or reuse the holdout for tuning.

| Feature/training variant | Mean validation AUC | Mean fold stability | Features |
|---|---:|---:|---:|
| Static only | 0.72001 | 0.41414 | 10 |
| Add history | 0.75323 | 0.48232 | 35 |
| Keep stable missingness | 0.62569 | 0.24337 | 29 |
| Recent-time weighting | 0.75524 | 0.48948 | 35 |

The selected missingness-stability rule materially hurts discrimination. Recent-time weighting gives a small validation gain that has not been independently confirmed on a new untouched test period. Neither is presented as a proven production improvement.

## Validation extension run — October 31, 2026

The validation extension was rerun on the same 1,526,659 historical applications and the same Week 88–91 retrospective evaluation population. This is additional analysis of a previously inspected period, not a new independent holdout. Full and Internal both selected sigmoid calibration using only Week 76–87 temporal out-of-fold development predictions; method selection used Weeks 84–87 and did not use final-evaluation labels.

| Mode / score | AUC | Brier ↓ | Log loss ↓ | ECE ↓ | O/E | Calibration intercept | Calibration slope |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full raw | 0.767139 | 0.021428100 | 0.097868 | 0.005481 | 0.820808 | 0.613774 | 1.257168 |
| Full calibrated | 0.767139 | 0.021428147 | 0.097084 | 0.003104 | 1.142024 | 0.170518 | 1.008225 |
| Internal raw | 0.764634 | 0.021461090 | 0.098195 | 0.005690 | 0.824359 | 0.724698 | 1.288337 |
| Internal calibrated | 0.764634 | 0.021449216 | 0.097388 | 0.003388 | 1.140522 | 0.189152 | 1.014140 |

Calibration preserved AUC, reduced log loss and ECE, and moved slope closer to 1. It also moved O/E from underprediction (about 0.82) to overprediction (about 1.14); Full Brier worsened by approximately 0.000000047 while Internal Brier improved by approximately 0.0000119. The evidence is therefore mixed rather than a blanket calibration improvement.

The frozen segment analysis produced 260 overall/segment/week rows; 3 low-event slices were explicitly marked not estimable. Illustrative retrospective differences include Full AUC 0.680567 for credit type `CAL` versus 0.801319 for `COL`, and Full AUC 0.746134 in the highest credit-amount band versus 0.788324 in the next-lowest band. External-bureau coverage was associated with a higher observed default rate (5.27% with coverage versus 2.16% without), but this is association, not evidence that coverage causes risk. These are robustness diagnostics, not fairness certification.

Weekly Full AUC ranged from 0.750314 to 0.786778. `person__records` exceeded the example PSI threshold of 0.2 in all four weeks (0.643–0.945 against the Week 0–87 development reference); this is a traceable diagnostic alert, not an automatic retraining decision. The run completed successfully in 117.33 seconds with approximately 4,596 MB peak RSS. Aggregate evidence and the generated report are under [`competition-extension/`](competition-extension/).

## Engineering measurements

- Final full-data run: **136.63 seconds**, peak process RSS **3,932 MB**, macOS 26.6.2 / Apple ARM64. Includes source snapshot copying, features, candidate training and reports. Other local workloads were running; this is a single observation, not a benchmark distribution.
- API verification: **30 serial in-process requests × 100 rows**, P95 **22.87 ms**, approximately **5,974 rows/sec**. Includes bundle loading and request handling; excludes network transport and concurrent-client load.
- API/batch maximum score difference: **0.0** on the verification sample.
- Explicit external-source fallback, modified-row backfill equivalence, failed-run audit, and rollback between distinct version identifiers all passed.
- Docker API: live health endpoint returned ready; HTTP scores matched local batch scores exactly on three verification records.
- Automated tests: **14 passed**. Optional MLflow and Prefect integrations were exercised on synthetic data.

Machine-readable evidence: [evaluation](competition/evaluation.json), [verification](competition/verification.json), [dataset scope](competition/dataset-summary.json), [ablations](competition/ablations.csv), [run metrics](competition/bi/pipeline_runs.csv). The [reporting guide](REPORTING.md) explains the chart and aggregate CSVs.

## Synthetic demonstration

The separate [demo results](demo/evaluation.json) use 3,200 fictional applications and test engineering behavior. Their distributions and scores are unrelated to competition performance. Negative stability results in the demo are retained, not suppressed.

## Scope and limitations

Reporting deliverables include a Python-generated model-stability chart, machine-readable aggregate CSV summaries, an automatically generated validation report for new runs, and the packaged Tableau workbook under `reports/tableau/`. No managed cloud deployment is included. Historical source timing, real label maturity, fairness, production calibration, model governance and operational service-level objectives require separate validation before real lending use.
