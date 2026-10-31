# Reporting outputs

The project's reporting layer includes Python-generated charts, compact CSV summaries, an artifact-generated Markdown validation report, and a packaged Tableau workbook under `tableau/`. It runs locally and has no managed dashboard service.

## Chart

[`model-stability.png`](model-stability.png) shows final-holdout behavior for the full-source and internal-fallback models. The figure is generated from the evaluation outputs:

```bash
python scripts/plot_results.py reports/competition/bi reports/model-stability.png
```

To regenerate against the fictional end-to-end example, point the script at `reports/demo/bi` and write to a separate path so the real-data figure stays clearly identified.

## Summary tables

The pipeline writes small CSV tables under `reports/competition/bi/` and `reports/demo/bi/`:

- `risk_summary_weekly.csv` tracks scored volume, risk bands, model mode, observation cutoff and whether labels are mature.
- `model_performance_weekly.csv` tracks weekly AUC, Gini and Brier score where enough mature labels are available.
- `calibration_weekly.csv` groups raw and calibrated predictions into development-frozen score bins and includes application/event counts.
- `calibration_comparison.csv` quantifies raw versus calibrated AUC/Gini, Brier, log loss, ECE, O/E and calibration intercept/slope.
- `segment_performance.csv` compares overall and frozen application/risk segments, with explicit non-estimable states.
- `data_quality_weekly.csv` tracks numeric/categorical PSI, missingness and source coverage.
- `score_stability_weekly.csv` tracks raw/calibrated score distributions and PSI.
- `stability_alerts.csv` records threshold, affected week, model version and reference scope.
- `pipeline_runs.csv` records each pipeline run, its status, duration, row count and peak memory.

Python computes metrics before export. Keep model modes, weeks, label maturity and sample scope visible when comparing rows. AUC and Gini are not additive; use the overall evaluation metrics for whole-period comparisons. Empty metric cells mean unavailable, not zero.

Each new run also writes `MODEL_VALIDATION_REPORT.md`, `validation-calibration.svg` and `validation-weekly-auc.svg`. See the committed [validation report](MODEL_VALIDATION_REPORT.md) for the existing retrospective evidence; unavailable extension results remain explicitly marked until the restricted real dataset is rerun.

See [measured results](RESULTS.md) for real-data values and limitations, and [the data guide](../docs/data-dictionary.md) for table grain and definitions.
