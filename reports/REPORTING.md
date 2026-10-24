# Reporting outputs

The project's reporting layer is a Python-generated chart with compact CSV summaries. It runs locally and has no separate dashboard application.

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
- `calibration_weekly.csv` groups predictions into score bins and compares mean score with observed default rate.
- `data_quality_weekly.csv` tracks missingness, source coverage and population stability index.
- `pipeline_runs.csv` records each pipeline run, its status, duration, row count and peak memory.

Python computes metrics before export. Keep model modes, weeks, label maturity and sample scope visible when comparing rows. AUC and Gini are not additive; use the overall evaluation metrics for whole-period comparisons. Empty metric cells mean unavailable, not zero.

See [measured results](RESULTS.md) for real-data values and limitations, and [the data guide](../docs/data-dictionary.md) for table grain and definitions.
