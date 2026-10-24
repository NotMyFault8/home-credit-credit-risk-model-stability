# Tableau risk monitoring

This folder contains the Tableau workbook and the weekly holdout data used by its charts.

## Files

- `home-credit-risk-stability.twbx` — packaged Tableau workbook with model-performance and risk-calibration views.
- `tableau-weekly-holdout.csv` — weekly holdout summary data used by the workbook.

## Open the workbook

Download the `.twbx` file and open it in Tableau Desktop. The workbook includes a packaged extract for viewing. The CSV is included alongside it for reference and data-source refreshes; if Tableau prompts for the source file, point it to `tableau-weekly-holdout.csv`.

The data is aggregated by week and model. It contains no borrower-level records.
