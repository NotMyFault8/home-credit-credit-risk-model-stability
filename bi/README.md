# Credit Risk & Model Stability Dashboard

Open **credit-risk-stability.twb** in Tableau Desktop or Tableau Public. The workbook references the small aggregate CSVs under `../reports/competition/bi/`; raw applications and per-case predictions are not embedded.

Four pages, sixteen worksheets:

1. **Risk overview:** applications, mean score, observed mature default rate, and high-risk-band counts (score ≥ 0.5 is a demonstration band).
2. **Model stability:** weekly Gini, weekly AUC, Brier score and calibration. The header gives the separately calculated full-period stability scores.
3. **Data quality and drift:** missingness, PSI, source coverage and reference missingness.
4. **Pipeline operations:** run duration, processed rows, peak memory and scoring volume.

The shipped real-data report is retrospective: the observation cutoff is week 93, after the two-week assumed label delay for the final holdout ending at week 91. The pipeline's original replay exports retain pending labels at cutoff week 91. A blank value is unavailable, not zero.

Color separates full and internal models. Compare them without summing their application counts. Weekly metrics are calculated in Python, not recomputed or averaged into overall scores by Tableau. Any displayed average operates only within the worksheet's week/model/feature grain. The overview is a fixed final-holdout report, not a dashboard of actual approvals, losses or income.

## Regenerate

```bash
python scripts/publish_reports.py artifacts/competition reports/competition --label 'REAL COMPETITION DATA'
python scripts/build_tableau.py --source reports/competition/bi \
  --output bi/credit-risk-stability.twb --label 'REAL DATA / RETROSPECTIVE'
python scripts/plot_results.py reports/competition/bi reports/model-stability.png
```

Use `reports/demo/bi` and the default synthetic label for the fictional demonstration. Do not mix these datasets. Reconnect the text-file sources if Tableau cannot resolve a relative path after moving the workbook.

## Validation status

The workbook contains four dashboards and sixteen linked worksheets and passes the structural constraints in Tableau's published 2026.2 workbook schema. The schema's unresolved imported namespaces were supplied locally; the workbook itself uses neither namespace. CSV paths and all referenced fields are checked locally.

The installed Tableau Public application repeatedly timed out during automated opening, so **rendered layout and interactive behavior have not been visually verified**. The static [evaluation figure](../reports/model-stability.png) is rendered and inspected separately. No Tableau Public upload was performed.
