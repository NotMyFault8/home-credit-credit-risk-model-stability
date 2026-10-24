# Data contract and source guide

Download the [competition data](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data) using your own authorized Kaggle account and extract it under `data/raw/`. Follow the competition's actual terms. Do not commit or redistribute the download.

Expected layout:

```text
data/raw/
  feature_definitions.csv
  sample_submission.csv
  parquet_files/train/train_base.parquet
  parquet_files/train/train_static_0_0.parquet
  parquet_files/train/train_static_0_1.parquet
  parquet_files/train/train_person_1.parquet
  parquet_files/train/train_credit_bureau_b_1.parquet
  parquet_files/train/train_applprev_2.parquet
```

## Grain and meaning

| Source | Required key | Meaning / selected features |
|---|---|---|
| base | case_id | date_decision, WEEK_NUM, MONTH, target |
| static | case_id | credit amount, annuity, current debt, recent application count, historical delinquency/payment summaries, credit/disbursement type |
| person | case_id + num_group1 | occupation income and distinct housing/income types |
| bureau | case_id + num_group1 | debt, overdue debt, delinquency, credit quantity |
| previous | case_id + num_group1 + num_group2 | distinct contract types within each prior-application parent |

The source `feature_definitions.csv` is authoritative for individual raw field descriptions. The selected names are explicit in `configs/kaggle.yaml`. This initial feature set avoids direct sex, birthdate and marital-status inputs; that does not establish fairness or remove all proxies.

`case_id`, `target`, `WEEK_NUM`, `MONTH`, decision date and group indices are excluded from predictors. Numeric histories emit mean/min/max; category histories emit distinct counts; sources emit record counts and availability. Depth-two summaries are parent-balanced. Zero coverage differs from an observed numeric zero. Unknown categories are handled using training-fitted encoding.

## Extending a source

Add its file glob, depth, unique keys, selected columns and whether it is external. Validate its actual grain first. Files sharing a glob are vertically concatenated; horizontally split tables need separate source entries. The configuration can use `case_modulus: 10` for a deterministic 10% engineering subset, but the default is 1 (all applications). The input hash and selected schema both participate in the data version.

Source arrival timestamps and label-availability dates are not provided by this prototype. Do not manufacture a point-in-time guarantee or recency window from an ambiguous date field. Unexpected duplicates, orphan IDs and missing required sources stop processing; optional-source absence is explicit.

## BI tables

- `risk_summary_weekly`: one row per week, model version, mode, observation cutoff and sample scope. Counts/risk bands include all scored rows; actual default rate is null until maturity.
- `model_performance_weekly`: same grain; weekly AUC, Gini and Brier only for mature labels and sufficiently sized two-class weeks.
- `data_quality_weekly`: one row per feature/week, with missingness, source coverage where applicable, and PSI against development data.
- `calibration_weekly`: probability-bin counts, mean score and observed default rate for mature labels.
- `pipeline_runs`: one row per execution, including failures, timing, rows and process peak memory.

Do not sum AUC/Gini or average weekly AUC to claim an overall score. Overall metrics are calculated directly from per-application predictions in Python. Model modes must remain separate to avoid double-counting applications.
