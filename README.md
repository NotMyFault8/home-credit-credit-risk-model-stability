# Home Credit Risk & Stability Platform

A reproducible data engineering and machine learning project for **credit-risk prediction across time**. It turns multi-grain Parquet sources into one row per application, evaluates models on later weeks, serves versioned scores, and produces risk, stability, data-quality, and run-health reports.

Built around the [Home Credit — Credit Risk Model Stability competition](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability). This is a historical-data engineering prototype, not a lending decision system.

![Real-data holdout results](reports/model-stability.png)

## Start here

| I want to… | Open |
|---|---|
| Run a complete example in minutes | [Quick start](#quick-start) |
| Understand the design | [Architecture](docs/architecture.md) |
| Inspect measured results and limitations | [Results](reports/RESULTS.md) · [Validation report](reports/MODEL_VALIDATION_REPORT.md) · [Model card](docs/model-card.md) |
| Read the generated charts and metric tables | [Reporting guide](reports/REPORTING.md) |
| Use the competition data | [Data guide](docs/data-dictionary.md) |
| Recover a run or roll back models | [Runbook](docs/runbook.md) |
| Review what was verified | [Acceptance evidence](docs/acceptance.md) |

## Quick start

Python 3.10–3.12. On macOS, LightGBM may need `brew install libomp`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
home-credit demo --output artifacts/demo
pytest -q
```

The demo creates 3,200 **fictional** applications across 32 weeks, ingests four sources, builds features, compares logistic regression and LightGBM, evaluates an untouched final period, activates a full/internal model pair, and writes monitoring tables. No Kaggle account is needed for the demo.

```bash
# Train with locally downloaded competition data:
home-credit run --config configs/kaggle.yaml --output artifacts/competition

# Score application-level features using the same preprocessing as training:
home-credit score --input artifacts/competition/data/gold.parquet \
  --registry artifacts/competition/registry --output artifacts/scores.parquet

# Start the local scoring API; interactive contract at /docs:
HOME_CREDIT_REGISTRY=artifacts/competition/registry \
  uvicorn home_credit.api:app --host 127.0.0.1 --port 8000
```

## What the project implements

- **Data engineering:** content-addressed Bronze snapshots, explicit table grains, orphan and duplicate checks, projected Parquet reads, Silver normalization, parent-balanced depth-two aggregation, and Gold uniqueness checks in Python and DuckDB.
- **Model engineering:** train-only imputation and encoding, three expanding temporal folds, baseline/LightGBM comparison, retrospective final four-week evaluation, weekly AUC/Gini/Brier/log loss/ECE, confidence intervals, the original stability metric, and versioned bundles.
- **Probability calibration:** development temporal out-of-fold predictions compare raw, Platt/sigmoid, and isotonic methods. Full/Internal calibrators are selected and version-bound separately; legacy `score` remains raw while `raw_score` and `calibrated_score` are explicit.
- **Robustness and monitoring:** development-frozen segments, small-sample non-estimable states, numeric/categorical feature PSI, missingness/source coverage, raw/calibrated score drift, traceable example alerts, and simulated label maturity.
- **Operations:** batch/API score parity, separately evaluated internal-source fallback, explicit model activation and rollback, week-scoped backfill, and run manifests.
- **Reporting:** reproducible aggregate CSVs, a generated validation report, Python charts, and the packaged Tableau workbook, alongside Docker and GitHub Actions checks.

```mermaid
flowchart LR
  A[Parquet sources] --> B[Bronze snapshots]
  B --> C[Silver quality checks]
  C --> D[Gold application features]
  D --> E[Forward validation]
  E --> F[Versioned model pair]
  F --> G[Batch / FastAPI]
  G --> H[Replay monitoring]
  H --> I[CSV summaries and charts]
```

## Repository map

```text
src/home_credit/     Pipeline, features, training, API, monitoring, CLI
configs/             Explicit competition source and feature selection
scripts/             Report generation and verification evidence
tests/               Data, temporal validation, API and recovery tests
reports/             Results, charts, aggregate CSV summaries and reporting guide
docs/                Architecture, data contract, model card and runbook
.github/workflows/   Continuous integration
```

Raw data, per-application predictions, serialized models, credentials, and runtime artifacts are excluded from Git. The competition download is local-only. Committed synthetic data and aggregate reports are explicitly labeled.

## Reproducibility and scope

The real-data configuration uses all application rows and a curated feature subset, rather than every available competition column. Data hashes, selected schemas, feature version, source-code fingerprint, parameters, seed, and model versions are recorded. `requirements-lock.txt` records the tested local environment; use the package version ranges for cross-platform installation.

Date availability and label maturity cannot be proven from these historical tables. The two-week label delay is a demonstration assumption. Category histories use distinct counts; date intervals use decision date only when date columns are explicitly selected. The final four weeks are not used to pick a model or calibrator. They have already been inspected and reported, so new results on those weeks are explicitly called retrospective evaluation rather than a newly untouched independent test. Undefined or undersized weeks make relevant metrics unavailable rather than silently disappearing.

Local model activation is an engineering gate (valid evaluation and compatible artifacts), not an automatic approval for real lending. No leaderboard rank, financial benefit, fairness certification, or production deployment is claimed.
