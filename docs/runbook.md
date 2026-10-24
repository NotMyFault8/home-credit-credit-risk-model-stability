# Operations runbook

## Repeat a run

```bash
home-credit run --config configs/kaggle.yaml --output artifacts/competition
```

The same input reuses Bronze snapshots and replaces Gold atomically. Run history is append-only. If a source fails validation, inspect `artifacts/competition/runs/*.json`, repair the source/configuration, then rerun. Existing active models remain available until a complete new pair passes the gate.

## Backfill weeks

```bash
home-credit backfill --config configs/kaggle.yaml \
  --existing artifacts/competition/data/gold.parquet \
  --weeks 80 81 --output artifacts/backfilled.parquet
```

Provide the complete relevant source history. This command replaces affected case IDs, including removals; it is not incremental event ingestion. The test suite checks changed rows, deletions and insertions against full reconstruction.

## External-source outage

```bash
home-credit score --input artifacts/competition/data/gold.parquet \
  --registry artifacts/competition/registry --unavailable bureau \
  --output artifacts/degraded-scores.parquet
```

The API accepts `unavailable_sources: ["bureau"]`. Both batch and API use the same scorer and return the active model version. Missing full-model contracted fields without an explicit outage produce an error; they are not silently accepted.

## Promote and roll back

```bash
home-credit activate --registry artifacts/competition/registry \
  --versions artifacts/competition/registry/candidate.json
home-credit rollback --registry artifacts/competition/registry
```

Promotion stores the previous full/internal pair. Check evaluation and data-quality reports before manual promotion. A first activation has no rollback target. Activation validates both bundle modes and feature versions. Keep registry artifacts local and trusted.

## API

```bash
HOME_CREDIT_REGISTRY=artifacts/competition/registry \
HOME_CREDIT_TIMEOUT_SECONDS=10 \
uvicorn home_credit.api:app --host 127.0.0.1 --port 8000
```

`GET /health`: 503 until an active registry exists. `POST /score`: contract version, 1–1,000 application feature records, unique non-null integer case IDs, optional unavailable-source list. Contract errors are 422; incompatible feature versions 409; deadline exceeded 504. Use the generated `/docs` for examples. This is a local prototype; authenticate and rate-limit before exposing it to a network.

## Docker

```bash
docker build -t home-credit .
docker run --rm -p 127.0.0.1:8000:8000 \
  -v "$PWD/artifacts/competition/registry:/app/artifacts/registry:ro" home-credit
```

## Optional orchestration and tracking

```bash
pip install -e '.[ops]'
home-credit run --config configs/kaggle.yaml --output artifacts/competition --prefect
home-credit run --config configs/kaggle.yaml --output artifacts/competition --mlflow-uri ./mlruns
```

The Prefect wrapper retries a failed run twice. Each attempt gets its own run record. For local replay monitoring, `report(..., as_of_week=..., label_delay_weeks=...)` explicitly controls label maturity. Rerun the report after moving the observation cutoff; the fixed holdout scores do not change.
