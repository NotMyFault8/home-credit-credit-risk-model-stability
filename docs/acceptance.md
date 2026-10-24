# Acceptance evidence

| Requirement | Evidence | Status |
|---|---|---|
| One Gold row per application | Data tests, join validation, DuckDB distinct count; 1,526,659 real rows | Passed |
| Idempotent snapshots and features | Repeated ingestion hash check, snapshot count, repeated replacement tests | Passed |
| Incremental/full equivalence | Insert/change/delete fixtures and changed-row real-data drill | Passed |
| Consistent temporal comparisons | Three forward folds; shared windows for baseline and LightGBM | Passed |
| Time stability | Per-week Gini, trend, residual variation and exact composite metric | Passed |
| Batch/API consistency | Zero maximum difference on real-data verification sample | Passed |
| Recovery | External outage, audited failed run, distinct-version rollback | Passed |
| Performance measurement | Runtime, RSS, in-process serial P95 and throughput recorded | Passed within stated scope |
| Reporting consistency | Python-generated chart and CSV summaries use explicit week, model, maturity and sample-scope fields | Passed |
| Reproducible small flow | Demo, tests and optional Prefect/MLflow runs | Passed |

The final competition configuration uses all applications and a curated field subset; it is not an exhaustive feature search. Tests use synthetic data and require no competition download. CI reruns lint, tests and the small demonstration. The Docker image builds and starts successfully. Its live health endpoint returned ready and its HTTP scoring matched local batch predictions exactly on three verification records. CI also runs the fictional end-to-end flow inside the container.

Not claimed: interactive dashboards, Kubernetes/cloud deployment, real CDC, live banking integration, production fairness certification, business profit, or competition ranking.
