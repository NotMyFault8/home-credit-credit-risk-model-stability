# Project design

**Goal:** build an auditable historical credit-risk prototype that demonstrates data engineering and machine learning engineering.

**Questions:** Can multiple source grains be aggregated without multiplying applications? Does the selected model generalize to later weeks? Can external-source loss and failed runs be recovered? Can reports distinguish population changes, model changes and label delay?

**Method:** explicit source manifests → immutable Bronze snapshots → grain-aware Silver validation → parent-balanced Gold features → rolling temporal validation → versioned full/internal models → batch/API scoring → replay monitoring → Python-generated chart and CSV summaries.

**Outcomes:** measured real-data results, reproducible synthetic execution, automated acceptance tests, operational verification, and compact reporting artifacts. See the [results report](../reports/RESULTS.md) and [reporting guide](../reports/REPORTING.md) for outputs, and the [acceptance matrix](acceptance.md) for evidence and limitations.

The original handoff was a design proposal, not evidence of implementation or results. The implementation chooses a local filesystem model registry, Python reporting, and optional Prefect/MLflow rather than provisioning managed infrastructure. It implements a documented subset of competition features across all base applications. No cloud service or live bank integration is implied.
