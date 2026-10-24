# Architecture and design decisions

## Problem and outcome

The project demonstrates how to create stable application-level credit-risk features from multiple source grains, validate performance on later weeks, and expose measured behavior through reproducible operational and BI outputs.

The unit of analysis is `case_id`, an application identifier. It is not a persistent customer identifier. Sources are linked only through their declared keys. Every join must preserve the base application's cardinality.

## Data flow

1. Read the YAML source manifest, hash input files, and copy immutable source snapshots to `data/bronze/<source>/<hash>`.
2. Project selected columns, optionally filter a deterministic `case_id` modulus, enforce non-null unique grain keys and reject orphan applications.
3. Normalize explicitly selected dates, persist Silver tables, and record row counts, types and hashes.
4. Aggregate depth-one records per application. For depth two, aggregate within `(case_id, num_group1)` first, then average parent summaries. This gives each parent equal weight. Parent count and average child count remain separate features.
5. Join one-to-one, preserve explicit source-availability features, and validate Gold with DuckDB.
6. Fit preprocessing separately for each training fold. Compare logistic regression and LightGBM with the same temporal windows. Train selected full/internal candidates on development weeks only.
7. Evaluate the final four weeks once and record performance. Activate a compatible pair only after computable metrics exist.
8. Score via a shared function. Explicit external outages route to the internal model. Replay monitoring hides labels until simulated maturity.

## Storage and recovery

Content hashes prevent duplicate Bronze snapshots. Silver and Gold use atomic replace semantics. Run IDs are unique; repeat execution records another run without duplicating Gold applications. Backfill recomputes selected applications from their complete available history and replaces their rows. This is historical replay, not source-system CDC.

Model pairs have an atomic active pointer and a previous pointer. Only trusted, locally produced joblib artifacts should be loaded. Filesystem registry writes assume a single writer; production multi-worker promotion would need locking or a transactional registry.

## Resource strategy

This is a single-machine implementation. Polars projects and filters Parquet before materializing selected data; aggregation and sklearn training operate in memory. The tested configuration uses all base rows with a small, documented field set. An unrestricted all-column/all-history run is not a claim of this implementation. Scale to distributed processing only after measuring memory and runtime on the intended data volume.

## Optional integrations

`pip install -e '.[ops]'` adds Prefect and MLflow. `home-credit run --prefect ...` wraps execution with two retries. `--mlflow-uri <uri>` records chosen model metadata and holdout metrics for the direct runner. Local JSON artifacts remain the canonical, dependency-light audit trail. A managed server, cloud deployment, authentication gateway, and scheduling infrastructure are not provisioned by this repository.
