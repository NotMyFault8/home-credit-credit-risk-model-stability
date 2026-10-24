"""Single-machine orchestration with run records and optional Prefect scheduling."""

import json
import platform
import resource
import time
import uuid
from pathlib import Path

import duckdb
import pandas as pd

from .common import now, write_json, write_parquet
from .data import build_features, ingest
from .evaluation import evaluate, temporal_splits
from .model import activate, train
from .monitoring import report


def run(config, output, mlflow_uri=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started, run_id = time.perf_counter(), uuid.uuid4().hex[:12]
    record = {"run_id": run_id, "started_at": now(), "status": "running", "rows": 0}
    try:
        cfg, tables, manifest = ingest(config, output / "data")
        gold = build_features(cfg, tables)
        write_parquet(gold, output / "data/gold.parquet")
        connection = duckdb.connect()
        connection.register("gold", gold)
        counts = connection.execute("select count(*), count(distinct case_id) from gold").fetchone()
        connection.close()
        assert counts[0] == counts[1], "SQL gold grain validation failed"
        external = [name for name, spec in cfg["tables"].items() if spec.get("external")]
        versions, predictions, experiments = train(
            gold, manifest, output / "registry", external, mlflow_uri
        )
        # Explicit engineering-demo release gate, not approval for lending use.
        metrics = {mode: evaluate(group)[0] for mode, group in predictions.groupby("mode")}
        if any(m["auc"] is None or m["stability"] is None for m in metrics.values()):
            raise ValueError("Release gate failed: undefined holdout evaluation")
        activate(output / "registry", versions)
        write_parquet(predictions, output / "predictions.parquet")
        _, development, _ = temporal_splits(gold)
        report(gold, predictions, gold.loc[development], output / "bi")
        write_json(
            output / "evaluation.json",
            {
                "synthetic": manifest["synthetic"],
                "holdout": metrics,
                "versions": versions,
                "validation_experiments": experiments,
                "label_delay_weeks": 2,
                "label_delay_is_simulated": True,
            },
        )
        record.update(status="success", rows=len(gold), data_version=manifest["data_version"])
        return output
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        raise
    finally:
        record.update(
            finished_at=now(),
            duration_seconds=time.perf_counter() - started,
            hardware=platform.platform(),
            peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 * 1024 if platform.system() == "Darwin" else 1024),
        )
        write_json(output / "runs" / f"{run_id}.json", record)
        runs = [json.loads(p.read_text()) for p in sorted((output / "runs").glob("*.json"))]
        (output / "bi").mkdir(exist_ok=True)
        pd.DataFrame(runs).to_csv(output / "bi/pipeline_runs.csv", index=False)


def prefect_run(config, output):
    from prefect import flow

    return flow(name="home-credit-replay", retries=2, retry_delay_seconds=5)(run)(config, output)
