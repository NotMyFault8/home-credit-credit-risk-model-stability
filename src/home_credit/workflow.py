"""Single-machine orchestration with run records and optional Prefect scheduling."""

import json
import platform
import resource
import time
import uuid
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .calibration import calibration_comparison
from .common import now, write_json, write_parquet
from .data import build_features, ingest
from .evaluation import evaluate, temporal_splits
from .model import activate, train
from .monitoring import report
from .segmentation import evaluate_segments, fit_segment_definitions, risk_segment_definition
from .validation_report import generate_validation_report


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
        calibration_rows, calibration_weekly = [], []
        protocols, all_edges = {}, {}
        for mode, group in predictions.groupby("mode"):
            metadata = json.loads(
                (output / "registry" / versions[mode] / "metadata.json").read_text()
            )
            protocol = metadata["calibration"]
            protocols[mode] = protocol
            edges = {}
            for score_type, stored in protocol["score_bin_edges"].items():
                edges[score_type] = np.array(
                    [
                        -np.inf if i == 0 and value is None else np.inf if value is None else value
                        for i, value in enumerate(stored)
                    ]
                )
            all_edges[mode] = edges
            comparison, weekly = calibration_comparison(group, edges)
            comparison.insert(0, "mode", mode)
            comparison.insert(1, "model_version", versions[mode])
            weekly.insert(0, "mode", mode)
            weekly.insert(1, "model_version", versions[mode])
            calibration_rows.append(comparison)
            calibration_weekly.append(weekly)
        report(
            gold,
            predictions,
            gold.loc[development],
            output / "bi",
            score_bin_edges=all_edges,
        )
        pd.concat(calibration_rows, ignore_index=True).to_csv(
            output / "bi/calibration_comparison.csv", index=False
        )
        pd.concat(calibration_weekly, ignore_index=True).to_csv(
            output / "bi/calibration_metrics_weekly.csv", index=False
        )
        definitions = fit_segment_definitions(gold.loc[development])
        for mode, edges in all_edges.items():
            definitions.append(risk_segment_definition(edges["raw_score"], mode))
        write_json(output / "bi/segment_definitions.json", definitions)
        evaluate_segments(gold, predictions, definitions).to_csv(
            output / "bi/segment_performance.csv", index=False
        )
        write_json(
            output / "evaluation.json",
            {
                "synthetic": manifest["synthetic"],
                "holdout": metrics,
                "versions": versions,
                "validation_experiments": experiments,
                "calibration_protocols": protocols,
                "final_evaluation_scope": "retrospective_previously_inspected_holdout",
                "label_delay_weeks": 2,
                "label_delay_is_simulated": True,
            },
        )
        generate_validation_report(output)
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
