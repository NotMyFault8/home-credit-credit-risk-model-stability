"""Measure API parity, outage fallback, failures, backfill, and real rollback."""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi.testclient import TestClient

from home_credit.api import create_app
from home_credit.common import write_json
from home_credit.data import incremental
from home_credit.model import activate, load_active, rollback, score
from home_credit.workflow import run


def verify(root):
    root = Path(root)
    registry = root / "registry"
    gold = pd.read_parquet(root / "data/gold.parquet")
    bundle = load_active(registry)
    sample = gold.iloc[:100].copy()
    payload = {
        "feature_version": bundle["feature_version"],
        "applications": json.loads(
            sample[["case_id"] + bundle["columns"]].to_json(orient="records")
        ),
    }
    client = TestClient(create_app(registry))
    durations = []
    for _ in range(30):
        started = time.perf_counter()
        response = client.post("/score", json=payload)
        response.raise_for_status()
        durations.append(time.perf_counter() - started)
    batch = score(bundle, sample)
    api = [p["score"] for p in response.json()["predictions"]]
    delta = float(np.max(np.abs(batch - api)))
    assert delta < 1e-9
    payload["unavailable_sources"] = bundle["external_sources"]
    outage = client.post("/score", json=payload)
    outage.raise_for_status()
    assert outage.json()["mode"] == "internal"
    internal = load_active(registry, bundle["external_sources"])
    np.testing.assert_allclose(
        [p["score"] for p in outage.json()["predictions"]], score(internal, sample)
    )
    changed = gold.copy()
    column = next(c for c in bundle["columns"] if pd.api.types.is_numeric_dtype(gold[c]))
    affected = gold.case_id.iloc[:100]
    changed.loc[changed.case_id.isin(affected), column] = 17.5
    pd.testing.assert_frame_equal(incremental(gold, changed, affected), changed)
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        copy = temp / "registry"
        shutil.copytree(registry, copy)
        versions = json.loads((copy / "active.json").read_text())
        alternate = {}
        for mode, version in versions.items():
            name = version + "-recovery-drill"
            shutil.copytree(copy / version, copy / name)
            model = joblib.load(copy / name / "model.joblib")
            model["version"] = name
            joblib.dump(model, copy / name / "model.joblib")
            alternate[mode] = name
        activate(copy, alternate)
        assert load_active(copy)["version"] == alternate["full"]
        rollback(copy)
        assert load_active(copy)["version"] == versions["full"]
        np.testing.assert_allclose(score(load_active(copy), sample), batch)
        bad = temp / "missing.yaml"
        bad.write_text(
            yaml.safe_dump(
                {
                    "data_root": ".",
                    "tables": {"base": {"glob": "absent.parquet", "keys": ["case_id"], "depth": 0}},
                }
            )
        )
        try:
            run(bad, temp / "failure")
        except ValueError:
            failure = json.loads(next((temp / "failure/runs").glob("*.json")).read_text())
            assert failure["status"] == "failed"
        else:
            raise AssertionError("Missing source should fail")
    evidence = {
        "api_batch_max_abs_difference": delta,
        "requests": 30,
        "rows_per_request": 100,
        "in_process_p95_ms": float(np.percentile(durations, 95) * 1000),
        "in_process_rows_per_second": float(3000 / sum(durations)),
        "measurement_scope": "FastAPI TestClient, serial warm requests; includes bundle loading, excludes network",
        "fallback_mode": outage.json()["mode"],
        "fallback_parity": True,
        "incremental_matches_full_rebuild": True,
        "distinct_version_rollback": True,
        "failed_run_audited": True,
    }
    write_json(root / "verification.json", evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    verify(parser.parse_args().output)
