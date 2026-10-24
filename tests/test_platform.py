import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from home_credit.api import create_app
from home_credit.data import aggregate, build_features, incremental, ingest
from home_credit.evaluation import evaluate, temporal_splits
from home_credit.model import activate, load_active, rollback, score
from home_credit.monitoring import psi, report
from home_credit.synthetic import generate
from home_credit.workflow import run


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("demo")
    config = generate(root / "source", per_week=80)
    run(config, root / "output")
    return root, config


def test_ingest_idempotent_and_grain(demo):
    root, config = demo
    a, tables, first = ingest(config, root / "check")
    _, _, second = ingest(config, root / "check")
    assert first["data_version"] == second["data_version"]
    gold = build_features(a, tables)
    assert gold.case_id.is_unique and len(gold) == len(tables["base"])
    assert len(list((root / "check/bronze/base").iterdir())) == 1
    pd.testing.assert_frame_equal(incremental(gold, gold, gold.case_id[:100]), gold)


def test_depth_two_parent_weighting():
    data = pd.DataFrame(
        {
            "case_id": [1, 1, 1, 1],
            "num_group1": [0, 1, 1, 1],
            "num_group2": [0, 0, 1, 2],
            "amountA": [100, 0, 0, 0],
        }
    )
    assert aggregate(data, {"depth": 2}).iloc[0]["amountA__mean"] == 50


def test_duplicate_and_orphan_rejected(tmp_path):
    cfg = generate(tmp_path / "source", per_week=2)
    path = tmp_path / "source/history.parquet"
    frame = pd.read_parquet(path)
    pd.concat([frame, frame.iloc[:1]]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        ingest(cfg, tmp_path / "out")
    frame.loc[0, "case_id"] = 999999
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="orphan"):
        ingest(cfg, tmp_path / "out")


def test_stability_formula_and_undefined():
    frame = pd.DataFrame(
        {
            "WEEK_NUM": np.repeat([1, 2, 3], 20),
            "target": [0, 1] * 30,
            "score": [0.1, 0.9] * 30,
        }
    )
    metrics, _ = evaluate(frame)
    assert metrics["stability"] == pytest.approx(1)
    frame.loc[frame.WEEK_NUM == 2, "target"] = 0
    assert evaluate(frame)[0]["stability"] is None


def test_time_boundaries(demo):
    root, _ = demo
    frame = pd.read_parquet(root / "output/data/gold.parquet")
    folds, development, test = temporal_splits(frame)
    for tr, va in folds:
        assert frame.loc[tr, "WEEK_NUM"].max() < frame.loc[va, "WEEK_NUM"].min()
        assert not (va & test).any()
    assert not (development & test).any()


def test_api_batch_parity_fallback_and_contract(demo):
    root, _ = demo
    registry = root / "output/registry"
    frame = pd.read_parquet(root / "output/data/gold.parquet").iloc[:3]
    bundle = load_active(registry)
    assert not {"case_id", "target", "WEEK_NUM"} & set(bundle["columns"])
    request = {
        "feature_version": "aggregate-v1",
        "applications": json.loads(
            frame[["case_id"] + bundle["columns"]].to_json(orient="records")
        ),
    }
    client = TestClient(create_app(registry))
    response = client.post("/score", json=request)
    assert response.status_code == 200, response.text
    np.testing.assert_allclose(
        [p["score"] for p in response.json()["predictions"]], score(bundle, frame)
    )
    request["unavailable_sources"] = ["bureau"]
    fallback = client.post("/score", json=request)
    assert fallback.json()["mode"] == "internal"
    internal = load_active(registry, ["bureau"])
    np.testing.assert_allclose(
        [p["score"] for p in fallback.json()["predictions"]], score(internal, frame)
    )
    request["feature_version"] = "wrong"
    assert client.post("/score", json=request).status_code == 409
    request["feature_version"] = "aggregate-v1"
    request["applications"][0]["incomeA"] = "bad"
    assert client.post("/score", json=request).status_code == 422


def test_rollback(demo):
    root, _ = demo
    registry = root / "output/registry"
    versions = json.loads((registry / "active.json").read_text())
    activate(registry, versions)
    rollback(registry)
    assert json.loads((registry / "active.json").read_text()) == versions


def test_label_delay_and_psi(demo):
    root, _ = demo
    gold = pd.read_parquet(root / "output/data/gold.parquet")
    pred = pd.read_parquet(root / "output/predictions.parquet")
    tables = report(gold, pred, gold[gold.WEEK_NUM < 28], root / "monitor", as_of_week=31)
    risk = tables["risk_summary_weekly"]
    assert risk.loc[risk.WEEK_NUM >= 30, "default_rate"].isna().all()
    assert risk.loc[risk.WEEK_NUM < 30, "default_rate"].notna().all()
    assert psi(np.arange(100), np.arange(100)) == pytest.approx(0)
    assert psi(np.arange(100), np.arange(100) + 100) > 0.2


def test_incremental_changed_and_removed():
    old = pd.DataFrame({"case_id": [1, 2, 3], "value": [10, 20, 30]})
    new = pd.DataFrame({"case_id": [1, 2, 4], "value": [10, 25, 40]})
    pd.testing.assert_frame_equal(incremental(old, new, [2, 3, 4]), new)


def test_schema_change_blocks_downstream(tmp_path):
    cfg = generate(tmp_path / "source", per_week=2)
    ingest(cfg, tmp_path / "out")
    path = tmp_path / "source/base.parquet"
    data = pd.read_parquet(path)
    data["incomeA"] = data.incomeA.astype(str)
    data.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="schema changed"):
        ingest(cfg, tmp_path / "out")


def test_backfill_is_idempotent():
    old = pd.DataFrame({"case_id": [1, 2], "value": [1, 2]})
    updated = pd.DataFrame({"case_id": [1, 2], "value": [1, 7]})
    once = incremental(old, updated, [2])
    pd.testing.assert_frame_equal(incremental(once, updated, [2]), once)


def test_api_rejects_fractional_id_and_unknown_source(demo):
    root, _ = demo
    registry = root / "output/registry"
    bundle = load_active(registry)
    data = pd.read_parquet(root / "output/data/gold.parquet").iloc[:1]
    records = json.loads(data[["case_id"] + bundle["columns"]].to_json(orient="records"))
    client = TestClient(create_app(registry))
    payload = {"feature_version": "aggregate-v1", "applications": records}
    records[0]["case_id"] = 0.5
    assert client.post("/score", json=payload).status_code == 422
    records[0]["case_id"] = 1
    payload["unavailable_sources"] = ["typo"]
    assert client.post("/score", json=payload).status_code == 422


def test_api_unavailable_registry(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/health").status_code == 503
    assert (
        client.post(
            "/score", json={"feature_version": "aggregate-v1", "applications": [{"case_id": 1}]}
        ).status_code
        == 503
    )


def test_api_timeout(demo, monkeypatch):
    import time

    import home_credit.api as api_module

    root, _ = demo

    def slow(*args):
        time.sleep(0.1)
        return None

    monkeypatch.setattr(api_module, "load_active", slow)
    monkeypatch.setenv("HOME_CREDIT_TIMEOUT_SECONDS", "0.01")
    client = TestClient(create_app(root / "output/registry"))
    assert (
        client.post(
            "/score", json={"feature_version": "aggregate-v1", "applications": [{"case_id": 1}]}
        ).status_code
        == 504
    )
