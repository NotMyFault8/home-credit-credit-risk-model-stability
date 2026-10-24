"""Deterministic fictional applications for engineering demonstrations only."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def generate(root, weeks=32, per_week=100, seed=42):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = weeks * per_week
    week = np.repeat(np.arange(weeks), per_week)
    income = rng.lognormal(10, 0.5, n)
    debt = rng.uniform(0, 2, n)
    external = rng.normal(size=n)
    probability = 1 / (1 + np.exp(-(-2.4 + 1.2 * debt - 0.5 * external + 0.008 * week)))
    base = pd.DataFrame(
        {
            "case_id": np.arange(n),
            "WEEK_NUM": week,
            "date_decision": pd.Timestamp("2020-01-01") + pd.to_timedelta(week * 7, unit="D"),
            "target": rng.binomial(1, probability),
            "incomeA": income,
            "debt_ratioA": debt,
            "regionM": rng.choice(["north", "south", "west"], n),
        }
    )
    bureau = pd.DataFrame({"case_id": np.arange(n), "bureau_scoreA": external})
    history = pd.DataFrame(
        [
            {
                "case_id": i,
                "num_group1": j,
                "amountA": float(income[i] * rng.uniform(0.01, 0.15)),
                "lateA": int(rng.random() < 0.1 + debt[i] * 0.1),
            }
            for i in range(n)
            for j in range(int(rng.integers(1, 4)))
        ]
    )
    detail = history[["case_id", "num_group1"]].loc[history.index.repeat(2)].copy()
    detail["num_group2"] = np.tile([0, 1], len(history))
    detail["paymentA"] = rng.uniform(20, 200, len(detail))
    for name, frame in [
        ("base", base),
        ("bureau", bureau),
        ("history", history),
        ("payments", detail),
    ]:
        frame.to_parquet(root / f"{name}.parquet", index=False)
    cfg = {
        "data_root": ".",
        "synthetic": True,
        "tables": {
            "base": {"glob": "base.parquet", "depth": 0, "keys": ["case_id"]},
            "bureau": {
                "glob": "bureau.parquet",
                "depth": 0,
                "keys": ["case_id"],
                "external": True,
            },
            "history": {
                "glob": "history.parquet",
                "depth": 1,
                "keys": ["case_id", "num_group1"],
            },
            "payments": {
                "glob": "payments.parquet",
                "depth": 2,
                "keys": ["case_id", "num_group1", "num_group2"],
            },
        },
    }
    path = root / "sources.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path
