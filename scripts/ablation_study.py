"""Development-only ablations; never inspect or use the final holdout."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from home_credit.common import write_json
from home_credit.evaluation import evaluate, temporal_splits
from home_credit.model import clean, features, pipeline


def study(output):
    output = Path(output)
    gold = pd.read_parquet(output / "data/gold.parquet")
    splits, _, _ = temporal_splits(gold)
    all_columns = features(gold)
    rows = []
    for variant in ["static_only", "history", "stable_missingness", "time_weighted"]:
        for fold, (tr, va) in enumerate(splits):
            cols = (
                [c for c in all_columns if c.startswith("static__")]
                if variant == "static_only"
                else all_columns
            )
            if variant == "stable_missingness":
                # Selection only sees the training window; missingness variation <= 10pp.
                missing = gold.loc[tr].groupby("WEEK_NUM")[cols].agg(lambda s: s.isna().mean())
                cols = [c for c in cols if missing[c].max() - missing[c].min() <= 0.1]
            if not cols:
                continue
            X = clean(gold, cols)
            model = pipeline(X.loc[tr], "lightgbm")
            params = {}
            if variant == "time_weighted":
                age = gold.loc[tr, "WEEK_NUM"].max() - gold.loc[tr, "WEEK_NUM"]
                weights = np.power(0.5, age / 26)
                params["model__sample_weight"] = weights / weights.mean()
            model.fit(X.loc[tr], gold.loc[tr, "target"], **params)
            pred = gold.loc[va, ["case_id", "WEEK_NUM", "target"]].copy()
            pred["score"] = model.predict_proba(X.loc[va])[:, 1]
            metrics, _ = evaluate(pred)
            rows.append({"variant": variant, "fold": fold, "features": len(cols), **metrics})
    write_json(
        output / "ablations.json",
        {"scope": "development validation only; no promotion or holdout tuning", "results": rows},
    )
    pd.DataFrame(rows).to_csv(output / "ablations.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    study(parser.parse_args().output)
