"""Export aggregate results only; never copy source rows or per-case predictions."""

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from home_credit.common import write_json
from home_credit.monitoring import report


def publish(source, output, label):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    evaluation = json.loads((source / "evaluation.json").read_text())
    gold = pd.read_parquet(source / "data/gold.parquet")
    pred = pd.read_parquet(source / "predictions.parquet")
    # Retrospective report: all archived holdout labels now mature. Replay exports
    # in the original run retain the original observation cutoff and pending labels.
    report(
        gold,
        pred,
        gold[gold.WEEK_NUM < pred.WEEK_NUM.min()],
        output / "bi",
        as_of_week=int(pred.WEEK_NUM.max()) + 2,
    )
    shutil.copy2(source / "bi/pipeline_runs.csv", output / "bi/pipeline_runs.csv")
    for filename in ["evaluation.json", "verification.json", "ablations.json", "ablations.csv"]:
        if (source / filename).exists():
            shutil.copy2(source / filename, output / filename)
    manifest = json.loads((source / "data/manifest.json").read_text())
    write_json(
        output / "dataset-summary.json",
        {
            "label": label,
            "applications": len(gold),
            "week_min": int(gold.WEEK_NUM.min()),
            "week_max": int(gold.WEEK_NUM.max()),
            "holdout_min_week": int(pred.WEEK_NUM.min()),
            "holdout_max_week": int(pred.WEEK_NUM.max()),
            "holdout_applications": int(pred[pred["mode"] == "full"].shape[0]),
            "data_version": manifest["data_version"],
            "feature_version": manifest["feature_version"],
            "source_rows": {k: v["rows"] for k, v in manifest["sources"].items()},
            "synthetic": evaluation["synthetic"],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    publish(args.source, args.output, args.label)
