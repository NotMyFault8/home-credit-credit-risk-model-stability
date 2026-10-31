"""Command-line entry point; run `home-credit --help`."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .common import now, write_parquet
from .data import FEATURE_VERSION, build_features, incremental, ingest
from .model import activate, load_active, rollback, score_details
from .synthetic import generate
from .workflow import prefect_run, run


def main():
    parser = argparse.ArgumentParser(description="Home Credit Risk & Stability Platform")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run the complete fictional-data demonstration")
    demo.add_argument("--output", default="artifacts/demo")
    demo.add_argument("--per-week", type=int, default=100)
    execute = sub.add_parser("run", help="Run configured Parquet sources")
    execute.add_argument("--config", required=True)
    execute.add_argument("--output", default="artifacts/run")
    execute.add_argument("--prefect", action="store_true")
    execute.add_argument("--mlflow-uri")
    batch = sub.add_parser("score", help="Score a contracted Gold feature table")
    batch.add_argument("--input", required=True)
    batch.add_argument("--output", required=True)
    batch.add_argument("--registry", required=True)
    batch.add_argument("--unavailable", nargs="*", default=[])
    backfill = sub.add_parser(
        "backfill", help="Replace selected weeks from a complete source snapshot"
    )
    backfill.add_argument("--config", required=True)
    backfill.add_argument("--existing", required=True)
    backfill.add_argument("--output", required=True)
    backfill.add_argument("--weeks", type=int, nargs="+", required=True)
    for command in ["rollback", "activate"]:
        child = sub.add_parser(command)
        child.add_argument("--registry", required=True)
        if command == "activate":
            child.add_argument(
                "--versions",
                required=True,
                help="JSON file containing full/internal version pair",
            )
    args = parser.parse_args()
    if args.command == "demo":
        cfg = generate(Path(args.output) / "synthetic", per_week=args.per_week)
        run(cfg, args.output)
    elif args.command == "run":
        if args.prefect:
            prefect_run(args.config, args.output)
        else:
            run(args.config, args.output, args.mlflow_uri)
    elif args.command == "score":
        frame = pd.read_parquet(args.input)
        bundle = load_active(args.registry, args.unavailable)
        predictions = score_details(bundle, frame)
        result = frame[["case_id"]].copy()
        result["raw_score"] = predictions["raw_score"]
        result["calibrated_score"] = predictions["calibrated_score"]
        result["score"] = result["raw_score"]
        result["model_version"], result["feature_version"] = (
            bundle["version"],
            FEATURE_VERSION,
        )
        result["calibrator_version"] = bundle.get("calibration", {}).get("version")
        result["scored_at"] = now()
        result["quality_status"] = "degraded" if bundle["mode"] == "internal" else "ok"
        write_parquet(result, args.output)
    elif args.command == "backfill":
        cfg, tables, _ = ingest(args.config, Path(args.output).parent / "backfill")
        rebuilt, existing = build_features(cfg, tables), pd.read_parquet(args.existing)
        ids = set(existing.loc[existing.WEEK_NUM.isin(args.weeks), "case_id"]) | set(
            rebuilt.loc[rebuilt.WEEK_NUM.isin(args.weeks), "case_id"]
        )
        write_parquet(incremental(existing, rebuilt, ids), args.output)
    elif args.command == "rollback":
        rollback(args.registry)
    elif args.command == "activate":
        activate(args.registry, json.loads(Path(args.versions).read_text()))


if __name__ == "__main__":
    main()
