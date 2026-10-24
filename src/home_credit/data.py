"""Manifest-driven ingestion, grain checks, and application-level aggregation."""

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import polars as pl
import yaml

from .common import digest, now, write_json, write_parquet

FEATURE_VERSION = "aggregate-v1"
RESERVED = {
    "case_id",
    "target",
    "WEEK_NUM",
    "MONTH",
    "date_decision",
    "num_group1",
    "num_group2",
}


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def ingest(config_path, output):
    config_path, output = Path(config_path), Path(output)
    cfg = load_config(config_path)
    root = (config_path.parent / cfg["data_root"]).resolve()
    tables, sources = {}, {}
    for name, spec in cfg["tables"].items():
        paths = sorted(root.glob(spec["glob"]))
        if not paths:
            if spec.get("optional", False):
                continue
            raise ValueError(f"Missing required source: {name}")
        hashes = [digest(p) for p in paths]
        snapshot = hashlib.sha256("".join(hashes).encode()).hexdigest()[:16]
        dest = output / "bronze" / name / snapshot
        dest.mkdir(parents=True, exist_ok=True)
        frames = []
        for p, checksum in zip(paths, hashes):
            copy = dest / p.name
            if not copy.exists():
                shutil.copy2(p, copy)
            elif digest(copy) != checksum:
                raise ValueError(f"Corrupt snapshot: {copy}")
            scan = pl.scan_parquet(copy)
            if spec.get("columns"):
                scan = scan.select(list(dict.fromkeys(spec["keys"] + spec["columns"])))
            if cfg.get("case_modulus", 1) > 1:
                scan = scan.filter(pl.col("case_id") % cfg["case_modulus"] == 0)
            frames.append(scan.collect())
        frame = pl.concat(frames, how="diagonal_relaxed").to_pandas()
        keys = spec["keys"]
        if not set(keys) <= set(frame):
            raise ValueError(f"{name}: missing grain keys {keys}")
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():
            raise ValueError(f"{name}: null or duplicate grain keys {keys}")
        if name != "base" and ("target" in frame or "WEEK_NUM" in frame):
            raise ValueError(f"{name}: reserved outcome/time fields in history")
        for col in frame:
            if col.endswith("D") or col == "date_decision":
                frame[col] = pd.to_datetime(frame[col], errors="raise")
        sources[name] = {
            "snapshot": snapshot,
            "files": [p.name for p in paths],
            "sha256": hashes,
            "rows": len(frame),
            "schema": {c: str(t) for c, t in frame.dtypes.items()},
            "case_modulus": cfg.get("case_modulus", 1),
        }
        tables[name] = frame
    if "base" not in tables:
        raise ValueError("A base table is required")
    base = tables["base"]
    if not {"case_id", "date_decision", "WEEK_NUM"} <= set(base):
        raise ValueError("Base requires case_id, date_decision, WEEK_NUM")
    if base[["case_id", "date_decision", "WEEK_NUM"]].isna().any().any():
        raise ValueError("Null base identifiers or time fields")
    if "target" in base and not base.target.isin([0, 1]).all():
        raise ValueError("Training target must be binary and non-null")
    previous_path = output / "manifest.json"
    warnings = []
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        for name, current in sources.items():
            old = previous["sources"].get(name)
            if old and old["schema"] != current["schema"]:
                raise ValueError(f"{name}: schema changed; use a new output version after review")
            if old and old["rows"] and not 0.5 <= current["rows"] / old["rows"] <= 1.5:
                warnings.append(f"{name}: row volume changed by more than 50%")
        for name in set(previous["sources"]) - set(sources):
            warnings.append(f"{name}: optional source is now unavailable")
    for name, frame in tables.items():
        if not frame.case_id.isin(base.case_id).all():
            raise ValueError(f"{name}: orphan case_id")
        write_parquet(frame, output / "silver" / f"{name}.parquet")
    version = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:16]
    manifest = {
        "data_version": version,
        "feature_version": FEATURE_VERSION,
        "created_at": now(),
        "sources": sources,
        "quality_warnings": warnings,
        "synthetic": cfg.get("synthetic", False),
    }
    write_json(output / "manifest.json", manifest)
    return cfg, tables, manifest


def aggregate(frame, spec):
    """Depth 2: summarize each parent first, then give parents equal weight."""
    keys = ["case_id", "num_group1"] if spec["depth"] == 2 else ["case_id"]
    numeric = [c for c in frame.select_dtypes(include="number") if c not in RESERVED]
    dates = list(frame.select_dtypes(include=["datetime", "datetimetz"]).columns)
    categories = [c for c in frame if c not in RESERVED and c not in numeric + dates]
    ops = {c: ["mean", "min", "max"] for c in numeric}
    ops.update({c: ["min", "max"] for c in dates})
    ops.update({c: ["nunique"] for c in categories})
    grouped = frame.groupby(keys, dropna=False)
    result = grouped.agg(ops) if ops else pd.DataFrame(index=grouped.size().index)
    result.columns = ["__".join(c) for c in result.columns]
    result["records"] = grouped.size()
    result = result.reset_index()
    if spec["depth"] == 2:
        second = {
            c: ("min" if c.endswith("__min") else "max")
            if pd.api.types.is_datetime64_any_dtype(result[c])
            else "mean"
            for c in result
            if c not in keys
        }
        result = result.groupby("case_id").agg(second).reset_index()
        result["parents"] = (
            frame.groupby("case_id").num_group1.nunique().reindex(result.case_id).to_numpy()
        )
    return result


def build_features(cfg, tables):
    gold = tables["base"].copy()
    for name, spec in cfg["tables"].items():
        if name == "base" or name not in tables:
            continue
        frame = tables[name]
        if spec["depth"]:
            frame = aggregate(frame, spec)
        frame = frame.rename(columns={c: f"{name}__{c}" for c in frame if c != "case_id"})
        gold = gold.merge(frame, on="case_id", how="left", validate="one_to_one")
        gold[f"{name}__available"] = gold.case_id.isin(frame.case_id).astype(int)
    for col in list(gold):
        if col != "date_decision" and pd.api.types.is_datetime64_any_dtype(gold[col]):
            gold[col + "__days_before_decision"] = (gold.date_decision - gold[col]).dt.days
            gold = gold.drop(columns=col)
    if len(gold) != len(tables["base"]) or gold.case_id.duplicated().any():
        raise ValueError("Gold grain violation")
    return gold.sort_values("case_id").reset_index(drop=True)


def incremental(existing, rebuilt, affected_ids):
    """Replace affected applications, including removals; caller supplies full history."""
    ids = set(affected_ids)
    return (
        pd.concat(
            [existing[~existing.case_id.isin(ids)], rebuilt[rebuilt.case_id.isin(ids)]],
            ignore_index=True,
        )
        .sort_values("case_id")
        .reset_index(drop=True)
    )
