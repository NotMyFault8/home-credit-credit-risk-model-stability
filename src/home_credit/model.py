"""Train-only preprocessing, forward validation, versioned bundles and rollback."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .calibration import select_calibrator
from .common import now, write_json
from .data import FEATURE_VERSION, RESERVED
from .evaluation import evaluate, temporal_splits


def pipeline(frame, kind, seed=42):
    num = list(frame.select_dtypes(include="number").columns)
    cat = [c for c in frame if c not in num]
    prep = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="median", keep_empty_features=True),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                num,
            ),
            (
                "category",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                        ),
                        (
                            "encode",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                    ]
                ),
                cat,
            ),
        ]
    )
    estimator = (
        LogisticRegression(max_iter=1000, solver="liblinear", random_state=seed)
        if kind == "baseline"
        else LGBMClassifier(
            n_estimators=120,
            num_leaves=15,
            max_depth=5,
            min_child_samples=30,
            learning_rate=0.04,
            verbosity=-1,
            random_state=seed,
            n_jobs=2,
        )
    )
    return Pipeline([("preprocess", prep), ("model", estimator)])


def features(frame, external_sources=(), internal=False):
    return [
        c
        for c in frame
        if c not in RESERVED
        and not (internal and any(c.startswith(s + "__") for s in external_sources))
    ]


def clean(frame, columns):
    result = frame.reindex(columns=columns).copy()
    for col in columns:
        if not pd.api.types.is_numeric_dtype(result[col]):
            result[col] = result[col].astype(object).where(result[col].notna(), np.nan)
    numeric = result.select_dtypes(include="number").columns
    result[numeric] = result[numeric].replace([np.inf, -np.inf], np.nan)
    return result


def _json_records(frame):
    return [
        {
            key: None
            if pd.isna(value)
            else value.item()
            if isinstance(value, np.generic)
            else value
            for key, value in row.items()
        }
        for row in frame.to_dict(orient="records")
    ]


def train(gold, manifest, registry, external_sources=(), mlflow_uri=None):
    registry = Path(registry)
    registry.mkdir(parents=True, exist_ok=True)
    splits, development, test = temporal_splits(gold)
    experiments, bundles, test_predictions = [], {}, []
    for mode in ["full", "internal"]:
        cols = features(gold, external_sources, internal=mode == "internal")
        X = clean(gold, cols)
        rankings, oof_by_kind = [], {}
        for kind in ["baseline", "lightgbm"]:
            fold_metrics, oof_predictions = [], []
            for i, (tr, va) in enumerate(splits):
                if gold.loc[tr, "target"].nunique() != 2:
                    raise ValueError("Training window needs both classes")
                model = pipeline(X.loc[tr], kind)
                model.fit(X.loc[tr], gold.loc[tr, "target"])
                pred = gold.loc[va, ["case_id", "WEEK_NUM", "target"]].copy()
                pred["score"] = model.predict_proba(X.loc[va])[:, 1]
                pred["raw_score"] = pred["score"]
                pred["train_max_week"] = int(gold.loc[tr, "WEEK_NUM"].max())
                pred["fold"] = i
                oof_predictions.append(pred)
                metrics, weekly = evaluate(pred)
                experiments.append(
                    {
                        "mode": mode,
                        "model": kind,
                        "fold": i,
                        "train_max_week": int(gold.loc[tr, "WEEK_NUM"].max()),
                        "validation_min_week": int(gold.loc[va, "WEEK_NUM"].min()),
                        "validation_max_week": int(gold.loc[va, "WEEK_NUM"].max()),
                        "weekly_metrics": _json_records(weekly),
                        **metrics,
                    }
                )
                fold_metrics.append(metrics)
            valid_scores = [m["stability"] for m in fold_metrics if m["stability"] is not None]
            ranking = float(np.mean(valid_scores)) if len(valid_scores) == len(splits) else -np.inf
            rankings.append((ranking, kind))
            oof_by_kind[kind] = pd.concat(oof_predictions, ignore_index=True)
        if all(not np.isfinite(r[0]) for r in rankings):
            raise ValueError("No candidate has defined stability across every validation fold")
        _, chosen = max(rankings)
        model = pipeline(X.loc[development], chosen)
        model.fit(X.loc[development], gold.loc[development, "target"])
        code_hash = hashlib.sha256(
            b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
        ).hexdigest()[:16]
        identity = {
            "code_version": code_hash,
            "parameters": model.get_params()["model"].get_params(),
            "data": manifest["data_version"],
            "feature_version": FEATURE_VERSION,
            "mode": mode,
            "model": chosen,
            "seed": 42,
            "columns": cols,
        }
        version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        calibrator, calibration = select_calibrator(
            oof_by_kind[chosen], version, manifest["data_version"], mode
        )
        bundle = {
            "pipeline": model,
            "calibrator": calibrator,
            "calibration": calibration,
            "columns": cols,
            "version": version,
            "mode": mode,
            "feature_version": FEATURE_VERSION,
            "identity": identity,
            "trained_at": now(),
            "external_sources": list(external_sources),
            "types": {
                c: "number" if pd.api.types.is_numeric_dtype(X[c]) else "category" for c in cols
            },
        }
        folder = registry / version
        folder.mkdir(exist_ok=True)
        joblib.dump(bundle, folder / "model.joblib.tmp")
        (folder / "model.joblib.tmp").replace(folder / "model.joblib")
        write_json(
            folder / "metadata.json",
            {k: v for k, v in bundle.items() if k not in {"pipeline", "calibrator"}},
        )
        pred = gold.loc[test, ["case_id", "WEEK_NUM", "target"]].copy()
        pred["raw_score"] = model.predict_proba(X.loc[test])[:, 1]
        pred["calibrated_score"] = calibrator.predict(pred.raw_score)
        # Backward compatibility: score remains the uncalibrated model probability.
        pred["score"] = pred["raw_score"]
        pred["calibrator_version"] = calibration["version"]
        pred["model_version"], pred["mode"] = version, mode
        test_predictions.append(pred)
        bundles[mode] = version
        if mlflow_uri:
            import mlflow

            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("home-credit-stability")
            with mlflow.start_run(run_name=f"{mode}-{version}"):
                mlflow.log_params(
                    {
                        "model": chosen,
                        "seed": 42,
                        "data_version": manifest["data_version"],
                        "mode": mode,
                    }
                )
                mlflow.log_metrics({k: v for k, v in evaluate(pred)[0].items() if v is not None})
                mlflow.log_artifact(str(folder / "metadata.json"))
    write_json(registry / "candidate.json", bundles)
    write_json(registry / "experiments.json", experiments)
    return bundles, pd.concat(test_predictions, ignore_index=True), experiments


def activate(registry, versions):
    """Atomic pair activation; versions must refer to locally validated bundles."""
    registry = Path(registry)
    for mode in ["full", "internal"]:
        bundle = joblib.load(registry / versions[mode] / "model.joblib")
        if bundle["mode"] != mode or bundle["feature_version"] != FEATURE_VERSION:
            raise ValueError("Incompatible model bundle")
        _validate_calibrator_binding(bundle)
    active = registry / "active.json"
    if active.exists():
        write_json(registry / "previous.json", json.loads(active.read_text()))
    write_json(active, versions)


def rollback(registry):
    registry = Path(registry)
    activate(registry, json.loads((registry / "previous.json").read_text()))


def load_active(registry, unavailable=()):
    registry = Path(registry)
    versions = json.loads((registry / "active.json").read_text())
    full = joblib.load(registry / versions["full"] / "model.joblib")
    unknown = set(unavailable) - set(full["external_sources"])
    if unknown:
        raise ValueError(f"Unknown external sources: {sorted(unknown)}")
    mode = "internal" if unavailable else "full"
    bundle = joblib.load(registry / versions[mode] / "model.joblib")
    _validate_calibrator_binding(bundle)
    return bundle


def _validate_calibrator_binding(bundle):
    """Reject a calibrator copied from another model or mode."""
    calibration = bundle.get("calibration")
    if calibration is None:
        return
    if calibration.get("model_version") != bundle.get("version"):
        raise ValueError("Calibrator is not bound to this model version")
    if calibration.get("mode") != bundle.get("mode"):
        raise ValueError("Calibrator mode does not match model mode")


def score(bundle, frame):
    """Return the legacy raw model probability."""
    return score_details(bundle, frame)["raw_score"]


def score_details(bundle, frame):
    """Return raw and calibrated probabilities from the bound model bundle."""
    missing = set(bundle["columns"]) - set(frame)
    if missing:
        raise ValueError(f"Missing contracted features: {sorted(missing)}")
    for col, kind in bundle["types"].items():
        if kind == "number":
            try:
                numeric = pd.to_numeric(frame[col], errors="raise")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{col} must be numeric") from exc
            if np.isinf(numeric.dropna()).any():
                raise ValueError(f"{col} contains infinity")
    raw = bundle["pipeline"].predict_proba(clean(frame, bundle["columns"]))[:, 1]
    calibrator = bundle.get("calibrator")
    calibrated = calibrator.predict(raw) if calibrator is not None else raw.copy()
    return {"raw_score": raw, "calibrated_score": calibrated}
