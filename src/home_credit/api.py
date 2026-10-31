"""Feature-contract scoring API. Registry contents must be trusted local artifacts."""

import asyncio
import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .common import now
from .data import FEATURE_VERSION
from .model import load_active, score_details


class ScoreRequest(BaseModel):
    feature_version: str
    applications: list[dict] = Field(min_length=1, max_length=1000)
    unavailable_sources: list[str] = Field(default_factory=list)


def create_app(registry=None):
    registry = Path(registry or os.getenv("HOME_CREDIT_REGISTRY", "artifacts/registry"))
    app = FastAPI(title="Home Credit Risk & Stability", version="0.1.0")

    @app.get("/health")
    def health():
        if not (registry / "active.json").exists():
            raise HTTPException(503, "No active model")
        return {"status": "ready", "feature_version": FEATURE_VERSION}

    @app.post("/score")
    async def predict(request: ScoreRequest):
        if request.feature_version != FEATURE_VERSION:
            raise HTTPException(409, "Incompatible feature version")
        frame = pd.DataFrame(request.applications)
        if (
            "case_id" not in frame
            or frame.case_id.isna().any()
            or frame.case_id.duplicated().any()
            or any(type(row.get("case_id")) is not int for row in request.applications)
        ):
            raise HTTPException(422, "Unique non-null case_id required")

        def run():
            bundle = load_active(registry, request.unavailable_sources)
            predictions = score_details(bundle, frame)
            return {
                "model_version": bundle["version"],
                "feature_version": FEATURE_VERSION,
                "mode": bundle["mode"],
                "scored_at": now(),
                "quality_status": "degraded" if bundle["mode"] == "internal" else "ok",
                "reason": "external_source_unavailable" if bundle["mode"] == "internal" else None,
                "predictions": [
                    {
                        "case_id": int(case),
                        # score remains raw for clients of the v0.1 contract.
                        "score": float(raw),
                        "raw_score": float(raw),
                        "calibrated_score": float(calibrated),
                    }
                    for case, raw, calibrated in zip(
                        frame.case_id,
                        predictions["raw_score"],
                        predictions["calibrated_score"],
                    )
                ],
            }

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(run),
                float(os.getenv("HOME_CREDIT_TIMEOUT_SECONDS", "10")),
            )
        except asyncio.TimeoutError:
            raise HTTPException(504, "Scoring deadline exceeded")
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc))
        except FileNotFoundError:
            raise HTTPException(503, "Model unavailable")

    return app


app = create_app()
