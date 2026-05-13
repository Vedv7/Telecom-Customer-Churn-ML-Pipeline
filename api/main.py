from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from telecom_churn.train import predict_row


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = _bundle_path()
    if path.is_file():
        app.state.bundle = joblib.load(path)
    else:
        app.state.bundle = None
    yield


app = FastAPI(title="Telecom churn inference", version="0.2.0", lifespan=lifespan)


class PredictBody(BaseModel):
    """Raw workbook-like row: optional ``year`` / ``user_account_id`` stripped; ``churn`` ignored if present."""

    features: dict[str, float] = Field(
        ...,
        description="Feature dict as in the training spreadsheet (see artifacts/metrics.json expected_input_columns).",
    )


def _bundle_path() -> Path:
    raw = os.environ.get("CHURN_BUNDLE_PATH", "artifacts/churn_bundle.joblib")
    return Path(raw).resolve()


@app.get("/health")
def health() -> dict:
    bundle = getattr(app.state, "bundle", None)
    loaded = bundle is not None
    version = bundle.get("version") if isinstance(bundle, dict) else None
    return {
        "status": "ok",
        "bundle_loaded": loaded,
        "bundle_version": version,
        "bundle_path": str(_bundle_path()),
    }


@app.get("/schema")
def inference_schema() -> dict:
    """Column names expected on each raw scoring row (after automatic cleaning)."""
    bundle = getattr(app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Bundle not loaded.")
    if bundle.get("version") != 2 or bundle.get("pipeline") is None:
        raise HTTPException(
            status_code=400,
            detail="Bundle is legacy v1; use metrics.json scaler_columns or retrain with bundle v2.",
        )
    return {"expected_input_columns": bundle["pipeline"].raw_feature_names()}


@app.post("/predict")
def predict(body: PredictBody) -> dict:
    bundle = getattr(app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Bundle not loaded; train first (churn-train) or set CHURN_BUNDLE_PATH.",
        )

    try:
        return predict_row(bundle, body.features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
