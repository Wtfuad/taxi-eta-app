"""FastAPI route handlers for Porto Taxi ETA Prediction."""
from __future__ import annotations

import io
import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from taxi_eta.config import Settings
from taxi_eta.features import STEP_S
from taxi_eta.predictor import ArtifactError, Predictor
from taxi_eta.traces import (
    TripRequest,
    read_batch,
    resample_route,
    validate_points,
)

from .schemas import (
    BatchPredictRequest,
    BatchPredictionItemResult,
    BatchResponse,
    CurvePoint,
    CurveRequest,
    CurveResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResult,
    ResampleRouteRequest,
    ResampleRouteResponse,
    TripInput,
    format_duration,
)

log = logging.getLogger("taxi_eta.api")
router = APIRouter()

# Global predictor holder (initialized during app lifespan)
_PREDICTOR: Optional[Predictor] = None
_PREDICTOR_LOAD_ERROR: Optional[str] = None


def get_predictor() -> Predictor:
    """Dependency that returns the loaded Predictor instance or raises 503."""
    global _PREDICTOR, _PREDICTOR_LOAD_ERROR
    if _PREDICTOR is not None:
        return _PREDICTOR

    # Try on-demand load if not yet initialized
    try:
        settings = Settings.from_env()
        _PREDICTOR = Predictor(str(settings.model_dir))
        _PREDICTOR_LOAD_ERROR = None
        return _PREDICTOR
    except Exception as exc:
        _PREDICTOR_LOAD_ERROR = str(exc)
        log.error("Failed to load model: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model is not loaded or could not be found: {exc}",
        )


def set_predictor(pred: Optional[Predictor], err: Optional[str] = None) -> None:
    global _PREDICTOR, _PREDICTOR_LOAD_ERROR
    _PREDICTOR = pred
    _PREDICTOR_LOAD_ERROR = err


# -----------------------------------------------------------------------------
# System & Information Endpoints
# -----------------------------------------------------------------------------
@router.get("/", tags=["Info"])
def api_root() -> Dict[str, Any]:
    """Root endpoint providing service information and API documentation links."""
    return {
        "service": "Porto Taxi Trip-Time Predictor API",
        "description": "FastAPI service for predicting total taxi trip duration from partial GPS traces.",
        "documentation": "/docs",
        "openapi_schema": "/openapi.json",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "metrics": "/model/metrics",
            "demo_trips": "/demo/trips",
            "predict_single": "POST /predict",
            "predict_batch_json": "POST /predict/batch",
            "predict_batch_csv": "POST /predict/batch/csv",
            "predict_curve": "POST /predict/curve",
            "resample_route": "POST /tools/resample-route",
        },
    }


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Health check endpoint indicating whether the service and model are ready."""
    try:
        pred = get_predictor()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_version=pred.info.get("version"),
            synthetic=bool(pred.info.get("synthetic", False)),
        )
    except HTTPException as exc:
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            details=exc.detail,
        )


@router.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def model_info(pred: Predictor = Depends(get_predictor)) -> ModelInfoResponse:
    """Detailed metadata about the loaded model."""
    info = pred.info
    return ModelInfoResponse(
        version=info.get("version", "unknown"),
        created_utc=info.get("created_utc"),
        n_features=info.get("n_features", len(pred.features)),
        n_final_models=info.get("n_final_models", len(pred.raw_models) + len(pred.log_models)),
        blend_weight_raw=float(info.get("blend_weight_raw", pred.w_blend)),
        training_cuts=info.get("training_cuts"),
        synthetic=bool(info.get("synthetic", False)),
        lightgbm_trained_with=info.get("lightgbm_trained_with"),
        features=pred.features,
    )


@router.get("/model/metrics", tags=["Model"])
def model_metrics(pred: Predictor = Depends(get_predictor)) -> Dict[str, Any]:
    """Validation performance metrics and slice breakdowns from training."""
    if not pred.metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No metrics.json found in the model directory.",
        )
    return pred.metrics


@router.get("/demo/trips", tags=["Model"])
def list_demo_trips(
    limit: int = Query(10, ge=1, le=100, description="Max number of demo trips to return"),
    pred: Predictor = Depends(get_predictor),
) -> List[Dict[str, Any]]:
    """Retrieve sample demo trips from the model artifacts directory."""
    demo_path = pred.dir / "demo_trips.csv"
    if not demo_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No demo_trips.csv found in the model directory.",
        )
    df = pd.read_csv(demo_path).head(limit)
    records = []
    for _, row in df.iterrows():
        records.append({
            "trip_id": int(row["trip_id"]),
            "timestamp": int(row["timestamp"]),
            "call_type": str(row["call_type"]),
            "origin_stand": int(row["origin_stand"]) if not pd.isna(row["origin_stand"]) else None,
            "taxi_id": int(row["taxi_id"]) if not pd.isna(row["taxi_id"]) else None,
            "true_total_s": float(row["true_total_s"]),
            "formatted_true_total": format_duration(float(row["true_total_s"])),
            "polyline": json.loads(row["polyline"]),
        })
    return records


# -----------------------------------------------------------------------------
# Prediction Endpoints
# -----------------------------------------------------------------------------
@router.post("/predict", response_model=PredictionResult, tags=["Prediction"])
def predict_single(
    payload: TripInput,
    pred: Predictor = Depends(get_predictor),
) -> PredictionResult:
    """Predict total and remaining duration for a single taxi trip."""
    try:
        req, tr = payload.to_trip_request()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    if payload.include_features:
        pred_df, feat_df = pred.predict_with_features([req], level=payload.level)
        feats = feat_df.iloc[0].to_dict()
        # convert numpy types to native Python
        clean_feats = {k: (None if pd.isna(v) else (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating, float)) else str(v))) for k, v in feats.items()}
    else:
        pred_df = pred.predict([req], level=payload.level)
        clean_feats = None

    r = pred_df.iloc[0]
    low_s = float(r.low_s) if np.isfinite(r.low_s) else None
    high_s = float(r.high_s) if np.isfinite(r.high_s) else None
    range_str = f"{format_duration(low_s)} to {format_duration(high_s)}" if low_s and high_s else None

    return PredictionResult(
        observed_points=int(r.observed_points),
        elapsed_s=round(float(r.elapsed_s), 1),
        total_s=round(float(r.total_s), 1),
        remaining_s=round(float(r.remaining_s), 1),
        low_s=round(low_s, 1) if low_s is not None else None,
        high_s=round(high_s, 1) if high_s is not None else None,
        range_level=payload.level,
        formatted_total=format_duration(r.total_s),
        formatted_remaining=format_duration(r.remaining_s),
        formatted_range=range_str,
        notes=tr.notes,
        warnings=tr.warnings,
        features=clean_feats,
        model_version=pred.info.get("version", "unknown"),
    )


@router.post("/predict/curve", response_model=CurveResponse, tags=["Prediction"])
def predict_curve(
    payload: CurveRequest,
    pred: Predictor = Depends(get_predictor),
) -> CurveResponse:
    """Simulate how the ETA estimate evolves point-by-point as the trip progresses."""
    trip_in = TripInput(
        polyline=payload.polyline,
        timestamp=payload.timestamp,
        call_type=payload.call_type,
        origin_stand=payload.origin_stand,
        taxi_id=payload.taxi_id,
        level=payload.level,
    )
    req, tr = trip_in.to_trip_request()
    if len(req.pts) < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Trace must contain at least 1 point to compute an evolution curve.",
        )

    cur = pred.predict_curve(req.pts, req, step=payload.step, level=payload.level)
    points = []
    for _, row in cur.iterrows():
        points.append(
            CurvePoint(
                k=int(row["k"]),
                elapsed_s=round(float(row["elapsed_s"]), 1),
                total_s=round(float(row["total_s"]), 1),
                remaining_s=round(float(row["remaining_s"]), 1),
                low_s=round(float(row["low_s"]), 1) if np.isfinite(row["low_s"]) else None,
                high_s=round(float(row["high_s"]), 1) if np.isfinite(row["high_s"]) else None,
            )
        )

    return CurveResponse(curve=points, model_version=pred.info.get("version", "unknown"))


@router.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch_json(
    payload: BatchPredictRequest,
    pred: Predictor = Depends(get_predictor),
) -> BatchResponse:
    """Predict multiple trips sent as a JSON array."""
    settings = Settings.from_env()
    if len(payload.trips) > settings.max_batch_rows:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {len(payload.trips)} exceeds maximum allowed of {settings.max_batch_rows}.",
        )

    results: List[BatchPredictionItemResult] = []
    valid_requests: List[TripRequest] = []
    valid_indices: List[int] = []

    for idx, item in enumerate(payload.trips):
        row_id = item.id if item.id is not None else idx + 1
        trip_in = TripInput(
            polyline=item.polyline,
            timestamp=item.timestamp,
            call_type=item.call_type or "C",
            origin_stand=item.origin_stand,
            taxi_id=item.taxi_id,
            level=payload.level,
        )
        try:
            req, tr = trip_in.to_trip_request()
            valid_requests.append(req)
            valid_indices.append(idx)
            results.append(
                BatchPredictionItemResult(
                    id=row_id,
                    status="ok",
                    message="; ".join(tr.notes + tr.warnings) if (tr.notes or tr.warnings) else None,
                    actual_total_s=item.actual_total_s,
                )
            )
        except Exception as exc:
            results.append(
                BatchPredictionItemResult(
                    id=row_id,
                    status="error",
                    message=str(exc),
                    actual_total_s=item.actual_total_s,
                )
            )

    # Perform prediction on all valid requests
    if valid_requests:
        preds = pred.predict(valid_requests, level=payload.level)
        for i, row_pred in preds.iterrows():
            res_idx = valid_indices[i]
            low_s = float(row_pred.low_s) if np.isfinite(row_pred.low_s) else None
            high_s = float(row_pred.high_s) if np.isfinite(row_pred.high_s) else None
            range_str = f"{format_duration(low_s)} to {format_duration(high_s)}" if low_s and high_s else None

            pred_res = PredictionResult(
                observed_points=int(row_pred.observed_points),
                elapsed_s=round(float(row_pred.elapsed_s), 1),
                total_s=round(float(row_pred.total_s), 1),
                remaining_s=round(float(row_pred.remaining_s), 1),
                low_s=round(low_s, 1) if low_s is not None else None,
                high_s=round(high_s, 1) if high_s is not None else None,
                range_level=payload.level,
                formatted_total=format_duration(row_pred.total_s),
                formatted_remaining=format_duration(row_pred.remaining_s),
                formatted_range=range_str,
                model_version=pred.info.get("version", "unknown"),
            )
            results[res_idx].prediction = pred_res
            if results[res_idx].actual_total_s is not None:
                results[res_idx].error_s = round(float(row_pred.total_s - results[res_idx].actual_total_s), 1)

    # Summary
    ok_count = sum(1 for r in results if r.status == "ok")
    summary = {
        "total_rows": len(payload.trips),
        "successful_predictions": ok_count,
        "failed_rows": len(payload.trips) - ok_count,
    }

    # Evaluate scoring if actual values are present
    scored = [r for r in results if r.status == "ok" and r.actual_total_s is not None and r.prediction is not None]
    if scored:
        preds_tot = np.array([r.prediction.total_s for r in scored])
        acts_tot = np.array([r.actual_total_s for r in scored])
        summary["scored_trips"] = len(scored)
        summary["rmsle"] = round(float(np.sqrt(np.mean((np.log1p(preds_tot) - np.log1p(acts_tot)) ** 2))), 4)
        summary["rmse_s"] = round(float(np.sqrt(np.mean((preds_tot - acts_tot) ** 2))), 1)

    return BatchResponse(results=results, summary=summary)


@router.post("/predict/batch/csv", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch_csv(
    file: UploadFile = File(..., description="CSV file with polyline and timestamp columns"),
    level: float = Query(0.8, ge=0.5, le=0.99, description="Confidence level for range"),
    pred: Predictor = Depends(get_predictor),
) -> BatchResponse:
    """Predict trips uploaded via a CSV file."""
    settings = Settings.from_env()
    content = await file.read()
    if len(content) > 100_000_000:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="CSV file exceeds maximum upload size of 100 MB.",
        )

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded file could not be parsed as CSV: {exc}",
        )

    if len(df) > settings.max_batch_rows:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"CSV has {len(df):,} rows; limit is {settings.max_batch_rows:,}.",
        )

    try:
        requests, meta = read_batch(df)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    ok_indices = [i for i, r in enumerate(requests) if r is not None]
    results: List[BatchPredictionItemResult] = []

    for i, m in enumerate(meta):
        results.append(
            BatchPredictionItemResult(
                id=m["id"],
                status=m["status"],
                message=m.get("message") or None,
                actual_total_s=m.get("actual_total_s") if np.isfinite(m.get("actual_total_s", np.nan)) else None,
            )
        )

    if ok_indices:
        pred_df = pred.predict([requests[i] for i in ok_indices], level=level)
        for j, row_idx in enumerate(ok_indices):
            row_pred = pred_df.iloc[j]
            low_s = float(row_pred.low_s) if np.isfinite(row_pred.low_s) else None
            high_s = float(row_pred.high_s) if np.isfinite(row_pred.high_s) else None
            range_str = f"{format_duration(low_s)} to {format_duration(high_s)}" if low_s and high_s else None

            results[row_idx].prediction = PredictionResult(
                observed_points=int(row_pred.observed_points),
                elapsed_s=round(float(row_pred.elapsed_s), 1),
                total_s=round(float(row_pred.total_s), 1),
                remaining_s=round(float(row_pred.remaining_s), 1),
                low_s=round(low_s, 1) if low_s is not None else None,
                high_s=round(high_s, 1) if high_s is not None else None,
                range_level=level,
                formatted_total=format_duration(row_pred.total_s),
                formatted_remaining=format_duration(row_pred.remaining_s),
                formatted_range=range_str,
                model_version=pred.info.get("version", "unknown"),
            )
            if results[row_idx].actual_total_s is not None:
                results[row_idx].error_s = round(float(row_pred.total_s - results[row_idx].actual_total_s), 1)

    summary = {
        "total_rows": len(df),
        "successful_predictions": len(ok_indices),
        "failed_rows": len(df) - len(ok_indices),
    }
    scored = [r for r in results if r.status == "ok" and r.actual_total_s is not None and r.prediction is not None]
    if scored:
        preds_tot = np.array([r.prediction.total_s for r in scored])
        acts_tot = np.array([r.actual_total_s for r in scored])
        summary["scored_trips"] = len(scored)
        summary["rmsle"] = round(float(np.sqrt(np.mean((np.log1p(preds_tot) - np.log1p(acts_tot)) ** 2))), 4)
        summary["rmse_s"] = round(float(np.sqrt(np.mean((preds_tot - acts_tot) ** 2))), 1)

    return BatchResponse(results=results, summary=summary)


# -----------------------------------------------------------------------------
# Utility Tools
# -----------------------------------------------------------------------------
@router.post("/tools/resample-route", response_model=ResampleRouteResponse, tags=["Tools"])
def resample_drawn_route(payload: ResampleRouteRequest) -> ResampleRouteResponse:
    """Convert drawn map coordinates into 15-second GPS points at an assumed constant speed."""
    if len(payload.vertices) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Route requires at least 2 vertices.",
        )
    pts = resample_route(payload.vertices, payload.speed_kmh)
    pts_list = [[round(float(p[0]), 6), round(float(p[1]), 6)] for p in pts]
    driven_seconds = (len(pts) - 1) * STEP_S
    return ResampleRouteResponse(
        points=pts_list,
        n_points=len(pts_list),
        step_seconds=STEP_S,
        driven_seconds=driven_seconds,
    )
