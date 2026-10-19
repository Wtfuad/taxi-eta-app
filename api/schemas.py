"""Pydantic schemas for the FastAPI service."""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator

from taxi_eta.features import CALL_MAP
from taxi_eta.traces import (
    TraceResult,
    TripRequest,
    local_to_unix,
    parse_trace_text,
    validate_points,
)


def format_duration(seconds: Optional[float]) -> str:
    """Format duration in seconds to a human-readable string."""
    if seconds is None or not np.isfinite(seconds):
        return "n/a"
    s = int(round(seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min" if h else f"{m} min {s:02d} s"


class TripInput(BaseModel):
    """Input payload for a single trip prediction."""
    polyline: Union[List[List[float]], str] = Field(
        ...,
        description="GPS coordinates as [[lon, lat], ...] or raw JSON / text / GeoJSON string.",
        examples=[[[-8.6110, 41.1496], [-8.6105, 41.1498], [-8.6098, 41.1500]]],
    )
    timestamp: Union[int, float, str, datetime] = Field(
        ...,
        description="Trip start time as Unix timestamp (seconds/ms) or ISO date-time string.",
        examples=[1407000000],
    )
    call_type: Union[str, int] = Field(
        default="C",
        description="Call type: 'A' (dispatch), 'B' (stand), 'C' (street hail) or 0, 1, 2.",
        examples=["C"],
    )
    origin_stand: Optional[int] = Field(
        default=None,
        description="Taxi stand ID (1-63) if call_type is 'B', otherwise null or -1.",
        examples=[None],
    )
    taxi_id: Optional[int] = Field(
        default=None,
        description="Taxi ID (e.g. 20000589). If omitted, model uses global prior.",
        examples=[20000589],
    )
    level: float = Field(
        default=0.8,
        ge=0.5,
        le=0.99,
        description="Prediction confidence level (e.g. 0.8 for 80% range).",
        examples=[0.8],
    )
    include_features: bool = Field(
        default=False,
        description="Whether to include engineered features in the response.",
    )

    def to_trip_request(self) -> tuple[TripRequest, TraceResult]:
        """Convert this input into a TripRequest and validation result."""
        # 1. Parse polyline
        if isinstance(self.polyline, list):
            raw_text = json.dumps(self.polyline)
        else:
            raw_text = str(self.polyline)
        tr = parse_trace_text(raw_text)
        if not tr.ok:
            raise ValueError("; ".join(tr.errors) if tr.errors else "Invalid polyline")

        # 2. Parse timestamp
        ts = self.timestamp
        if isinstance(ts, (int, float)):
            ts_int = int(ts / 1000) if ts > 1e11 else int(ts)
        elif isinstance(ts, datetime):
            ts_int = local_to_unix(ts)
        else:
            ts_int = local_to_unix(str(ts))

        # 3. Parse call type
        if isinstance(self.call_type, int):
            call_int = self.call_type if self.call_type in (0, 1, 2) else 2
        else:
            call_int = CALL_MAP.get(str(self.call_type).strip().upper(), 2)

        # 4. Stand & Taxi
        stand_int = int(self.origin_stand) if self.origin_stand is not None and self.origin_stand >= 0 else -1
        taxi_int = int(self.taxi_id) if self.taxi_id is not None else None

        req = TripRequest(pts=tr.pts, ts=ts_int, call=call_int, stand=stand_int, taxi=taxi_int)
        return req, tr


class PredictionResult(BaseModel):
    """Prediction output details for a trip."""
    observed_points: int = Field(..., description="Number of 15-second GPS points observed.")
    elapsed_s: float = Field(..., description="Seconds driven so far.")
    total_s: float = Field(..., description="Predicted total trip duration in seconds.")
    remaining_s: float = Field(..., description="Predicted remaining trip duration in seconds.")
    low_s: Optional[float] = Field(None, description="Lower bound of empirical prediction range.")
    high_s: Optional[float] = Field(None, description="Upper bound of empirical prediction range.")
    range_level: float = Field(0.8, description="Confidence level for the range (e.g. 0.8).")
    formatted_total: str = Field(..., description="Human-readable total trip time.")
    formatted_remaining: str = Field(..., description="Human-readable remaining trip time.")
    formatted_range: Optional[str] = Field(None, description="Human-readable prediction range.")
    notes: List[str] = Field(default_factory=list, description="Parsing/coordinate normalization notes.")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings (e.g. outside Porto).")
    features: Optional[Dict[str, Any]] = Field(None, description="Engineered features if requested.")
    model_version: str = Field(..., description="Model version string.")


class BatchTripItem(BaseModel):
    """One trip item in a JSON batch request."""
    id: Optional[Union[str, int]] = Field(None, description="Custom identifier for the trip.")
    polyline: Union[List[List[float]], str] = Field(..., description="GPS trace coordinates or text.")
    timestamp: Union[int, float, str, datetime] = Field(..., description="Trip start timestamp.")
    call_type: Optional[Union[str, int]] = Field("C", description="Call type (A, B, C).")
    origin_stand: Optional[int] = Field(None, description="Stand ID.")
    taxi_id: Optional[int] = Field(None, description="Taxi ID.")
    actual_total_s: Optional[float] = Field(None, description="Actual trip duration for scoring.")


class BatchPredictRequest(BaseModel):
    """Batch prediction request payload."""
    trips: List[BatchTripItem] = Field(..., description="List of trip records to predict.")
    level: float = Field(0.8, ge=0.5, le=0.99, description="Confidence level.")


class BatchPredictionItemResult(BaseModel):
    """Result for one row in a batch prediction."""
    id: Union[str, int]
    status: str = Field(..., description="'ok' or 'error'")
    message: Optional[str] = None
    prediction: Optional[PredictionResult] = None
    actual_total_s: Optional[float] = None
    error_s: Optional[float] = None


class BatchResponse(BaseModel):
    """Overall batch prediction response."""
    results: List[BatchPredictionItemResult]
    summary: Dict[str, Any]


class CurveRequest(BaseModel):
    """Request for incremental prefix predictions along a trace."""
    polyline: Union[List[List[float]], str]
    timestamp: Union[int, float, str, datetime]
    call_type: Union[str, int] = "C"
    origin_stand: Optional[int] = None
    taxi_id: Optional[int] = None
    step: int = Field(1, ge=1, description="Interval step across GPS points.")
    level: float = Field(0.8, ge=0.5, le=0.99)


class CurvePoint(BaseModel):
    k: int
    elapsed_s: float
    total_s: float
    remaining_s: float
    low_s: Optional[float]
    high_s: Optional[float]


class CurveResponse(BaseModel):
    curve: List[CurvePoint]
    model_version: str


class ResampleRouteRequest(BaseModel):
    """Resample drawn vertices to 15-second GPS points at assumed constant speed."""
    vertices: List[List[float]] = Field(..., description="List of [lon, lat] vertices.")
    speed_kmh: float = Field(25.0, ge=1.0, le=120.0, description="Assumed speed in km/h.")


class ResampleRouteResponse(BaseModel):
    points: List[List[float]]
    n_points: int
    step_seconds: float = 15.0
    driven_seconds: float


class ModelInfoResponse(BaseModel):
    version: str
    created_utc: Optional[str] = None
    n_features: int
    n_final_models: int
    blend_weight_raw: float
    training_cuts: Optional[int] = None
    synthetic: bool
    lightgbm_trained_with: Optional[str] = None
    features: List[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    synthetic: bool = False
    details: Optional[str] = None
