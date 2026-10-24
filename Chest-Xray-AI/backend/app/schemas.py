from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    detail: str


class PreprocessStep(BaseModel):
    name: str
    image: str


class QualityWarning(BaseModel):
    code: str
    message: str


class Explanation(BaseModel):
    available: bool
    method: str | None = None
    title: str | None = None
    heatmap: str | None = None
    overlay: str | None = None
    note: str


class ModelRef(BaseModel):
    name: str
    version: str
    architecture: str


class ReportSection(BaseModel):
    title: str
    body: str | None = None
    items: list[str] | None = None


class PredictionResponse(BaseModel):
    prediction_id: str
    created_at: str
    filename: str
    model: ModelRef
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    low_confidence: bool
    processed_image: str
    preprocessing_steps: list[PreprocessStep]
    explanation: Explanation
    quality_warnings: list[QualityWarning]
    report: list[ReportSection]


class ReportRequest(BaseModel):
    prediction_id: str


class ErrorResponse(BaseModel):
    error: str
