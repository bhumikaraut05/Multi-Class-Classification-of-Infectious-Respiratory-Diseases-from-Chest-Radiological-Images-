"""FastAPI application."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import get_settings
from .model import ModelService
from .prediction import predict_image
from .preprocessing import validate_image
from .report import make_pdf
from .schemas import ErrorResponse, HealthResponse, PredictionResponse, ReportRequest
from .utils import ApiError

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("cxr.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = ModelService(settings)
    service.load()  # once, at startup
    app.state.model_service = service
    yield


app = FastAPI(title="CXR-AI API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.frontend_urls, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.exception_handler(ApiError)
async def api_error_handler(_: Request, exc: ApiError):
    return JSONResponse({"error": exc.message}, status_code=exc.status_code)


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(_: Request, exc: StarletteHTTPException):
    return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, __: RequestValidationError):
    return JSONResponse({"error": "Please upload an X-ray image."}, status_code=400)


@app.exception_handler(Exception)
async def unhandled_handler(_: Request, exc: Exception):
    log.error("Unhandled error: %s", type(exc).__name__)  # no stack trace to client
    return JSONResponse({"error": "Unable to analyze this image. Please try another X-ray."}, status_code=500)


@app.get("/api/health", response_model=HealthResponse)
def health(request: Request):
    svc: ModelService = request.app.state.model_service
    return HealthResponse(status="healthy", model_loaded=svc.loaded, detail=svc.status_message())


@app.get("/api/model-info")
def model_info(request: Request):
    return request.app.state.model_service.info()


@app.post("/api/predict", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}})
def predict(request: Request, file: UploadFile = File(...)):
    svc: ModelService = request.app.state.model_service
    data = file.file.read(settings.max_upload_bytes + 1)  # bounded read; never stored on disk
    validate_image(file.filename, data, settings.max_upload_bytes)
    log.info("Received image (%d bytes)", len(data))        # never log image content
    return predict_image(svc, data, file.filename)


@app.post("/api/report")
def report(request: Request, body: ReportRequest):
    """PDF built from a prediction this server actually produced (kept in memory, not on disk)."""
    svc: ModelService = request.app.state.model_service
    result = svc.results.get(body.prediction_id)
    if result is None:
        raise ApiError(404, "This analysis is no longer available. Please analyze the image again.")
    pdf = make_pdf(result)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="cxr-report-{body.prediction_id}.pdf"'})


PLOTS = {"confusion_matrix", "roc_curves", "pr_curves", "training_curves"}


@app.get("/api/evaluation-plot/{name}")
def evaluation_plot(name: str):
    path = settings.reports_dir / f"{name}.png"
    if name not in PLOTS or not path.is_file():
        raise ApiError(404, "Evaluation plot not available.")
    return FileResponse(path, media_type="image/png")


# Serve the built website (frontend/dist) from the same port when it exists -> one-command local use.
_dist = settings.frontend_dist
if (_dist / "index.html").is_file():
    from fastapi.staticfiles import StaticFiles
    app.mount("/assets", StaticFiles(directory=_dist / "assets"), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise ApiError(404, "Not found.")
        target = (_dist / full_path).resolve()
        if full_path and target.is_file() and _dist.resolve() in target.parents:
            return FileResponse(target)
        return FileResponse(_dist / "index.html")
