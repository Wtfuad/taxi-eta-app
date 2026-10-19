"""FastAPI Application entrypoint for Porto Taxi Trip-Time Predictor."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from taxi_eta.config import Settings
from taxi_eta.predictor import ArtifactError, Predictor

from .routes import router, set_predictor

log = logging.getLogger("taxi_eta.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager to pre-load predictor models."""
    settings = Settings.from_env()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.info("Starting Taxi ETA FastAPI Service...")
    log.info("Model directory configured: %s", settings.model_dir)

    from . import routes
    if routes._PREDICTOR is None:
        try:
            predictor = Predictor(str(settings.model_dir))
            set_predictor(predictor, None)
            log.info(
                "Predictor initialized successfully: version=%s, features=%d, final_models=%d",
                predictor.info.get("version"),
                predictor.info.get("n_features"),
                predictor.info.get("n_final_models"),
            )
        except Exception as exc:
            log.warning("Could not pre-load model during startup: %s", exc)
            log.warning("Requests to /predict will return 503 until valid model artifacts are provided in %s", settings.model_dir)
            set_predictor(None, str(exc))
    else:
        log.info("Predictor already initialized.")

    yield

    log.info("Shutting down Taxi ETA FastAPI Service.")


app = FastAPI(
    title="Porto Taxi Trip-Time Predictor API",
    description=(
        "Production-ready REST API for predicting the total duration of Porto taxi trips "
        "given partial GPS traces, start timestamp, call type, taxi stand, and taxi ID."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Enable Cross-Origin Resource Sharing (CORS) for external web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API endpoints
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
