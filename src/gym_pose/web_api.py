"""Opt-in, no-recording source-label inference API (research only).

Consumes ONLY explicit JSON MediaPipe landmark vectors from consenting
browser users. No image, video, session history, database, or TEST data.
Models are NEVER fetched online; only operator-provisioned, trusted local
joblib artifacts are eligible. Web deployments using the Tasks-Vision
pose estimator have NOT been validated against the original Python
MediaPipe training distribution; scores are exploratory source labels.
"""
from __future__ import annotations

import os
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .model_registry import DEFAULT_PATHS, KNOWN_TASKS, ModelRegistry

MAX_JSON_BYTES = 16_384
EXERCISE_TASKS = {
    "squat": ("error_knees_forward", "error_knees_inward"),
    "ohp": ("error_elbows", "error_knees"),
    "barbell_row": ("torso_angle", "lumbar"),
}
DISCALIMER = (
    "Experimental publisher source-annotation association only; "
    "NOT exercise correctness, injury prevention, or medical advice"
)


class Landmark(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    x: float = Field(ge=-20, le=20, allow_inf_nan=False)
    y: float = Field(ge=-20, le=20, allow_inf_nan=False)
    z: float = Field(ge=-20, le=20, allow_inf_nan=False)
    visibility: float = Field(ge=0, le=1, allow_inf_nan=False)


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exercise: Literal["squat", "ohp", "barbell_row"]
    landmarks: list[Landmark] = Field(min_length=33, max_length=33)


def _allowed_origins() -> list[str]:
    raw = os.environ.get("POSE_WEB_ORIGINS", "http://localhost:5173")
    origins = [entry.strip().rstrip("/") for entry in raw.split(",") if entry.strip()]
    if any(not origin.startswith(("https://", "http://localhost:", "http://127.0.0.1:"))
           or "*" in origin or "/" in origin.split("://", 1)[-1]
           for origin in origins):
        raise ValueError("POSE_WEB_ORIGINS must contain explicit HTTPS/localhost origins only")
    return sorted(set(origins))


def create_app(registry: ModelRegistry | None = None, *,
               model_root: Path | None = None,
               trust_local_artifacts: bool | None = None) -> FastAPI:
    """Construct app with injectable synthetic model registry for tests.

    Production defaults to NO trust, NO automatic pickle loading.
    Owner enables trust after placing their OWN bundles on the backend.
    """
    if trust_local_artifacts is None:
        trust_local_artifacts = os.environ.get("POSE_TRUST_LOCAL_ARTIFACTS") == "1"
    if model_root is None:
        model_root = Path(os.environ.get("POSE_MODEL_ROOT", ".")).resolve()
    if registry is None:
        paths = {
            key: model_root / original
            for key, original in DEFAULT_PATHS.items()
        }
        registry = ModelRegistry(paths=paths)

    states = {}
    for exercise, target in KNOWN_TASKS:
        key = (exercise, target)
        p = Path(registry.paths[key]) if hasattr(registry, "paths") else None
        states[f"{exercise}/{target}"] = (
            "not_checked" if p is None or p.is_file() else "model_artifact_missing"
        )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        nonlocal states
        if trust_local_artifacts:
            # Only the server OPERATOR can opt into unsafe pickle execution.
            # Client request fields cannot set this. Never download models.
            states = registry.load_local(trust_local_pickle_files=True)
        # Log ONLY aggregate readiness, never artifact paths, tokens or landmarks.
        loaded = sum(v == "loaded_experimental_only" for v in states.values())
        logging.getLogger("uvicorn.error").info(
            "Gym Pose original-source research models loaded: %d/%d",
            loaded, len(KNOWN_TASKS),
        )
        application.state.registry = registry
        application.state.models = states
        yield

    app = FastAPI(
        title="Gym Pose research inference",
        description=DISCALIMER,
        version="0.1.0",
        docs_url=None, redoc_url=None, openapi_url=None,
        lifespan=lifespan,
    )
    origins = _allowed_origins()
    @app.exception_handler(RequestValidationError)
    async def redact_validation_errors(_request: Request, _exc: RequestValidationError):
        # FastAPI normally includes original invalid input in 422 responses.
        # Pose coordinates must never be echoed back to clients or logs.
        return JSONResponse(status_code=422, content={"detail": "Invalid request payload"})

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        max_age=3600,
    )

    @app.middleware("http")
    async def privacy_guard(request: Request, call_next):
        # Fail closed for API requests from another website even when made
        # by a client that bypasses browser CORS. This is NOT authentication.
        origin = request.headers.get("origin")
        if origin and origin not in origins:
            return JSONResponse(status_code=403, content={"detail": "Origin not permitted"})
        if request.url.path == "/score":
            try:
                declared = int(request.headers.get("content-length", "0"))
            except ValueError:
                return JSONResponse(status_code=413, content={"detail": "Invalid request size"})
            if declared > MAX_JSON_BYTES:
                return JSONResponse(status_code=413, content={"detail": "Request too large"})
            raw = await request.body()
            if len(raw) > MAX_JSON_BYTES:
                return JSONResponse(status_code=413, content={"detail": "Request too large"})
            # Reject nonstandard JSON NaN/Infinity before FastAPI's
            # validation exception renderer can echo nonfinite inputs.
            # Never echo user landmarks in error responses or access logs.
            def nonfinite_constant(_):
                raise ValueError("Nonfinite JSON number forbidden")
            try:
                json.loads(raw, parse_constant=nonfinite_constant)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                return JSONResponse(
                    status_code=422, content={"detail": "Invalid or nonfinite JSON"}
                )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/health")
    def health(request: Request):
        statuses = request.app.state.models
        loaded = sum(v == "loaded_experimental_only" for v in statuses.values())
        expected = len(KNOWN_TASKS)
        return {
            "status": ("research_api_ready" if loaded == expected
                       else "partial_model_artifacts" if loaded
                       else "model_artifacts_not_provisioned"),
            "ready_for_six_detector_demo": loaded == expected,
            "loaded_model_count": loaded,
            "expected_model_count": expected,
            "models": statuses,
            "pose_estimator_source_parity_validated": False,
            "research_only": True,
            "raw_camera_frames_processed": False,
            "request_landmarks_logged": False,
            "warning": DISCALIMER,
        }

    @app.get("/ready")
    def readiness(request: Request):
        """Deployment probe: never treat partial provisioning as full readiness."""
        loaded = sum(
            s == "loaded_experimental_only" for s in request.app.state.models.values()
        )
        if loaded != len(KNOWN_TASKS):
            raise HTTPException(status_code=503, detail="Six source models not ready")
        return {"status": "ready", "loaded_model_count": loaded, "research_only": True}

    @app.post("/score")
    def score(body: ScoreRequest, request: Request):
        if not trust_local_artifacts:
            raise HTTPException(
                status_code=503,
                detail="Model serving disabled until operator explicitly trusts local artifacts"
            )
        landmarks = [SimpleNamespace(**landmark.model_dump()) for landmark in body.landmarks]
        registry_instance = request.app.state.registry
        reports = {}
        for target in EXERCISE_TASKS[body.exercise]:
            try:
                result = registry_instance.score_landmarks(
                    body.exercise, target, landmarks, min_visibility=.5
                )
                if result.get("status") not in ("abstain", "experimental_source_label_score"):
                    raise ValueError("Unrecognized model result")
            except (ValueError, TypeError, RuntimeError, FloatingPointError):
                result = {
                    "exercise": body.exercise, "target": target,
                    "status": "abstain", "reason": "model_inference_error",
                    "error_probability": None, "threshold": None,
                    "source_label_flag": None, "experimental_only": True,
                }
            # Do not return model internals/estimator, video or raw pose.
            reports[target] = {
                "status": result["status"],
                "reason": result.get("reason"),
                "error_probability": result.get("error_probability"),
                "threshold": result.get("threshold"),
                "source_label_flag": result.get("source_label_flag"),
            }
        return {
            "status": "experimental_source_labels_only",
            "exercise": body.exercise,
            "targets": reports,
            "camera_frames_retained": 0,
            "pose_estimator_source_parity_validated": False,
            "warning": DISCALIMER,
        }

    return app


app = create_app()
