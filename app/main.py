"""CI build-log line-ranking service.

POST /predict  -> top-K suspect lines of a failed Travis CI build log
GET  /health   -> liveness and model version

Contract version 1.0 is documented in README.md. Log content is never written to logs
or to error messages.
"""
import json
import logging
import os
import secrets
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import xgboost as xgb
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import incident_features as inf

logger = logging.getLogger("incident-ml-service")
SCHEMA_VERSION = "1.0"


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class BoosterAdapter:
    """Loads the saved XGBoost JSON with the core API (no scikit-learn needed) and returns probabilities."""

    def __init__(self, path: Path):
        self.booster = xgb.Booster()
        self.booster.load_model(str(path))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.booster.predict(xgb.DMatrix(X))
        return np.column_stack([1.0 - p, p])


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.environ.get("MODEL_DIR", "model"))
    meta = json.loads((model_dir / "metadata.json").read_text())
    model = BoosterAdapter(model_dir / "line_model.json")
    features = list(meta["features"])
    model.predict_proba(np.zeros((1, len(features)), dtype=np.float32))      # fail fast if the model and metadata disagree
    trained_with = meta.get("xgboost_version")
    if trained_with and trained_with != xgb.__version__:
        logger.warning("xgboost runtime %s differs from training version %s", xgb.__version__, trained_with)
    app.state.meta, app.state.model, app.state.features = meta, model, features
    app.state.slots = threading.BoundedSemaphore(_int_env("MAX_CONCURRENT", 2))
    logger.info("model %s loaded with %d features", meta.get("model_version"), len(features))
    yield


app = FastAPI(title="Incident log line ranker", version=SCHEMA_VERSION, lifespan=lifespan, redoc_url=None,
              docs_url=None if os.environ.get("DISABLE_DOCS") == "1" else "/docs",
              openapi_url=None if os.environ.get("DISABLE_DOCS") == "1" else "/openapi.json")


class PredictRequest(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    log_text: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)


class LineScore(BaseModel):
    line_no: int
    score: float
    section: str
    text: str


class PredictResponse(BaseModel):
    schema_version: str
    model_version: str
    n_lines: int
    n_content_lines: int
    travis_fold_markers: int
    top_lines: List[LineScore]
    warnings: List[str]


def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = os.environ.get("MODEL_API_KEY", "")
    if not expected:
        if os.environ.get("ALLOW_NO_AUTH") == "1":          # local development only
            return
        raise HTTPException(status_code=503, detail="service is not configured")
    if not x_api_key or not secrets.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    cl = request.headers.get("content-length")
    if request.method == "POST" and cl and cl.isdigit() and int(cl) > _int_env("MAX_BODY_BYTES", 8_000_000):
        return JSONResponse(status_code=413, content={"detail": "request too large"})
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # FastAPI's default response echoes the offending input, which here could be log content.
    errors = [{"loc": [str(p) for p in e.get("loc", [])], "msg": str(e.get("msg", "invalid"))} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health")
def health(request: Request):
    st = request.app.state
    return {"status": "ok", "schema_version": SCHEMA_VERSION,
            "model_version": st.meta.get("model_version", "unknown"), "n_features": len(st.features)}


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_key)])
def predict(req: PredictRequest, request: Request):
    st = request.app.state
    text = req.log_text
    if len(text) > _int_env("MAX_BODY_BYTES", 8_000_000) or text.count("\n") >= _int_env("MAX_LINES", 200_000):
        raise HTTPException(status_code=413, detail="log too large")
    if not st.slots.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="service busy, retry shortly", headers={"Retry-After": "5"})
    try:
        t0, rid = time.perf_counter(), uuid.uuid4().hex[:8]
        n_lines = text.count("\n") + 1
        folds = text.count("travis_fold:start")
        warnings: List[str] = []
        if folds == 0:
            warnings.append("No Travis CI fold markers found; this log may be outside the format the model was trained on.")
        c = inf.build_features(text)
        if len(c) == 0:
            warnings.append("The log has no content lines.")
            top: List[LineScore] = []
        else:
            scores = st.model.predict_proba(inf.feature_matrix(c, st.features))[:, 1]
            order = np.argsort(-scores, kind="stable")[: req.top_k]
            top = [LineScore(line_no=int(c["line_no"].iloc[i]), score=float(scores[i]),
                             section=str(c["section"].iloc[i]), text=c["text"].iloc[i][:300]) for i in order]
        logger.info("predict id=%s lines=%d content=%d ms=%d", rid, n_lines, len(c), (time.perf_counter() - t0) * 1000)
        return PredictResponse(schema_version=SCHEMA_VERSION, model_version=st.meta.get("model_version", "unknown"),
                               n_lines=n_lines, n_content_lines=len(c), travis_fold_markers=folds,
                               top_lines=top, warnings=warnings)
    finally:
        st.slots.release()
