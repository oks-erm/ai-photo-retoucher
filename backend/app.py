import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError

from backend.darktable_executor import DarktableExecutor
from backend.mapper import map_edit_plan
from backend.openai_client import VisionPlannerClient
from backend.planner import Planner
from backend.schemas import (
    AnalysisContext,
    ApplyRequest,
    ApplyResponse,
    DeltaPlan,
    EditPlan,
    RevertRequest,
    RevertResponse,
)
from backend.session_store import SessionStore
from backend.settings import Settings

MAX_PREVIEW_BYTES = 20 * 1024 * 1024


async def _read_preview(upload: UploadFile) -> bytes:
    if upload.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=415, detail="Preview must be JPEG or PNG")
    data = await upload.read(MAX_PREVIEW_BYTES + 1)
    if not data or len(data) > MAX_PREVIEW_BYTES:
        raise HTTPException(status_code=413, detail="Preview is empty or exceeds 20 MiB")
    return data


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    store = SessionStore(settings.resolved_session_dir)
    app.state.settings = settings
    app.state.executor = DarktableExecutor(store)
    app.state.planner = Planner(VisionPlannerClient(settings)) if settings.openai_api_key else None
    yield


app = FastAPI(
    title="Darktable AI Retoucher",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)


@app.get("/")
async def root(request: Request) -> dict[str, str | bool]:
    return {
        "name": "Darktable AI Retoucher",
        "status": "ok",
        "model_analysis_configured": request.app.state.planner is not None,
        "docs": "/docs",
    }


@app.get("/health")
async def health(request: Request) -> dict[str, str | bool]:
    return {"status": "ok", "model_analysis_configured": request.app.state.planner is not None}


@app.post("/v1/analyse", response_model=EditPlan)
async def analyse(
    request: Request,
    image: Annotated[UploadFile, File()],
    context: Annotated[str, Form()],
) -> EditPlan:
    planner: Planner | None = request.app.state.planner
    if planner is None:
        raise HTTPException(
            status_code=503, detail="Model analysis is disabled: configure OPENAI_API_KEY"
        )
    try:
        parsed_context = AnalysisContext.model_validate_json(context)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=json.loads(error.json())) from error
    return await planner.analyse(await _read_preview(image), parsed_context)


@app.post("/v1/critique", response_model=DeltaPlan)
async def critique(
    request: Request,
    original: Annotated[UploadFile, File()],
    edited: Annotated[UploadFile, File()],
    previous_plan: Annotated[str, Form()],
) -> DeltaPlan:
    planner: Planner | None = request.app.state.planner
    if planner is None:
        raise HTTPException(
            status_code=503, detail="Model analysis is disabled: configure OPENAI_API_KEY"
        )
    try:
        plan = EditPlan.model_validate_json(previous_plan)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=json.loads(error.json())) from error
    return await planner.critique(await _read_preview(original), await _read_preview(edited), plan)


@app.post("/v1/apply", response_model=ApplyResponse)
async def apply_edit(request: Request, body: ApplyRequest) -> ApplyResponse:
    settings: Settings = request.app.state.settings
    mapping = map_edit_plan(
        body.edit_plan,
        mask_confidence=body.mask_confidence,
        mask_threshold=settings.mask_confidence_threshold,
    )
    session_id = await request.app.state.executor.apply(body, mapping)
    return ApplyResponse(
        session_id=session_id,
        applied_operations=mapping.operations,
        skipped_operations=mapping.skipped,
    )


@app.post("/v1/revert", response_model=RevertResponse)
async def revert_edit(request: Request, body: RevertRequest) -> RevertResponse:
    try:
        reverted = await request.app.state.executor.revert(body.session_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Unknown edit session") from error
    return RevertResponse(session_id=body.session_id, reverted=reverted)


def run() -> None:
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8765, reload=False)
