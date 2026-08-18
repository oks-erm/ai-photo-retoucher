import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError

from backend.darktable_executor import DarktableExecutor
from backend.mapper import map_edit_plan
from backend.openai_client import VisionPlannerClient
from backend.planner import Planner
from backend.retouch import RetouchEngine
from backend.retouch.style import StyleRefiner
from backend.schemas import (
    AnalysisContext,
    ApplyRequest,
    ApplyResponse,
    DeltaPlan,
    EditPlan,
    MaskArtifact,
    RetouchReport,
    RetouchRequest,
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
    app.state.store = store
    app.state.executor = DarktableExecutor(store)
    app.state.retouch_engine = RetouchEngine()
    app.state.style_refiner = StyleRefiner(app.state.retouch_engine)
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


@app.get("/v1/plans/latest", response_model=EditPlan)
async def latest_plan(request: Request) -> EditPlan:
    try:
        record = await request.app.state.executor.latest()
    except KeyError as error:
        raise HTTPException(status_code=404, detail="No saved edit plan") from error
    # Reuse the untouched one-call model plan. This lets the photographer compare
    # several local presets without stacking one preset's convergence onto another.
    return EditPlan.model_validate(record.get("model_edit_plan", record["edit_plan"]))


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


@app.post("/v1/retouch", response_model=RetouchReport)
async def render_retouch(request: Request, body: RetouchRequest) -> RetouchReport:
    input_path, output_path = await asyncio.to_thread(
        lambda: (
            Path(body.input_path).expanduser().resolve(),
            Path(body.output_path).expanduser().resolve(),
        )
    )
    if not await asyncio.to_thread(input_path.is_file):
        raise HTTPException(status_code=404, detail="Input render does not exist")
    if input_path == output_path:
        raise HTTPException(status_code=422, detail="Output must not overwrite the input")
    if output_path.suffix.lower() not in {".tif", ".tiff"}:
        raise HTTPException(status_code=422, detail="Output must be a 16-bit TIFF")
    try:
        rgb, _ = await asyncio.to_thread(request.app.state.retouch_engine.read, input_path)
        refinement = await asyncio.to_thread(
            request.app.state.style_refiner.refine,
            rgb,
            body.edit_plan,
            style=body.style,
            maximum_passes=body.refinement_passes if body.auto_refine else 0,
        )
        result = await asyncio.to_thread(
            request.app.state.retouch_engine.render,
            input_path,
            output_path,
            refinement.plan,
            export_masks=body.export_masks,
        )
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    record = {
        "image_id": body.image_id,
        "image_path": str(input_path),
        "output_path": str(result.output_path),
        "model_edit_plan": body.edit_plan.model_dump(by_alias=True),
        "edit_plan": refinement.plan.model_dump(by_alias=True),
        "style": body.style,
        "refinement_passes": refinement.passes,
        "initial_style_distance": refinement.initial_distance,
        "final_style_distance": refinement.final_distance,
        "masks": {name: str(path) for name, path in result.mask_paths.items()},
        "applied": result.applied,
        "skipped": result.skipped,
        "warnings": result.warnings,
    }
    session_id = await request.app.state.store.create(record)
    return RetouchReport(
        session_id=session_id,
        output_path=str(result.output_path),
        masks=[
            MaskArtifact(name=name, path=str(path), confidence=result.mask_confidence[name])
            for name, path in result.mask_paths.items()
        ],
        applied=result.applied,
        skipped=result.skipped,
        warnings=result.warnings,
        final_plan=refinement.plan,
        refinement_passes=refinement.passes,
        initial_style_distance=refinement.initial_distance,
        final_style_distance=refinement.final_distance,
        input_bit_depth=result.input_bit_depth,
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
