from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Mode(StrEnum):
    TECHNICAL = "technical"
    PORTRAIT = "portrait"
    CREATIVE = "creative"


class SceneCategory(StrEnum):
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    ARCHITECTURE = "architecture"
    PRODUCT = "product"
    GENERAL = "general"


class Scene(StrictModel):
    category: SceneCategory
    lighting: str = Field(min_length=1, max_length=500)


class GlobalAdjustments(StrictModel):
    exposure_ev: Annotated[float, Field(ge=-1.5, le=1.5)] = 0
    contrast: Annotated[float, Field(ge=-0.30, le=0.30)] = 0
    black_depth: Annotated[float, Field(ge=-0.30, le=0.30)] = 0
    saturation: Annotated[float, Field(ge=-0.25, le=0.25)] = 0
    vibrance: Annotated[float, Field(ge=-0.25, le=0.25)] = 0


class WhiteBalance(StrictModel):
    temperature_delta_k: Annotated[int, Field(ge=-1500, le=1500)] = 0
    tint_delta: Annotated[float, Field(ge=-0.20, le=0.20)] = 0


class Highlights(StrictModel):
    recovery: Annotated[float, Field(ge=0, le=1)] = 0
    warmth: Annotated[float, Field(ge=-0.30, le=0.30)] = 0
    softness: Annotated[float, Field(ge=0, le=0.50)] = 0


class Shadows(StrictModel):
    lift: Annotated[float, Field(ge=-0.30, le=0.50)] = 0
    warmth: Annotated[float, Field(ge=-0.20, le=0.20)] = 0


class LocalAdjustments(StrictModel):
    enabled: bool = False
    exposure_ev: Annotated[float, Field(ge=-1.2, le=1.0)] = 0
    contrast: Annotated[float, Field(ge=-0.25, le=0.35)] = 0
    saturation: Annotated[float, Field(ge=-0.30, le=0.20)] = 0


class Foliage(StrictModel):
    green_lightness: Annotated[float, Field(ge=-0.30, le=0.20)] = 0
    green_chroma: Annotated[float, Field(ge=-0.35, le=0.20)] = 0
    yellow_chroma: Annotated[float, Field(ge=-0.30, le=0.20)] = 0


class Skin(StrictModel):
    protect: bool = True
    exposure: Annotated[float, Field(ge=-0.25, le=0.35)] = 0
    warmth: Annotated[float, Field(ge=-0.15, le=0.20)] = 0
    chroma: Annotated[float, Field(ge=-0.20, le=0.15)] = 0
    texture_softening: Annotated[float, Field(ge=0, le=0.20)] = 0


class Amount(StrictModel):
    amount: Annotated[float, Field(ge=0, le=0.35)] = 0


class Strength(StrictModel):
    strength: Annotated[float, Field(ge=0, le=0.60)] = 0


class Protections(StrictModel):
    preserve_highlights: bool = True
    preserve_deep_shadows: bool = True
    preserve_identity: Literal[True] = True
    preserve_geometry: Literal[True] = True


class EditPlan(StrictModel):
    version: Literal["1"] = "1"
    scene: Scene
    global_: GlobalAdjustments = Field(alias="global", serialization_alias="global")
    white_balance: WhiteBalance
    highlights: Highlights
    shadows: Shadows
    subject: LocalAdjustments
    background: LocalAdjustments
    foliage: Foliage
    skin: Skin
    sharpening: Amount
    denoise: Strength
    protections: Protections
    summary: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def enforce_local_ranges(self) -> "EditPlan":
        if self.subject.exposure_ev < -1.0:
            raise ValueError("subject.exposure_ev must be >= -1.0")
        if self.subject.contrast > 0.25 or self.subject.saturation < -0.20:
            raise ValueError("subject adjustments exceed their safety range")
        if self.background.exposure_ev > 0.6:
            raise ValueError("background.exposure_ev must be <= 0.6")
        return self


class DeltaPlan(StrictModel):
    accepted: bool
    global_exposure_delta: Annotated[float, Field(ge=-0.30, le=0.30)] = 0
    subject_exposure_delta: Annotated[float, Field(ge=-0.30, le=0.30)] = 0
    background_exposure_delta: Annotated[float, Field(ge=-0.30, le=0.30)] = 0
    temperature_delta_k: Annotated[int, Field(ge=-300, le=300)] = 0
    skin_warmth_delta: Annotated[float, Field(ge=-0.08, le=0.08)] = 0
    foliage_chroma_delta: Annotated[float, Field(ge=-0.10, le=0.10)] = 0
    highlight_warmth_delta: Annotated[float, Field(ge=-0.08, le=0.08)] = 0
    summary: str = Field(min_length=1, max_length=1000)


class AnalysisContext(StrictModel):
    mode: Mode = Mode.TECHNICAL
    intent: str = Field(default="Natural, technically correct photographic edit", max_length=2000)
    strength: Annotated[float, Field(ge=0, le=1)] = 0.5
    naturalness: Annotated[float, Field(ge=0, le=1)] = 0.8
    protect_skin: bool = True
    protect_highlights: bool = True
    preserve_deep_shadows: bool = True
    preserve_scene_colours: bool = True
    exif: dict[str, str | int | float | None] = Field(default_factory=dict)
    current_state: str = Field(default="", max_length=4000)


class AppliedOperation(StrictModel):
    module: str
    instance: str
    parameters: dict[str, float | int | str | bool]
    mask: str | None = None
    enabled: bool = True


class ApplyRequest(StrictModel):
    image_id: str = Field(min_length=1, max_length=512)
    image_path: str | None = None
    xmp_path: str | None = None
    edit_plan: EditPlan
    mask_confidence: Annotated[float | None, Field(ge=0, le=1)] = None


class ApplyResponse(StrictModel):
    session_id: str
    applied_operations: list[AppliedOperation]
    skipped_operations: list[str] = Field(default_factory=list)


class RevertRequest(StrictModel):
    session_id: str


class RevertResponse(StrictModel):
    session_id: str
    reverted: bool
