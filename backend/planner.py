import json
from pathlib import Path

from backend.image_analysis import prepare_preview
from backend.openai_client import VisionPlannerClient
from backend.schemas import AnalysisContext, DeltaPlan, EditPlan, StylePreset

PROMPTS = Path(__file__).resolve().parent.parent / "prompts"


def _prompt(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8")


class Planner:
    def __init__(self, client: VisionPlannerClient) -> None:
        self._client = client

    async def analyse(self, image: bytes, context: AnalysisContext) -> EditPlan:
        preview, statistics = await prepare_preview(image)
        request = {
            "intent": context.intent,
            "mode": context.mode,
            "style": context.style,
            "strength": context.strength,
            "naturalness": context.naturalness,
            "protections": {
                "skin": context.protect_skin,
                "highlights": context.protect_highlights,
                "deep_shadows": context.preserve_deep_shadows,
                "scene_colours": context.preserve_scene_colours,
            },
            "exif": context.exif,
            "current_state": context.current_state,
            "statistics": statistics,
        }
        instructions = _prompt(context.mode.value)
        if context.style is StylePreset.GOLDEN_CINEMATIC:
            instructions += "\n\n" + _prompt("golden_cinematic")
        return await self._client.parse(
            schema=EditPlan,
            instructions=instructions,
            prompt=json.dumps(request, separators=(",", ":"), default=str),
            images=[preview],
        )

    async def critique(self, original: bytes, edited: bytes, previous_plan: EditPlan) -> DeltaPlan:
        original_preview, original_stats = await prepare_preview(original)
        edited_preview, edited_stats = await prepare_preview(edited)
        payload = {
            "previous_plan": previous_plan.model_dump(by_alias=True),
            "original_statistics": original_stats,
            "edited_statistics": edited_stats,
        }
        return await self._client.parse(
            schema=DeltaPlan,
            instructions=_prompt("critique"),
            prompt=json.dumps(payload, separators=(",", ":"), default=str),
            images=[original_preview, edited_preview],
        )
