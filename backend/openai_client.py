import base64
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from backend.settings import Settings

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class VisionPlannerClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for model-backed analysis")
        self._model = settings.openai_model
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.request_timeout_seconds,
            max_retries=2,
        )

    async def parse(
        self,
        *,
        schema: type[SchemaT],
        instructions: str,
        prompt: str,
        images: list[bytes],
    ) -> SchemaT:
        content: list[dict[str, str]] = [{"type": "input_text", "text": prompt}]
        for image in images:
            encoded = base64.b64encode(image).decode("ascii")
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encoded}",
                    "detail": "high",
                }
            )
        response = await self._client.responses.parse(
            model=self._model,
            instructions=instructions,
            input=[{"role": "user", "content": content}],
            text_format=schema,
            store=False,
        )
        if response.output_parsed is None:
            raise RuntimeError("The model did not return a usable structured plan")
        return response.output_parsed
