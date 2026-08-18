from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="AI_RET_", extra="ignore", validate_default=True
    )

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = "gpt-5.6"
    debug: bool = False
    darktable_cli: str = "darktable-cli"
    session_dir: Path = Path("~/.cache/darktable-ai-retoucher/sessions")
    mask_confidence_threshold: float = Field(default=0.72, ge=0, le=1)
    request_timeout_seconds: float = Field(default=45, gt=0, le=180)

    @property
    def resolved_session_dir(self) -> Path:
        return self.session_dir.expanduser().resolve()
