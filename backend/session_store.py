import asyncio
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class SessionStore:
    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._lock = asyncio.Lock()

    async def create(self, record: dict[str, Any]) -> str:
        async with self._lock:
            await asyncio.to_thread(self._directory.mkdir, parents=True, exist_ok=True, mode=0o700)
            session_id = uuid.uuid4().hex
            payload = {
                "session_id": session_id,
                "created_at": datetime.now(UTC).isoformat(),
                **record,
            }
            target = self._directory / f"{session_id}.json"
            temporary = target.with_suffix(".tmp")
            await asyncio.to_thread(temporary.write_text, json.dumps(payload, indent=2), "utf-8")
            await asyncio.to_thread(os.chmod, temporary, 0o600)
            await asyncio.to_thread(temporary.replace, target)
            return session_id

    async def get(self, session_id: str) -> dict[str, Any]:
        if not session_id.isalnum():
            raise KeyError(session_id)
        target = self._directory / f"{session_id}.json"
        try:
            content = await asyncio.to_thread(target.read_text, "utf-8")
        except FileNotFoundError as error:
            raise KeyError(session_id) from error
        return json.loads(content)

    async def latest(self) -> dict[str, Any]:
        def newest_record() -> dict[str, Any]:
            candidates = list(self._directory.glob("*.json"))
            if not candidates:
                raise KeyError("No saved edit plans")
            target = max(candidates, key=lambda path: path.stat().st_mtime_ns)
            return json.loads(target.read_text("utf-8"))

        return await asyncio.to_thread(newest_record)

    async def delete(self, session_id: str) -> None:
        target = self._directory / f"{session_id}.json"
        await asyncio.to_thread(target.unlink, missing_ok=True)
