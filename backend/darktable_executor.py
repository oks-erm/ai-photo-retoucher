import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

from backend.mapper import MappingResult
from backend.schemas import ApplyRequest
from backend.session_store import SessionStore


class DarktableExecutor:
    """Transactional sidecar executor.

    Darktable's Lua surface does not expose every processing-module parameter reliably.
    This MVP writes a deterministic, auditable operation manifest next to a session and
    snapshots an existing XMP. The Lua bridge (or a future versioned XMP adapter) consumes
    the allow-listed operations; no model-produced code is executed.
    """

    def __init__(self, store: SessionStore) -> None:
        self._store = store

    @staticmethod
    def _resolve(path: str) -> Path:
        return Path(path).expanduser().resolve()

    async def apply(self, request: ApplyRequest, mapping: MappingResult) -> str:
        xmp_path = (
            await asyncio.to_thread(self._resolve, request.xmp_path) if request.xmp_path else None
        )
        snapshot: dict[str, Any] | None = None
        if xmp_path and await asyncio.to_thread(xmp_path.is_file):
            snapshot_path = xmp_path.with_suffix(f"{xmp_path.suffix}.ai-retoucher-backup")
            await asyncio.to_thread(shutil.copy2, xmp_path, snapshot_path)
            snapshot = {"source": str(xmp_path), "backup": str(snapshot_path)}

        record = {
            "image_id": request.image_id,
            "image_path": request.image_path,
            "xmp_snapshot": snapshot,
            "edit_plan": request.edit_plan.model_dump(by_alias=True),
            "operations": [operation.model_dump() for operation in mapping.operations],
            "skipped": mapping.skipped,
        }
        try:
            return await self._store.create(record)
        except Exception:
            if snapshot:
                await asyncio.to_thread(shutil.copy2, snapshot["backup"], snapshot["source"])
            raise

    async def revert(self, session_id: str) -> bool:
        record = await self._store.get(session_id)
        snapshot = record.get("xmp_snapshot")
        if snapshot:
            source = Path(snapshot["source"])
            backup = Path(snapshot["backup"])
            if await asyncio.to_thread(backup.is_file):
                temporary = source.with_suffix(f"{source.suffix}.reverting")
                await asyncio.to_thread(shutil.copy2, backup, temporary)
                await asyncio.to_thread(os.replace, temporary, source)
                await asyncio.to_thread(backup.unlink, missing_ok=True)
        await self._store.delete(session_id)
        return True


async def write_operation_manifest(path: Path, mapping: MappingResult) -> None:
    """Utility for Lua/helper integrations that need a JSON operation manifest."""
    payload = {
        "version": "1",
        "operations": [operation.model_dump() for operation in mapping.operations],
        "skipped": mapping.skipped,
    }
    await asyncio.to_thread(path.write_text, json.dumps(payload, indent=2), "utf-8")
