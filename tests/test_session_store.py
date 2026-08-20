from pathlib import Path

import pytest

from backend.session_store import SessionStore


@pytest.mark.asyncio
async def test_session_round_trip(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    session_id = await store.create({"image_id": "42", "operations": []})
    record = await store.get(session_id)
    assert record["image_id"] == "42"
    await store.delete(session_id)
    with pytest.raises(KeyError):
        await store.get(session_id)


@pytest.mark.asyncio
async def test_latest_session(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    await store.create({"image_id": "first", "edit_plan": {}})
    await store.create({"image_id": "latest", "edit_plan": {}})
    assert (await store.latest())["image_id"] == "latest"
