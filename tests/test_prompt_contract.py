from pathlib import Path


def test_prompts_forbid_code_and_preserve_identity() -> None:
    prompt_dir = Path(__file__).parents[1] / "prompts"
    for name in ("technical.md", "portrait.md", "creative.md"):
        text = (prompt_dir / name).read_text(encoding="utf-8").lower()
        assert "never output code" in text
        assert "identity" in text


def test_critique_requires_delta_plan() -> None:
    text = (Path(__file__).parents[1] / "prompts" / "critique.md").read_text(encoding="utf-8")
    assert "DeltaPlan" in text
    assert "do not return a replacement EditPlan" in text


def test_lua_bridge_applies_real_darktable_actions() -> None:
    text = (Path(__file__).parents[1] / "lua" / "ai_retoucher.lua").read_text(encoding="utf-8")
    assert 'dt.gui.action, path, 0, "value", "set"' in text
    assert "apply_plan_to_darktable(last_plan)" in text
    assert "restore_darktable(applied_state)" in text
