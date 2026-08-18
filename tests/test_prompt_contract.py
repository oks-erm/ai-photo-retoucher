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


def test_all_named_style_presets_have_model_guidance() -> None:
    prompt_dir = Path(__file__).parents[1] / "prompts"
    for name in (
        "golden_cinematic",
        "malick_luminous",
        "coppola_nostalgic",
        "preraphaelite_enchanted",
        "fairytale_twilight",
    ):
        text = (prompt_dir / f"{name}.md").read_text(encoding="utf-8").lower()
        assert "preset" in text or "style" in text
        assert "identity" in text


def test_lua_exposes_every_style_without_requiring_expert_intent() -> None:
    text = (Path(__file__).parents[1] / "lua" / "ai_retoucher.lua").read_text(encoding="utf-8")
    for style in (
        "golden_cinematic",
        "malick_luminous",
        "coppola_nostalgic",
        "preraphaelite_enchanted",
        "fairytale_twilight",
        "custom",
    ):
        assert f'"{style}"' in text
    assert "Apply the selected preset faithfully" in text


def test_lua_bridge_uses_full_resolution_local_render() -> None:
    text = (Path(__file__).parents[1] / "lua" / "ai_retoucher.lua").read_text(encoding="utf-8")
    assert 'API .. "/v1/retouch"' in text
    assert "plugins/imageio/format/tiff/bpp=16" in text
    assert "--width 0 --height 0 --hq true" in text
    assert '"--width 0 --height 0 --hq true --out-ext tif",' in text
    assert '"--conf plugins/imageio/format/tiff/bpp=16")' in text
    assert 'export_options .. " --core --configdir "' in text
    assert '" --library :memory: " .. (core_options or "")' in text
    assert '" " .. options .. " --core' not in text
    assert "dt.database.import" in text
    assert "pcall(dt.gui.action" not in text


def test_lua_bridge_is_one_click_and_renders_its_own_preview() -> None:
    text = (Path(__file__).parents[1] / "lua" / "ai_retoucher.lua").read_text(encoding="utf-8")
    assert 'label = "Analyse & Retouch"' in text
    assert "darktable-cli" in text
    assert "preview_path" not in text
    assert '" --core --configdir " .. quote(config_dir)' in text
    assert '" --library :memory:' in text
    assert 'label = "Retouch with saved plan (free)"' in text
    assert 'API .. "/v1/plans/latest"' in text
    assert "Loading original model plan" in text


def test_lua_sliders_use_darktable_supported_bounds() -> None:
    text = (Path(__file__).parents[1] / "lua" / "ai_retoucher.lua").read_text(encoding="utf-8")
    assert "soft_min = 0" in text
    assert "soft_max = 1" in text
    assert "hard_min = 0" in text
    assert "hard_max = 1" in text
    assert 'label = "Strength", min =' not in text
    assert 'label = "Naturalness", min =' not in text


def test_macos_installer_registers_plugin_and_backend() -> None:
    text = (Path(__file__).parents[1] / "scripts" / "install_macos.sh").read_text(encoding="utf-8")
    assert 'require "ai_retoucher"' in text
    assert "LaunchAgents" in text
    assert "uv sync" in text
