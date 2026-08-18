from pathlib import Path

import numpy as np
import tifffile

from backend.retouch.engine import RetouchEngine
from backend.retouch.masks import MaskSet
from backend.schemas import EditPlan


def test_render_is_16_bit_editable_and_exports_masks(
    tmp_path: Path, plan: EditPlan, monkeypatch
) -> None:
    height, width = 96, 144
    source = np.zeros((height, width, 3), np.uint16)
    source[..., 0] = np.linspace(3000, 59000, width, dtype=np.uint16)
    source[..., 1] = 22000
    source[..., 2] = 16000
    input_path = tmp_path / "source.tif"
    output_path = tmp_path / "source-ai-retouched.tif"
    tifffile.imwrite(input_path, source, photometric="rgb")

    subject = np.zeros((height, width), np.float32)
    subject[18:90, 42:104] = 1
    skin = np.zeros_like(subject)
    skin[25:48, 60:84] = 1
    masks = MaskSet(
        subject=subject,
        background=1 - subject,
        skin=skin,
        face=skin,
        foliage=np.zeros_like(subject),
        sky=np.zeros_like(subject),
        confidence={
            "subject": 0.95,
            "background": 0.95,
            "skin": 0.9,
            "face": 0.9,
            "foliage": 0.0,
            "sky": 0.0,
        },
    )
    monkeypatch.setattr("backend.retouch.engine.build_masks", lambda *_args, **_kwargs: masks)

    result = RetouchEngine().render(input_path, output_path, plan)
    rendered = tifffile.imread(output_path)

    assert rendered.dtype == np.uint16
    assert rendered.shape == source.shape
    assert not np.array_equal(rendered, source)
    assert result.input_bit_depth == 16
    assert "masked subject adjustment" in result.applied
    assert "texture-preserving skin smoothing" in result.applied
    assert set(result.mask_paths) == {"subject", "background", "skin", "face", "foliage", "sky"}
    assert all(path.is_file() for path in result.mask_paths.values())


def test_low_confidence_mask_is_not_silently_applied(
    tmp_path: Path, plan: EditPlan, monkeypatch
) -> None:
    source = np.full((32, 48, 3), 24000, np.uint16)
    input_path, output_path = tmp_path / "in.tif", tmp_path / "out.tif"
    tifffile.imwrite(input_path, source, photometric="rgb")
    empty = np.zeros(source.shape[:2], np.float32)
    masks = MaskSet(
        subject=empty,
        background=1 - empty,
        skin=empty,
        face=empty,
        foliage=empty,
        sky=empty,
        confidence={
            name: 0.0 for name in ("subject", "background", "skin", "face", "foliage", "sky")
        },
    )
    monkeypatch.setattr("backend.retouch.engine.build_masks", lambda *_args, **_kwargs: masks)

    result = RetouchEngine().render(input_path, output_path, plan, export_masks=False)

    assert "subject: confidence below safe threshold" in result.skipped
    assert "background: confidence below safe threshold" in result.skipped
    assert "skin: no reliable skin region detected" in result.skipped


def test_sky_plan_is_rendered_through_its_mask(plan: EditPlan) -> None:
    height, width = 48, 72
    rgb = np.full((height, width, 3), (0.55, 0.62, 0.72), np.float32)
    empty = np.zeros((height, width), np.float32)
    sky = np.zeros_like(empty)
    sky[:, width // 2 :] = 1
    masks = MaskSet(
        subject=empty,
        background=np.ones_like(empty),
        skin=empty,
        face=empty,
        foliage=empty,
        sky=sky,
        confidence={
            "subject": 0,
            "background": 0,
            "skin": 0,
            "face": 0,
            "foliage": 0,
            "sky": 0.95,
        },
    )
    sky_plan = plan.model_copy(
        update={
            "global_": plan.global_.model_copy(
                update={"exposure_ev": 0, "contrast": 0, "black_depth": 0}
            ),
            "highlights": plan.highlights.model_copy(update={"recovery": 0, "warmth": 0}),
            "shadows": plan.shadows.model_copy(update={"lift": 0, "warmth": 0}),
            "subject": plan.subject.model_copy(update={"enabled": False}),
            "background": plan.background.model_copy(update={"enabled": False}),
            "skin": plan.skin.model_copy(
                update={"exposure": 0, "warmth": 0, "chroma": 0, "texture_softening": 0}
            ),
            "sharpening": plan.sharpening.model_copy(update={"amount": 0}),
            "denoise": plan.denoise.model_copy(update={"strength": 0}),
            "sky": plan.sky.model_copy(
                update={"enabled": True, "exposure_ev": -1, "warmth": 0.2, "saturation": 0.1}
            ),
        }
    )

    rendered, applied, _ = RetouchEngine().render_pixels(rgb, masks, sky_plan)

    assert rendered[:, width // 2 :].mean() < rendered[:, : width // 2].mean()
    assert "masked sky tone and warmth" in applied


def test_atmosphere_is_deterministic_and_keeps_full_resolution(plan: EditPlan) -> None:
    height, width = 64, 96
    rgb = np.full((height, width, 3), 0.34, np.float32)
    rgb[18:40, 35:61] = (0.82, 0.66, 0.48)
    subject = np.zeros((height, width), np.float32)
    subject[14:54, 30:66] = 1
    empty = np.zeros_like(subject)
    masks = MaskSet(
        subject=subject,
        background=1 - subject,
        skin=empty,
        face=empty,
        foliage=empty,
        sky=empty,
        confidence={
            "subject": 0.95,
            "background": 0.95,
            "skin": 0,
            "face": 0,
            "foliage": 0,
            "sky": 0,
        },
    )
    atmosphere_plan = plan.model_copy(
        update={
            "atmosphere": plan.atmosphere.model_copy(
                update={
                    "bloom": 0.12,
                    "halation": 0.08,
                    "grain": 0.10,
                    "background_softness": 0.12,
                    "directional_haze": 0.08,
                    "edge_darkening": 0.14,
                    "light_direction": "left",
                }
            )
        }
    )
    engine = RetouchEngine()

    first, applied, _ = engine.render_pixels(rgb, masks, atmosphere_plan)
    second, _, _ = engine.render_pixels(rgb, masks, atmosphere_plan)

    assert first.shape == rgb.shape
    assert np.array_equal(first, second)
    assert not np.allclose(first, rgb)
    assert "preset atmosphere and optical character" in applied
