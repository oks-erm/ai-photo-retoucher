import numpy as np

from backend.retouch.engine import RetouchEngine
from backend.retouch.masks import MaskSet
from backend.retouch.style import StyleRefiner
from backend.schemas import EditPlan, StylePreset


def _portrait_fixture() -> tuple[np.ndarray, MaskSet]:
    height, width = 120, 180
    rgb = np.full((height, width, 3), (0.38, 0.43, 0.30), np.float32)
    rgb[:72, 92:] = (0.62, 0.67, 0.72)  # pale sky
    rgb[74:] = (0.18, 0.30, 0.09)  # foliage

    subject = np.zeros((height, width), np.float32)
    subject[22:112, 54:108] = 1
    rgb[subject > 0] = (0.52, 0.40, 0.31)
    skin = np.zeros_like(subject)
    skin[28:52, 70:94] = 1
    face = skin.copy()
    foliage = np.zeros_like(subject)
    foliage[74:] = 1
    foliage *= 1 - subject
    sky = np.zeros_like(subject)
    sky[:72, 92:] = 1
    sky *= 1 - subject
    background = 1 - subject
    masks = MaskSet(
        subject=subject,
        background=background,
        skin=skin,
        face=face,
        foliage=foliage,
        sky=sky,
        confidence={
            name: 0.95 for name in ("subject", "background", "skin", "face", "foliage", "sky")
        },
    )
    return rgb, masks


def test_golden_style_refines_locally_without_another_model_call(
    plan: EditPlan, monkeypatch
) -> None:
    rgb, masks = _portrait_fixture()
    monkeypatch.setattr("backend.retouch.style.build_masks", lambda *_args, **_kwargs: masks)

    result = StyleRefiner(RetouchEngine()).refine(
        rgb,
        plan,
        style=StylePreset.GOLDEN_CINEMATIC,
        maximum_passes=4,
    )

    assert 1 <= result.passes <= 4
    assert result.final_distance < result.initial_distance
    assert result.plan.sky.enabled
    assert result.plan.background.exposure_ev < plan.background.exposure_ev


def test_custom_style_keeps_the_model_plan_exactly(plan: EditPlan) -> None:
    rgb, _ = _portrait_fixture()

    result = StyleRefiner(RetouchEngine()).refine(
        rgb,
        plan,
        style=StylePreset.CUSTOM,
        maximum_passes=4,
    )

    assert result.passes == 0
    assert result.plan == plan
