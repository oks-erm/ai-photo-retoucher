import numpy as np
import pytest

from backend.retouch.engine import RetouchEngine
from backend.retouch.masks import MaskSet
from backend.retouch.style import (
    RegionMetrics,
    StyleMetrics,
    StyleRefiner,
    StyleTargets,
    _adjust_plan,
)
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


@pytest.mark.parametrize(
    ("style", "assert_signature"),
    [
        (
            StylePreset.MALICK_LUMINOUS,
            lambda result: (
                result.plan.atmosphere.directional_haze >= 0.10
                and result.plan.atmosphere.bloom >= 0.075
            ),
        ),
        (
            StylePreset.COPPOLA_NOSTALGIC,
            lambda result: (
                result.plan.atmosphere.grain >= 0.08 and result.plan.shadows.tint >= 0.12
            ),
        ),
        (
            StylePreset.PRERAPHAELITE_ENCHANTED,
            lambda result: (
                result.plan.atmosphere.edge_darkening >= 0.18
                and result.plan.global_.vibrance >= 0.06
            ),
        ),
        (
            StylePreset.FAIRYTALE_TWILIGHT,
            lambda result: (
                result.plan.white_balance.temperature_delta_k < 0
                and result.plan.subject.warmth >= 0.24
            ),
        ),
    ],
)
def test_named_presets_have_distinct_local_signatures(
    plan: EditPlan, monkeypatch, style: StylePreset, assert_signature
) -> None:
    rgb, masks = _portrait_fixture()
    monkeypatch.setattr("backend.retouch.style.build_masks", lambda *_args, **_kwargs: masks)

    result = StyleRefiner(RetouchEngine()).refine(
        rgb,
        plan,
        style=style,
        maximum_passes=2,
    )

    assert result.final_distance <= result.initial_distance
    assert assert_signature(result)


def test_named_presets_render_distinct_pixels(plan: EditPlan, monkeypatch) -> None:
    rgb, masks = _portrait_fixture()
    monkeypatch.setattr("backend.retouch.style.build_masks", lambda *_args, **_kwargs: masks)
    engine = RetouchEngine()
    outputs = []
    for style in StylePreset:
        if style is StylePreset.CUSTOM:
            continue
        refinement = StyleRefiner(engine).refine(
            rgb,
            plan,
            style=style,
            maximum_passes=2,
        )
        rendered, _, _ = engine.render_pixels(rgb, masks, refinement.plan)
        outputs.append(rendered)

    pairwise_differences = [
        float(np.mean(np.abs(left - right)))
        for index, left in enumerate(outputs)
        for right in outputs[index + 1 :]
    ]
    assert min(pairwise_differences) > 0.004


def test_internal_refinement_respects_public_subject_safety_contract(plan: EditPlan) -> None:
    base = RegionMetrics(luminance=0.4, saturation=0.4, warmth=0)
    current = StyleMetrics(
        global_=base,
        subject=RegionMetrics(luminance=0.8, saturation=0.8, warmth=0),
        background=base,
        skin=base,
        foliage=base,
        sky=base,
    )
    target = StyleTargets(
        metrics=StyleMetrics(
            global_=base,
            subject=RegionMetrics(luminance=0.1, saturation=0.1, warmth=0),
            background=base,
            skin=base,
            foliage=base,
            sky=base,
        ),
        active=frozenset({"global", "subject", "background"}),
    )
    boundary_plan = plan.model_copy(
        update={
            "subject": plan.subject.model_copy(update={"exposure_ev": -0.9, "saturation": -0.19})
        }
    )

    adjusted = _adjust_plan(boundary_plan, current, target)

    assert adjusted.subject.exposure_ev == -1.0
    assert adjusted.subject.saturation == -0.20
    assert EditPlan.model_validate(adjusted.model_dump(by_alias=True)) == adjusted
