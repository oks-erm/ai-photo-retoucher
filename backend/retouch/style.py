from dataclasses import dataclass
from math import log2

import cv2
import numpy as np
from numpy.typing import NDArray

from backend.retouch.engine import FloatImage, RetouchEngine
from backend.retouch.masks import MaskSet, build_masks
from backend.schemas import EditPlan, StylePreset


@dataclass(frozen=True, slots=True)
class RegionMetrics:
    luminance: float
    saturation: float
    warmth: float


@dataclass(frozen=True, slots=True)
class StyleMetrics:
    global_: RegionMetrics
    subject: RegionMetrics
    background: RegionMetrics
    skin: RegionMetrics
    foliage: RegionMetrics
    sky: RegionMetrics


@dataclass(frozen=True, slots=True)
class StyleTargets:
    metrics: StyleMetrics
    active: frozenset[str]


@dataclass(frozen=True, slots=True)
class StyleRefinement:
    plan: EditPlan
    passes: int
    initial_distance: float
    final_distance: float


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _region_metrics(rgb: FloatImage, mask: NDArray[np.float32]) -> RegionMetrics:
    weights = np.clip(mask, 0, 1)
    total = max(float(weights.sum()), 1e-6)
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    hsv = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)
    return RegionMetrics(
        luminance=float(np.sum(luminance * weights) / total),
        saturation=float(np.sum(hsv[..., 1] * weights) / total),
        warmth=float(np.sum(lab[..., 2] * weights) / total),
    )


def measure_style(rgb: FloatImage, masks: MaskSet) -> StyleMetrics:
    full = np.ones(rgb.shape[:2], np.float32)
    return StyleMetrics(
        global_=_region_metrics(rgb, full),
        subject=_region_metrics(rgb, masks.subject),
        background=_region_metrics(rgb, masks.background),
        skin=_region_metrics(rgb, masks.skin),
        foliage=_region_metrics(rgb, masks.foliage),
        sky=_region_metrics(rgb, masks.sky),
    )


def golden_cinematic_targets(source: StyleMetrics, masks: MaskSet) -> StyleTargets:
    """Transferable ratios measured from the supplied before/after style reference.

    The values describe relationships rather than image coordinates or absolute output
    pixels, so they adapt to the exposure and palette of each new photograph.
    """

    def region(
        original: RegionMetrics,
        *,
        luminance_ratio: float = 1,
        saturation_ratio: float = 1,
        warmth_delta: float = 0,
        luminance_bounds: tuple[float, float] = (0.02, 0.85),
    ) -> RegionMetrics:
        return RegionMetrics(
            luminance=_clamp(
                original.luminance * luminance_ratio,
                luminance_bounds[0],
                luminance_bounds[1],
            ),
            saturation=_clamp(original.saturation * saturation_ratio, 0.04, 0.72),
            warmth=_clamp(original.warmth + warmth_delta, -15, 35),
        )

    active = {"global", "background"}
    for name in ("subject", "skin", "foliage", "sky"):
        if masks.confidence[name] >= 0.45:
            active.add(name)
    return StyleTargets(
        metrics=StyleMetrics(
            global_=region(source.global_, saturation_ratio=0.92),
            subject=region(
                source.subject,
                luminance_ratio=1.08,
                saturation_ratio=1.12,
                warmth_delta=7,
                luminance_bounds=(0.28, 0.58),
            ),
            background=region(
                source.background,
                luminance_ratio=0.61,
                saturation_ratio=0.90,
                warmth_delta=-1,
                luminance_bounds=(0.08, 0.30),
            ),
            skin=region(
                source.skin,
                luminance_ratio=1.03,
                saturation_ratio=1.18,
                warmth_delta=9,
                luminance_bounds=(0.36, 0.64),
            ),
            foliage=region(
                source.foliage,
                luminance_ratio=0.68,
                saturation_ratio=0.81,
                warmth_delta=-6,
                luminance_bounds=(0.06, 0.28),
            ),
            sky=region(
                source.sky,
                luminance_ratio=0.50,
                saturation_ratio=2.05,
                warmth_delta=12,
                luminance_bounds=(0.16, 0.42),
            ),
        ),
        active=frozenset(active),
    )


def style_distance(current: StyleMetrics, target: StyleTargets) -> float:
    terms: list[float] = []
    for name in target.active:
        current_region = current.global_ if name == "global" else getattr(current, name)
        target_region = (
            target.metrics.global_ if name == "global" else getattr(target.metrics, name)
        )
        terms.append(abs(log2(max(current_region.luminance, 0.01) / target_region.luminance)))
        terms.append(abs(current_region.saturation - target_region.saturation) / 0.20)
        if name in {"subject", "skin", "foliage", "sky"}:
            terms.append(abs(current_region.warmth - target_region.warmth) / 14)
    return float(np.mean(terms)) if terms else 0.0


def _exposure_correction(current: float, target: float, gain: float = 0.68) -> float:
    return _clamp(log2(max(target, 0.01) / max(current, 0.01)) * gain, -0.45, 0.45)


def _adjust_plan(plan: EditPlan, current: StyleMetrics, target: StyleTargets) -> EditPlan:
    subject = plan.subject.model_copy(
        update={
            "enabled": "subject" in target.active,
            "exposure_ev": _clamp(
                plan.subject.exposure_ev
                + _exposure_correction(current.subject.luminance, target.metrics.subject.luminance),
                -1.2,
                1.0,
            ),
            "saturation": _clamp(
                plan.subject.saturation
                + (target.metrics.subject.saturation - current.subject.saturation) * 0.55,
                -0.30,
                0.20,
            ),
            "warmth": _clamp(
                plan.subject.warmth + (target.metrics.subject.warmth - current.subject.warmth) / 28,
                -0.30,
                0.50,
            ),
        }
    )
    background = plan.background.model_copy(
        update={
            "enabled": True,
            "exposure_ev": _clamp(
                plan.background.exposure_ev
                + _exposure_correction(
                    current.background.luminance, target.metrics.background.luminance
                ),
                -1.2,
                0.6,
            ),
            "saturation": _clamp(
                plan.background.saturation
                + (target.metrics.background.saturation - current.background.saturation) * 0.55,
                -0.30,
                0.20,
            ),
            "warmth": _clamp(
                plan.background.warmth
                + (target.metrics.background.warmth - current.background.warmth) / 32,
                -0.30,
                0.50,
            ),
        }
    )
    sky = plan.sky.model_copy(
        update={
            "enabled": "sky" in target.active,
            "exposure_ev": _clamp(
                plan.sky.exposure_ev
                + _exposure_correction(current.sky.luminance, target.metrics.sky.luminance),
                -1.5,
                0.5,
            ),
            "saturation": _clamp(
                plan.sky.saturation
                + (target.metrics.sky.saturation - current.sky.saturation) * 0.5,
                -0.30,
                0.75,
            ),
            "warmth": _clamp(
                plan.sky.warmth + (target.metrics.sky.warmth - current.sky.warmth) / 32,
                -0.30,
                0.80,
            ),
        }
    )
    foliage = plan.foliage.model_copy(
        update={
            "green_lightness": _clamp(
                plan.foliage.green_lightness
                + _exposure_correction(
                    current.foliage.luminance, target.metrics.foliage.luminance, 0.45
                ),
                -0.75,
                0.35,
            ),
            "green_chroma": _clamp(
                plan.foliage.green_chroma
                + (target.metrics.foliage.saturation - current.foliage.saturation) * 0.7,
                -0.35,
                0.20,
            ),
            "yellow_chroma": _clamp(
                plan.foliage.yellow_chroma
                + (target.metrics.foliage.saturation - current.foliage.saturation) * 0.45,
                -0.30,
                0.20,
            ),
        }
    )
    skin = plan.skin.model_copy(
        update={
            "exposure": _clamp(
                plan.skin.exposure
                + _exposure_correction(current.skin.luminance, target.metrics.skin.luminance, 0.45),
                -0.25,
                0.35,
            ),
            "warmth": _clamp(
                plan.skin.warmth + (target.metrics.skin.warmth - current.skin.warmth) / 38,
                -0.15,
                0.40,
            ),
            "chroma": _clamp(
                plan.skin.chroma
                + (target.metrics.skin.saturation - current.skin.saturation) * 0.35,
                -0.20,
                0.15,
            ),
        }
    )
    global_ = plan.global_.model_copy(
        update={
            "saturation": _clamp(
                plan.global_.saturation
                + (target.metrics.global_.saturation - current.global_.saturation) * 0.38,
                -0.25,
                0.25,
            )
        }
    )
    return plan.model_copy(
        update={
            "global_": global_,
            "subject": subject,
            "background": background,
            "sky": sky,
            "foliage": foliage,
            "skin": skin,
        }
    )


def _preview(rgb: FloatImage, maximum: int = 960) -> FloatImage:
    height, width = rgb.shape[:2]
    scale = min(1.0, maximum / max(height, width))
    if scale == 1:
        return rgb
    return cv2.resize(
        rgb, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
    )


class StyleRefiner:
    def __init__(self, engine: RetouchEngine) -> None:
        self._engine = engine

    def refine(
        self,
        rgb: FloatImage,
        plan: EditPlan,
        *,
        style: StylePreset,
        maximum_passes: int,
    ) -> StyleRefinement:
        if style is StylePreset.CUSTOM or maximum_passes == 0:
            return StyleRefinement(plan, 0, 0, 0)
        preview = _preview(rgb)
        masks = build_masks(preview, portrait=plan.scene.category.value == "portrait")
        targets = golden_cinematic_targets(measure_style(preview, masks), masks)
        current_plan = plan
        rendered, _, _ = self._engine.render_pixels(preview, masks, current_plan)
        distance = style_distance(measure_style(rendered, masks), targets)
        initial_distance = distance
        completed = 0
        for completed in range(1, maximum_passes + 1):
            if distance <= 0.12:
                completed -= 1
                break
            candidate = _adjust_plan(
                current_plan,
                measure_style(rendered, masks),
                targets,
            )
            candidate_render, _, _ = self._engine.render_pixels(preview, masks, candidate)
            candidate_distance = style_distance(measure_style(candidate_render, masks), targets)
            if candidate_distance >= distance - 0.002:
                completed -= 1
                break
            current_plan, rendered, distance = candidate, candidate_render, candidate_distance
        return StyleRefinement(current_plan, completed, initial_distance, distance)
