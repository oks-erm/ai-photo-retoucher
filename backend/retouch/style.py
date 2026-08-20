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


@dataclass(frozen=True, slots=True)
class RegionTargetSpec:
    luminance_ratio: float = 1
    saturation_ratio: float = 1
    warmth_delta: float = 0
    luminance_bounds: tuple[float, float] = (0.02, 0.85)


@dataclass(frozen=True, slots=True)
class PresetTargetSpec:
    global_: RegionTargetSpec
    subject: RegionTargetSpec
    background: RegionTargetSpec
    skin: RegionTargetSpec
    foliage: RegionTargetSpec
    sky: RegionTargetSpec


PRESET_TARGETS: dict[StylePreset, PresetTargetSpec] = {
    StylePreset.GOLDEN_CINEMATIC: PresetTargetSpec(
        global_=RegionTargetSpec(1, 0.92, 0),
        subject=RegionTargetSpec(1.08, 1.12, 7, (0.28, 0.58)),
        background=RegionTargetSpec(0.61, 0.90, -1, (0.08, 0.30)),
        skin=RegionTargetSpec(1.03, 1.18, 9, (0.36, 0.64)),
        foliage=RegionTargetSpec(0.68, 0.81, -6, (0.06, 0.28)),
        sky=RegionTargetSpec(0.50, 2.05, 12, (0.16, 0.42)),
    ),
    StylePreset.MALICK_LUMINOUS: PresetTargetSpec(
        global_=RegionTargetSpec(0.94, 0.84, 1),
        subject=RegionTargetSpec(1.10, 1.08, 7, (0.30, 0.64)),
        background=RegionTargetSpec(0.78, 0.75, -3, (0.10, 0.42)),
        skin=RegionTargetSpec(1.05, 1.12, 8, (0.38, 0.66)),
        foliage=RegionTargetSpec(0.78, 0.70, -5, (0.07, 0.32)),
        sky=RegionTargetSpec(0.82, 0.80, 4, (0.20, 0.58)),
    ),
    StylePreset.COPPOLA_NOSTALGIC: PresetTargetSpec(
        global_=RegionTargetSpec(1.04, 0.78, 3),
        subject=RegionTargetSpec(1.12, 0.92, 5, (0.34, 0.68)),
        background=RegionTargetSpec(1.02, 0.68, 1, (0.16, 0.55)),
        skin=RegionTargetSpec(1.06, 1.05, 6, (0.40, 0.68)),
        foliage=RegionTargetSpec(1.00, 0.60, 4, (0.10, 0.38)),
        sky=RegionTargetSpec(1.03, 0.65, 3, (0.28, 0.72)),
    ),
    StylePreset.PRERAPHAELITE_ENCHANTED: PresetTargetSpec(
        global_=RegionTargetSpec(0.88, 0.96, 0),
        subject=RegionTargetSpec(1.10, 1.12, 9, (0.30, 0.62)),
        background=RegionTargetSpec(0.70, 0.88, -4, (0.08, 0.34)),
        skin=RegionTargetSpec(1.05, 1.20, 10, (0.38, 0.66)),
        foliage=RegionTargetSpec(0.72, 0.92, -6, (0.06, 0.30)),
        sky=RegionTargetSpec(0.72, 1.25, 4, (0.18, 0.52)),
    ),
    StylePreset.FAIRYTALE_TWILIGHT: PresetTargetSpec(
        global_=RegionTargetSpec(0.78, 0.82, -4),
        subject=RegionTargetSpec(1.08, 1.05, 8, (0.28, 0.60)),
        background=RegionTargetSpec(0.60, 0.78, -9, (0.06, 0.28)),
        skin=RegionTargetSpec(1.03, 1.12, 9, (0.36, 0.64)),
        foliage=RegionTargetSpec(0.62, 0.72, -10, (0.05, 0.25)),
        sky=RegionTargetSpec(0.62, 1.15, -6, (0.15, 0.44)),
    ),
}


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


def preset_targets(source: StyleMetrics, masks: MaskSet, style: StylePreset) -> StyleTargets:
    """Create scene-relative targets for a recognizable photographic preset."""
    spec = PRESET_TARGETS[style]

    def region(original: RegionMetrics, target: RegionTargetSpec) -> RegionMetrics:
        return RegionMetrics(
            luminance=_clamp(
                original.luminance * target.luminance_ratio,
                target.luminance_bounds[0],
                target.luminance_bounds[1],
            ),
            saturation=_clamp(original.saturation * target.saturation_ratio, 0.04, 0.72),
            warmth=_clamp(original.warmth + target.warmth_delta, -15, 35),
        )

    active = {"global", "background"}
    for name in ("subject", "skin", "foliage", "sky"):
        if masks.confidence[name] >= 0.45:
            active.add(name)
    return StyleTargets(
        metrics=StyleMetrics(
            global_=region(source.global_, spec.global_),
            subject=region(source.subject, spec.subject),
            background=region(source.background, spec.background),
            skin=region(source.skin, spec.skin),
            foliage=region(source.foliage, spec.foliage),
            sky=region(source.sky, spec.sky),
        ),
        active=frozenset(active),
    )


def golden_cinematic_targets(source: StyleMetrics, masks: MaskSet) -> StyleTargets:
    """Compatibility wrapper for the original calibrated preset."""
    return preset_targets(source, masks, StylePreset.GOLDEN_CINEMATIC)


def _infer_light_direction(rgb: FloatImage) -> str:
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    height, width = luminance.shape
    left = float(np.mean(luminance[: int(height * 0.72), : int(width * 0.32)]))
    right = float(np.mean(luminance[: int(height * 0.72), int(width * 0.68) :]))
    side = "left" if left >= right else "right"
    upper = float(np.mean(luminance[: int(height * 0.28)]))
    middle = float(np.mean(luminance[int(height * 0.28) : int(height * 0.62)]))
    return f"top_{side}" if upper > middle * 1.12 else side


def _seed_style(plan: EditPlan, style: StylePreset, light_direction: str) -> EditPlan:
    """Apply bounded optical/color signatures before metric convergence."""
    if style is StylePreset.GOLDEN_CINEMATIC:
        return plan.model_copy(
            update={
                "white_balance": plan.white_balance.model_copy(
                    update={"temperature_delta_k": 280, "tint_delta": 0.0}
                ),
                "global_": plan.global_.model_copy(
                    update={
                        "contrast": 0.10,
                        "black_depth": 0.08,
                        "saturation": -0.02,
                        "vibrance": 0.05,
                    }
                ),
                "highlights": plan.highlights.model_copy(update={"warmth": 0.12}),
                "shadows": plan.shadows.model_copy(update={"warmth": 0.02, "tint": 0.0}),
                "background": plan.background.model_copy(
                    update={"enabled": True, "saturation": -0.08, "warmth": -0.03}
                ),
                "foliage": plan.foliage.model_copy(
                    update={"green_lightness": -0.18, "green_chroma": -0.12, "yellow_chroma": -0.08}
                ),
                "skin": plan.skin.model_copy(update={"warmth": 0.10}),
                "atmosphere": plan.atmosphere.model_copy(
                    update={"bloom": 0.04, "edge_darkening": 0.10}
                )
            }
        )
    if style is StylePreset.MALICK_LUMINOUS:
        return plan.model_copy(
            update={
                "global_": plan.global_.model_copy(
                    update={
                        "black_depth": -0.05,
                        "saturation": -0.08,
                        "vibrance": 0.0,
                    }
                ),
                "white_balance": plan.white_balance.model_copy(
                    update={"temperature_delta_k": 120, "tint_delta": -0.01}
                ),
                "highlights": plan.highlights.model_copy(
                    update={
                        "recovery": max(plan.highlights.recovery, 0.32),
                        "warmth": 0.10,
                        "softness": max(plan.highlights.softness, 0.16),
                    }
                ),
                "shadows": plan.shadows.model_copy(
                    update={"warmth": -0.06, "tint": -0.04}
                ),
                "background": plan.background.model_copy(
                    update={"enabled": True, "saturation": -0.12, "warmth": -0.06}
                ),
                "foliage": plan.foliage.model_copy(
                    update={"green_lightness": -0.08, "green_chroma": -0.18, "yellow_chroma": -0.14}
                ),
                "skin": plan.skin.model_copy(update={"warmth": 0.10}),
                "atmosphere": plan.atmosphere.model_copy(
                    update={
                        "bloom": max(plan.atmosphere.bloom, 0.075),
                        "directional_haze": max(plan.atmosphere.directional_haze, 0.10),
                        "edge_darkening": max(plan.atmosphere.edge_darkening, 0.04),
                        "light_direction": light_direction,
                    }
                ),
            }
        )
    if style is StylePreset.COPPOLA_NOSTALGIC:
        return plan.model_copy(
            update={
                "global_": plan.global_.model_copy(
                    update={
                        "exposure_ev": max(plan.global_.exposure_ev, 0.12),
                        "contrast": -0.12,
                        "black_depth": -0.16,
                        "saturation": -0.12,
                        "vibrance": -0.04,
                    }
                ),
                "white_balance": plan.white_balance.model_copy(
                    update={"temperature_delta_k": 180, "tint_delta": 0.04}
                ),
                "highlights": plan.highlights.model_copy(
                    update={"recovery": max(plan.highlights.recovery, 0.52), "softness": 0.28}
                ),
                "shadows": plan.shadows.model_copy(
                    update={"lift": max(plan.shadows.lift, 0.18), "warmth": 0.01, "tint": 0.14}
                ),
                "background": plan.background.model_copy(
                    update={"enabled": True, "contrast": -0.12, "saturation": -0.16, "warmth": 0.01}
                ),
                "foliage": plan.foliage.model_copy(
                    update={"green_lightness": 0.02, "green_chroma": -0.22, "yellow_chroma": -0.12}
                ),
                "skin": plan.skin.model_copy(update={"warmth": 0.08, "chroma": -0.03}),
                "sharpening": plan.sharpening.model_copy(
                    update={"amount": min(plan.sharpening.amount, 0.06)}
                ),
                "atmosphere": plan.atmosphere.model_copy(
                    update={
                        "bloom": max(plan.atmosphere.bloom, 0.10),
                        "halation": max(plan.atmosphere.halation, 0.075),
                        "grain": max(plan.atmosphere.grain, 0.08),
                        "background_softness": max(plan.atmosphere.background_softness, 0.14),
                    }
                ),
            }
        )
    if style is StylePreset.PRERAPHAELITE_ENCHANTED:
        return plan.model_copy(
            update={
                "global_": plan.global_.model_copy(
                    update={
                        "contrast": 0.12,
                        "black_depth": 0.08,
                        "saturation": 0.02,
                        "vibrance": 0.10,
                    }
                ),
                "white_balance": plan.white_balance.model_copy(
                    update={"temperature_delta_k": 40, "tint_delta": -0.015}
                ),
                "highlights": plan.highlights.model_copy(update={"warmth": 0.14}),
                "shadows": plan.shadows.model_copy(
                    update={"warmth": -0.10, "tint": -0.08}
                ),
                "subject": plan.subject.model_copy(
                    update={"enabled": True, "contrast": 0.08, "warmth": 0.12}
                ),
                "background": plan.background.model_copy(
                    update={"enabled": True, "contrast": 0.10, "saturation": 0.04, "warmth": -0.08}
                ),
                "foliage": plan.foliage.model_copy(
                    update={"green_lightness": -0.10, "green_chroma": 0.10, "yellow_chroma": -0.08}
                ),
                "skin": plan.skin.model_copy(update={"warmth": 0.14, "chroma": 0.03}),
                "atmosphere": plan.atmosphere.model_copy(
                    update={
                        "bloom": max(plan.atmosphere.bloom, 0.045),
                        "edge_darkening": max(plan.atmosphere.edge_darkening, 0.18),
                    }
                ),
            }
        )
    if style is StylePreset.FAIRYTALE_TWILIGHT:
        return plan.model_copy(
            update={
                "white_balance": plan.white_balance.model_copy(
                    update={
                        "temperature_delta_k": -520,
                        "tint_delta": -0.025,
                    }
                ),
                "global_": plan.global_.model_copy(
                    update={
                        "contrast": 0.12,
                        "black_depth": 0.09,
                        "saturation": -0.08,
                        "vibrance": -0.02,
                    }
                ),
                "shadows": plan.shadows.model_copy(
                    update={"warmth": -0.16, "tint": -0.08}
                ),
                "subject": plan.subject.model_copy(
                    update={"enabled": True, "warmth": 0.28}
                ),
                "background": plan.background.model_copy(
                    update={"enabled": True, "saturation": -0.10, "warmth": -0.22}
                ),
                "foliage": plan.foliage.model_copy(
                    update={"green_lightness": -0.20, "green_chroma": -0.05, "yellow_chroma": -0.12}
                ),
                "skin": plan.skin.model_copy(update={"warmth": 0.22}),
                "atmosphere": plan.atmosphere.model_copy(
                    update={
                        "directional_haze": max(plan.atmosphere.directional_haze, 0.065),
                        "edge_darkening": max(plan.atmosphere.edge_darkening, 0.14),
                        "light_direction": light_direction,
                    }
                ),
            }
        )
    return plan


def _neutralise_model_style(plan: EditPlan) -> EditPlan:
    """Keep scene corrections, but remove the paid plan's preset contamination."""
    return plan.model_copy(
        update={
            "global_": plan.global_.model_copy(
                update={"black_depth": 0.0, "saturation": 0.0, "vibrance": 0.0}
            ),
            "white_balance": plan.white_balance.model_copy(
                update={"temperature_delta_k": 0, "tint_delta": 0.0}
            ),
            "highlights": plan.highlights.model_copy(update={"warmth": 0.0}),
            "shadows": plan.shadows.model_copy(update={"warmth": 0.0, "tint": 0.0}),
            "subject": plan.subject.model_copy(update={"saturation": 0.0, "warmth": 0.0}),
            "background": plan.background.model_copy(update={"saturation": 0.0, "warmth": 0.0}),
            "sky": plan.sky.model_copy(update={"enabled": False, "warmth": 0.0, "saturation": 0.0}),
            "foliage": plan.foliage.model_copy(
                update={"green_lightness": 0.0, "green_chroma": 0.0, "yellow_chroma": 0.0}
            ),
            "skin": plan.skin.model_copy(update={"warmth": 0.0, "chroma": 0.0}),
            "atmosphere": plan.atmosphere.model_copy(
                update={
                    "bloom": 0.0,
                    "halation": 0.0,
                    "grain": 0.0,
                    "background_softness": 0.0,
                    "directional_haze": 0.0,
                    "edge_darkening": 0.0,
                    "light_direction": "none",
                }
            ),
        }
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
                -1.0,
                1.0,
            ),
            "saturation": _clamp(
                plan.subject.saturation
                + (target.metrics.subject.saturation - current.subject.saturation) * 0.55,
                -0.20,
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
    white_balance = plan.white_balance.model_copy(
        update={
            "temperature_delta_k": int(
                _clamp(
                    plan.white_balance.temperature_delta_k
                    + (target.metrics.global_.warmth - current.global_.warmth) * 48,
                    -1500,
                    1500,
                )
            )
        }
    )
    return plan.model_copy(
        update={
            "global_": global_,
            "white_balance": white_balance,
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
        targets = preset_targets(measure_style(preview, masks), masks, style)
        current_plan = _seed_style(
            _neutralise_model_style(plan), style, _infer_light_direction(preview)
        )
        rendered, _, _ = self._engine.render_pixels(preview, masks, current_plan, style=style)
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
            candidate_render, _, _ = self._engine.render_pixels(
                preview, masks, candidate, style=style
            )
            candidate_distance = style_distance(measure_style(candidate_render, masks), targets)
            if candidate_distance >= distance - 0.002:
                completed -= 1
                break
            current_plan, rendered, distance = candidate, candidate_render, candidate_distance
        validated_plan = EditPlan.model_validate(current_plan.model_dump(by_alias=True))
        return StyleRefinement(validated_plan, completed, initial_distance, distance)
