from dataclasses import dataclass

from backend.schemas import AppliedOperation, EditPlan, Mode


@dataclass(frozen=True, slots=True)
class MappingResult:
    operations: list[AppliedOperation]
    skipped: list[str]


def _nonzero(**values: float | int | str | bool) -> dict[str, float | int | str | bool]:
    return {key: value for key, value in values.items() if value != 0}


def map_edit_plan(
    plan: EditPlan,
    *,
    mode: Mode = Mode.TECHNICAL,
    strength: float = 1,
    mask_confidence: float | None = None,
    mask_threshold: float = 0.72,
) -> MappingResult:
    scale = max(0.0, min(1.0, strength))
    operations: list[AppliedOperation] = []
    skipped: list[str] = []

    def scaled(value: float) -> float:
        return round(value * scale, 6)

    global_params = _nonzero(
        exposure=scaled(plan.global_.exposure_ev),
        contrast=scaled(plan.global_.contrast),
        black_depth=scaled(plan.global_.black_depth),
    )
    if global_params:
        operations.append(
            AppliedOperation(module="exposure", instance="AI global", parameters=global_params)
        )

    wb = _nonzero(
        temperature_delta_k=round(plan.white_balance.temperature_delta_k * scale),
        tint_delta=scaled(plan.white_balance.tint_delta),
    )
    if wb:
        operations.append(
            AppliedOperation(module="colorcalibration", instance="AI WB", parameters=wb)
        )

    tone = _nonzero(
        highlight_recovery=scaled(plan.highlights.recovery),
        highlight_softness=scaled(plan.highlights.softness),
        shadow_lift=scaled(plan.shadows.lift),
        preserve_highlights=plan.protections.preserve_highlights,
        preserve_deep_shadows=plan.protections.preserve_deep_shadows,
    )
    operations.append(AppliedOperation(module="toneequal", instance="AI tone", parameters=tone))

    colour = _nonzero(
        saturation=scaled(plan.global_.saturation),
        vibrance=scaled(plan.global_.vibrance),
        highlight_warmth=scaled(plan.highlights.warmth),
        shadow_warmth=scaled(plan.shadows.warmth),
    )
    if colour:
        operations.append(
            AppliedOperation(module="colorbalancergb", instance="AI colour", parameters=colour)
        )

    foliage = _nonzero(
        green_lightness=scaled(plan.foliage.green_lightness),
        green_chroma=scaled(plan.foliage.green_chroma),
        yellow_chroma=scaled(plan.foliage.yellow_chroma),
    )
    if foliage:
        operations.append(
            AppliedOperation(module="colorequal", instance="AI foliage", parameters=foliage)
        )

    local_allowed = mask_confidence is not None and mask_confidence >= mask_threshold
    for name, adjustment, mask in (
        ("subject", plan.subject, "subject"),
        ("background", plan.background, "background:inverse-subject"),
    ):
        if not adjustment.enabled:
            continue
        if not local_allowed:
            skipped.append(f"{name}: mask confidence unavailable or below threshold")
            continue
        operations.append(
            AppliedOperation(
                module="exposure",
                instance=f"AI {name}",
                mask=mask,
                parameters=_nonzero(
                    exposure=scaled(adjustment.exposure_ev),
                    contrast=scaled(adjustment.contrast),
                    saturation=scaled(adjustment.saturation),
                    feather_radius=24,
                ),
            )
        )

    if plan.skin.texture_softening and mode is not Mode.PORTRAIT:
        skipped.append("skin texture softening: portrait mode required")
    elif plan.skin.texture_softening:
        skipped.append("skin texture softening: face/skin masks are post-MVP")

    if plan.denoise.strength:
        operations.append(
            AppliedOperation(
                module="denoiseprofile",
                instance="AI denoise",
                parameters={"strength": scaled(plan.denoise.strength)},
            )
        )
    if plan.sharpening.amount:
        operations.append(
            AppliedOperation(
                module="diffuse",
                instance="AI sharpen",
                parameters={"sharpening": scaled(plan.sharpening.amount)},
            )
        )
    return MappingResult(operations=operations, skipped=skipped)
