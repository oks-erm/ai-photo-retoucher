from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tifffile
from numpy.typing import NDArray

from backend.retouch.masks import MaskSet, build_masks
from backend.schemas import EditPlan

FloatImage = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class RenderResult:
    output_path: Path
    mask_paths: dict[str, Path]
    mask_confidence: dict[str, float]
    applied: list[str]
    skipped: list[str]
    warnings: list[str]
    input_bit_depth: int


def _srgb_to_linear(value: FloatImage) -> FloatImage:
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(value: FloatImage) -> FloatImage:
    value = np.clip(value, 0, 1)
    return np.where(value <= 0.0031308, value * 12.92, 1.055 * value ** (1 / 2.4) - 0.055)


def _luma(rgb: FloatImage) -> NDArray[np.float32]:
    return np.sum(rgb * np.array([0.2126, 0.7152, 0.0722], np.float32), axis=2)


def _blend(base: FloatImage, adjusted: FloatImage, mask: NDArray[np.float32]) -> FloatImage:
    return base * (1 - mask[..., None]) + adjusted * mask[..., None]


def _saturation(rgb: FloatImage, amount: float) -> FloatImage:
    luminance = _luma(rgb)[..., None]
    return np.clip(luminance + (rgb - luminance) * (1 + amount), 0, 1)


def _vibrance(rgb: FloatImage, amount: float) -> FloatImage:
    channel_range = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    adaptive = np.clip(1 - channel_range, 0, 1) * amount
    return _blend(rgb, _saturation(rgb, amount), adaptive)


def _temperature(rgb: FloatImage, kelvin_delta: int, tint: float) -> FloatImage:
    scale = float(kelvin_delta) / 1500
    multipliers = np.array(
        [1 + 0.12 * scale + 0.04 * tint, 1 - 0.02 * tint, 1 - 0.12 * scale - 0.04 * tint],
        np.float32,
    )
    return np.clip(rgb * multipliers, 0, 1)


def _tone(rgb: FloatImage, plan: EditPlan) -> FloatImage:
    linear = _srgb_to_linear(rgb)
    linear *= 2**plan.global_.exposure_ev
    luminance = _luma(linear)
    shadows = np.clip((0.42 - luminance) / 0.42, 0, 1) ** 1.6
    highlight_start = 0.52 - plan.highlights.softness * 0.28
    highlights = np.clip((luminance - highlight_start) / (1 - highlight_start), 0, 1) ** 1.4
    linear *= (1 + shadows * plan.shadows.lift * 1.45)[..., None]
    recovery = plan.highlights.recovery
    if recovery:
        compressed = linear / (1 + recovery * np.maximum(linear - 0.45, 0) * 2.2)
        linear = _blend(linear, compressed, highlights)
    contrast_power = 1 + plan.global_.contrast * 0.9
    numerator = np.clip(linear, 0, 1) ** contrast_power
    denominator = numerator + np.clip(1 - linear, 0, 1) ** contrast_power
    linear = np.divide(numerator, denominator, out=np.zeros_like(linear), where=denominator > 0)
    black = plan.global_.black_depth
    if black:
        low_tone_weight = (1 - np.clip(luminance / 0.42, 0, 1)) ** 2
        linear *= (1 - black * low_tone_weight)[..., None]
    return _linear_to_srgb(linear.astype(np.float32))


def _local_adjust(
    rgb: FloatImage, mask: NDArray[np.float32], exposure: float, contrast: float, saturation: float
) -> FloatImage:
    adjusted = _linear_to_srgb(_srgb_to_linear(rgb) * (2**exposure))
    contrast_power = max(0.72, 1 + contrast * 0.9)
    numerator = adjusted**contrast_power
    denominator = numerator + np.clip(1 - adjusted, 0, 1) ** contrast_power
    adjusted = np.divide(
        numerator, denominator, out=np.zeros_like(adjusted), where=denominator > 0
    )
    adjusted = _saturation(adjusted, saturation)
    return _blend(rgb, adjusted, mask)


def _warm_luminance_zone(rgb: FloatImage, amount: float, highlights: bool) -> FloatImage:
    if not amount:
        return rgb
    luminance = _luma(rgb)
    mask = (
        np.clip((luminance - 0.45) / 0.5, 0, 1)
        if highlights
        else np.clip((0.5 - luminance) / 0.5, 0, 1)
    )
    warmed = np.clip(
        rgb * np.array([1 + amount * 0.20, 1 + amount * 0.04, 1 - amount * 0.18], np.float32), 0, 1
    )
    return _blend(rgb, warmed, mask.astype(np.float32))


def _skin_retouch(
    rgb: FloatImage, masks: MaskSet, plan: EditPlan, applied: list[str], skipped: list[str]
) -> FloatImage:
    skin = masks.skin
    if masks.confidence["skin"] < 0.5:
        skipped.append("skin: no reliable skin region detected")
        return rgb
    result = rgb
    if plan.skin.exposure or plan.skin.chroma:
        result = _local_adjust(result, skin, plan.skin.exposure, 0, plan.skin.chroma)
        applied.append("skin tone and luminance")
    if plan.skin.warmth:
        warm = np.clip(
            result
            * np.array(
                [
                    1 + plan.skin.warmth * 0.18,
                    1 + plan.skin.warmth * 0.04,
                    1 - plan.skin.warmth * 0.15,
                ],
                np.float32,
            ),
            0,
            1,
        )
        result = _blend(result, warm, skin)
        applied.append("skin warmth")
    if plan.skin.texture_softening:
        u8 = np.clip(result * 255, 0, 255).astype(np.uint8)
        smooth = cv2.bilateralFilter(u8, 9, 24 + plan.skin.texture_softening * 100, 7)
        smooth_f = smooth.astype(np.float32) / 255
        # Preserve pores/high-frequency detail by mixing only a bounded fraction.
        strength = min(0.42, plan.skin.texture_softening * 2.1)
        result = _blend(result, smooth_f, skin * strength)
        applied.append("texture-preserving skin smoothing")
    return result


class RetouchEngine:
    def read(self, path: Path) -> tuple[FloatImage, int]:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"Unsupported or unreadable image: {path}")
        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
        elif raw.shape[2] == 4:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)
        else:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        bit_depth = 16 if raw.dtype == np.uint16 else 8
        maximum = float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype, np.integer) else 1.0
        return np.clip(raw.astype(np.float32) / maximum, 0, 1), bit_depth

    def render(
        self, input_path: Path, output_path: Path, plan: EditPlan, *, export_masks: bool = True
    ) -> RenderResult:
        rgb, input_bit_depth = self.read(input_path)
        masks = build_masks(rgb, portrait=plan.scene.category.value == "portrait")
        applied: list[str] = []
        skipped: list[str] = []
        warnings: list[str] = []

        result = _temperature(
            rgb, plan.white_balance.temperature_delta_k, plan.white_balance.tint_delta
        )
        if plan.white_balance.temperature_delta_k or plan.white_balance.tint_delta:
            applied.append("white balance")
        result = _tone(result, plan)
        applied.extend(["global exposure and contrast", "highlight recovery and shadow shaping"])
        result = _saturation(result, plan.global_.saturation)
        result = _vibrance(result, plan.global_.vibrance)
        if plan.global_.saturation or plan.global_.vibrance:
            applied.append("global colour intensity")
        result = _warm_luminance_zone(result, plan.highlights.warmth, True)
        result = _warm_luminance_zone(result, plan.shadows.warmth, False)
        if plan.highlights.warmth or plan.shadows.warmth:
            applied.append("split tonal warmth")

        if plan.subject.enabled:
            if masks.confidence["subject"] >= 0.45:
                result = _local_adjust(
                    result,
                    masks.subject,
                    plan.subject.exposure_ev,
                    plan.subject.contrast,
                    plan.subject.saturation,
                )
                applied.append("masked subject adjustment")
            else:
                skipped.append("subject: confidence below safe threshold")
        if plan.background.enabled:
            if masks.confidence["background"] >= 0.45:
                result = _local_adjust(
                    result,
                    masks.background,
                    plan.background.exposure_ev,
                    plan.background.contrast,
                    plan.background.saturation,
                )
                applied.append("masked background adjustment")
            else:
                skipped.append("background: confidence below safe threshold")

        foliage_amount = max(
            abs(plan.foliage.green_chroma),
            abs(plan.foliage.yellow_chroma),
            abs(plan.foliage.green_lightness),
        )
        if foliage_amount:
            foliage_edit = _local_adjust(
                result,
                masks.foliage,
                plan.foliage.green_lightness,
                0,
                (plan.foliage.green_chroma + plan.foliage.yellow_chroma) / 2,
            )
            result = foliage_edit
            applied.append("masked foliage colour")

        result = _skin_retouch(result, masks, plan, applied, skipped)
        if plan.denoise.strength:
            smooth = (
                cv2.bilateralFilter(
                    np.clip(result * 255, 0, 255).astype(np.uint8),
                    7,
                    12 + plan.denoise.strength * 55,
                    5,
                ).astype(np.float32)
                / 255
            )
            result = _blend(
                result,
                smooth,
                np.full(result.shape[:2], min(0.5, plan.denoise.strength), np.float32),
            )
            applied.append("edge-preserving denoise")
        if plan.sharpening.amount:
            blur = cv2.GaussianBlur(result, (0, 0), 1.2)
            result = np.clip(result + (result - blur) * plan.sharpening.amount * 1.7, 0, 1)
            applied.append("detail sharpening")

        if plan.protections.preserve_highlights:
            original_luma = _luma(rgb)
            result_luma = _luma(result)
            protection = np.clip((original_luma - 0.78) / 0.20, 0, 1) * np.clip(
                (result_luma - original_luma) * 8, 0, 1
            )
            result = _blend(result, rgb, protection.astype(np.float32))
        if plan.protections.preserve_deep_shadows:
            original_luma = _luma(rgb)
            shadow_guard = np.clip((0.035 - original_luma) / 0.035, 0, 1)
            result = _blend(result, rgb, shadow_guard.astype(np.float32) * 0.75)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_u16 = np.clip(result * 65535 + 0.5, 0, 65535).astype(np.uint16)
        tifffile.imwrite(output_path, output_u16, photometric="rgb", compression="deflate")

        mask_paths: dict[str, Path] = {}
        if export_masks:
            mask_dir = output_path.with_suffix("").with_name(output_path.stem + "-masks")
            mask_dir.mkdir(parents=True, exist_ok=True)
            for name in ("subject", "background", "skin", "face", "foliage", "sky"):
                mask_path = mask_dir / f"{name}.tif"
                tifffile.imwrite(
                    mask_path,
                    (getattr(masks, name) * 65535).astype(np.uint16),
                    compression="deflate",
                )
                mask_paths[name] = mask_path
        if input_bit_depth < 16:
            warnings.append(
                "input was 8-bit; output is 16-bit but cannot restore missing source precision"
            )
        return RenderResult(
            output_path, mask_paths, masks.confidence, applied, skipped, warnings, input_bit_depth
        )
