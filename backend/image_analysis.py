import asyncio
from dataclasses import asdict, dataclass
from io import BytesIO

from PIL import Image, ImageStat


@dataclass(frozen=True, slots=True)
class ImageStatistics:
    width: int
    height: int
    mean_luminance: float
    p01_luminance: float
    p99_luminance: float
    shadow_clip_fraction: float
    highlight_clip_fraction: float


def _prepare(image_bytes: bytes, long_edge: int = 2048) -> tuple[bytes, ImageStatistics]:
    with Image.open(BytesIO(image_bytes)) as source:
        image = source.convert("RGB")
        image.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
        luminance = image.convert("L")
        histogram = luminance.histogram()
        total = max(sum(histogram), 1)

        def percentile(fraction: float) -> int:
            target = total * fraction
            running = 0
            for value, count in enumerate(histogram):
                running += count
                if running >= target:
                    return value
            return 255

        statistics = ImageStatistics(
            width=image.width,
            height=image.height,
            mean_luminance=round(ImageStat.Stat(luminance).mean[0] / 255, 4),
            p01_luminance=round(percentile(0.01) / 255, 4),
            p99_luminance=round(percentile(0.99) / 255, 4),
            shadow_clip_fraction=round(sum(histogram[:3]) / total, 6),
            highlight_clip_fraction=round(sum(histogram[253:]) / total, 6),
        )
        output = BytesIO()
        image.save(output, format="JPEG", quality=90, optimize=True)
        return output.getvalue(), statistics


async def prepare_preview(image_bytes: bytes) -> tuple[bytes, dict[str, int | float]]:
    preview, statistics = await asyncio.to_thread(_prepare, image_bytes)
    return preview, asdict(statistics)
