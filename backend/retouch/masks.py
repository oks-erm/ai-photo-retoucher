from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

FloatImage = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class MaskSet:
    subject: FloatImage
    background: FloatImage
    skin: FloatImage
    face: FloatImage
    foliage: FloatImage
    sky: FloatImage
    confidence: dict[str, float]


def _feather(mask: FloatImage, radius: int) -> FloatImage:
    if radius <= 0:
        return np.clip(mask, 0, 1).astype(np.float32)
    kernel = radius * 2 + 1
    return np.clip(cv2.GaussianBlur(mask, (kernel, kernel), 0), 0, 1).astype(np.float32)


def _largest_components(mask: NDArray[np.uint8], count: int = 6) -> NDArray[np.uint8]:
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if labels_count <= 1:
        return mask
    keep = sorted(
        range(1, labels_count), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True
    )
    result = np.zeros_like(mask)
    for label in keep[:count]:
        if stats[label, cv2.CC_STAT_AREA] >= max(24, mask.size // 30000):
            result[labels == label] = 255
    return result


def _smoothstep(values: FloatImage, lower: float, upper: float) -> FloatImage:
    scaled = np.clip((values - lower) / (upper - lower), 0, 1)
    return (scaled * scaled * (3 - 2 * scaled)).astype(np.float32)


def _skin_mask(rgb: FloatImage) -> NDArray[np.uint8]:
    image = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    y, cr, cb = cv2.split(ycrcb)
    hue, saturation, value = cv2.split(hsv)
    # Broad enough for warm/cool illumination while rejecting foliage and white fabric.
    skin = (
        (cr >= 132)
        & (cr <= 181)
        & (cb >= 76)
        & (cb <= 135)
        & (hue <= 28)
        & (saturation >= 24)
        & (value >= 38)
        & (y >= 35)
    ).astype(np.uint8) * 255
    size = max(3, int(round(min(image.shape[:2]) / 350)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel, iterations=2)
    return _largest_components(skin)


def _face_mask(rgb: FloatImage, skin: NDArray[np.uint8]) -> tuple[FloatImage, float]:
    gray = cv2.cvtColor(np.clip(rgb * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(24, 24))
    result = np.zeros(gray.shape, np.float32)
    for x, y, width, height in faces:
        skin_fraction = float(np.mean(skin[y : y + height, x : x + width] > 0))
        if skin_fraction < 0.08:
            continue
        center = (x + width // 2, y + int(height * 0.53))
        cv2.ellipse(result, center, (int(width * 0.45), int(height * 0.56)), 0, 0, 360, 1, -1)
    if np.any(result):
        face = _feather(result * (skin > 0).astype(np.float32), max(5, gray.shape[0] // 240))
        if np.mean(face > 0.2) > 0.0001:
            return face, 0.92

    # Profile faces often evade Haar. Use the uppermost substantial skin component,
    # constrained to a face-like ellipse so arms and hands are not softened as faces.
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(skin, 8)
    candidates = [
        i
        for i in range(1, count)
        if stats[i, cv2.CC_STAT_AREA] > skin.size // 15000
        and centroids[i][1] < skin.shape[0] * 0.68
    ]
    if not candidates:
        return result, 0.0
    label = min(candidates, key=lambda i: (centroids[i][1], -stats[i, cv2.CC_STAT_AREA]))
    x, y, width, height = stats[label, :4]
    pad_x, pad_y = int(width * 0.45), int(height * 0.35)
    center = (x + width // 2, y + height // 2)
    cv2.ellipse(
        result,
        center,
        (max(8, width // 2 + pad_x), max(10, height // 2 + pad_y)),
        0,
        0,
        360,
        1,
        -1,
    )
    return _feather(result * (skin > 0).astype(np.float32), max(5, gray.shape[0] // 220)), 0.62


def _subject_mask(rgb: FloatImage, skin: NDArray[np.uint8]) -> tuple[FloatImage, float]:
    height, width = skin.shape
    points = cv2.findNonZero(skin)
    if points is None or len(points) < max(20, skin.size // 50000):
        return np.zeros((height, width), np.float32), 0.0

    x, y, box_width, box_height = cv2.boundingRect(points)
    # Seed probable foreground around detected skin and the body beneath it. GrabCut
    # then follows real image boundaries instead of using a fixed portrait vignette.
    gc_mask = np.full((height, width), cv2.GC_PR_BGD, np.uint8)
    border = max(2, min(height, width) // 80)
    gc_mask[:border] = cv2.GC_BGD
    gc_mask[-border:] = cv2.GC_BGD
    gc_mask[:, :border] = cv2.GC_BGD
    gc_mask[:, -border:] = cv2.GC_BGD
    gc_mask[skin > 0] = cv2.GC_FGD

    center_x = x + box_width // 2
    top = max(0, y - box_height)
    bottom = min(height, y + max(box_height * 5, height // 3))
    body_width = max(box_width * 3, width // 10)
    left = max(0, center_x - body_width)
    right = min(width, center_x + body_width)
    gc_mask[top:bottom, left:right] = np.where(
        gc_mask[top:bottom, left:right] == cv2.GC_BGD,
        cv2.GC_BGD,
        cv2.GC_PR_FGD,
    )
    gc_mask[skin > 0] = cv2.GC_FGD

    sample = cv2.resize(np.clip(rgb * 255, 0, 255).astype(np.uint8), None, fx=0.5, fy=0.5)
    small_mask = cv2.resize(
        gc_mask, (sample.shape[1], sample.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(sample, small_mask, None, bg_model, fg_model, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return _feather((gc_mask == cv2.GC_FGD).astype(np.float32), 15), 0.35
    foreground = np.isin(small_mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8) * 255
    foreground = _largest_components(foreground, count=2)
    foreground = cv2.resize(foreground, (width, height), interpolation=cv2.INTER_LINEAR)
    result = _feather(foreground.astype(np.float32) / 255, max(9, height // 120))
    coverage = float(np.mean(result > 0.5))
    confidence = 0.78 if 0.025 <= coverage <= 0.70 else 0.48
    return result, confidence


@lru_cache(maxsize=2)
def _rembg_session(model_name: str):  # type: ignore[no-untyped-def]
    from rembg import new_session

    return new_session(model_name)


def _model_subject_mask(rgb: FloatImage, *, portrait: bool) -> tuple[FloatImage, float] | None:
    """Run a local ONNX saliency/person model; never sends pixels off-device."""
    from rembg import remove

    model_name = "u2net_human_seg" if portrait else "u2net"
    image = Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8), "RGB")
    try:
        mask_image = remove(image, session=_rembg_session(model_name), only_mask=True)
    except Exception:  # rembg wraps download and ONNX failures in several exception types
        return None
    mask = np.asarray(mask_image.convert("L"), dtype=np.float32) / 255
    binary = (mask >= 0.32).astype(np.uint8) * 255
    binary = _largest_components(binary, count=2)
    mask *= binary.astype(np.float32) / 255
    mask = _feather(mask, max(5, rgb.shape[0] // 320))
    coverage = float(np.mean(mask > 0.35))
    if not 0.005 <= coverage <= 0.85:
        return None
    return mask, 0.94


def build_masks(rgb: FloatImage, *, portrait: bool = True) -> MaskSet:
    image_u8 = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(image_u8, cv2.COLOR_RGB2HSV)
    hue, saturation, value = [channel.astype(np.float32) for channel in cv2.split(hsv)]
    skin_binary = _skin_mask(rgb)
    model_subject = _model_subject_mask(rgb, portrait=portrait)
    if model_subject is None:
        subject, subject_confidence = _subject_mask(rgb, skin_binary)
        # Classical fallback is useful offline, but not trustworthy enough for an
        # automatic local adjustment. The engine will export it for inspection and
        # fail closed rather than apply it as if it were a neural mask.
        subject_confidence = min(subject_confidence, 0.39)
    else:
        subject, subject_confidence = model_subject
    skin_binary = np.where(subject > 0.12, skin_binary, 0).astype(np.uint8)
    skin = _feather(skin_binary.astype(np.float32) / 255, max(5, rgb.shape[0] // 260))
    face, face_confidence = _face_mask(rgb, skin_binary)

    foliage_binary = ((hue >= 18) & (hue <= 92) & (saturation >= 35) & (value >= 18)).astype(
        np.float32
    )
    foliage = _feather(foliage_binary * (1 - subject), max(5, rgb.shape[0] // 300))
    background = np.clip(1 - subject, 0, 1).astype(np.float32)
    # A hue threshold is unstable for pale sky: hue becomes effectively random near
    # grey and produces islands and halos between branches.  Use a continuous
    # bright/low-chroma likelihood instead.  Subject exclusion protects faces and
    # white clothing, while the soft tonal selection also behaves sensibly for
    # overcast and golden skies and bright background bokeh.
    brightness = _smoothstep(value, 42, 190)
    low_chroma = 1 - _smoothstep(saturation, 58, 185)
    sky_likelihood = brightness * (0.30 + 0.70 * low_chroma)
    sky = _feather(
        sky_likelihood.astype(np.float32) * background,
        max(3, rgb.shape[0] // 520),
    )
    return MaskSet(
        subject=subject,
        background=background,
        skin=skin,
        face=face,
        foliage=foliage,
        sky=sky,
        confidence={
            "subject": subject_confidence,
            "background": subject_confidence,
            "skin": 0.82 if np.mean(skin > 0.3) > 0.001 else 0.0,
            "face": face_confidence,
            "foliage": 0.86 if np.mean(foliage > 0.3) > 0.01 else 0.0,
            "sky": 0.82 if np.mean(sky > 0.3) > 0.01 else 0.0,
        },
    )
