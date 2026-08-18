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
