import cv2
import numpy as np

from backend.retouch.masks import _face_mask, _portrait_support, _safe_local_masks, _skin_mask


def test_skin_mask_rejects_warm_white_fabric() -> None:
    rgb = np.zeros((80, 120, 3), np.float32)
    rgb[:, :60] = (0.95, 0.89, 0.80)  # warm ivory fabric
    rgb[:, 60:] = (0.72, 0.42, 0.28)  # warm skin

    mask = _skin_mask(rgb)

    assert np.mean(mask[:, :60] > 0) < 0.05
    assert np.mean(mask[:, 60:] > 0) > 0.90


def test_face_fallback_refuses_body_sized_skin_component(monkeypatch) -> None:
    rgb = np.full((160, 240, 3), 0.4, np.float32)
    skin = np.zeros((160, 240), np.uint8)
    skin[18:145, 55:190] = 255

    class NoDetections:
        def detectMultiScale(self, *_args, **_kwargs):  # noqa: N802
            return ()

    monkeypatch.setattr(cv2, "CascadeClassifier", lambda *_args: NoDetections())
    face, confidence = _face_mask(rgb, skin)

    assert confidence == 0
    assert not np.any(face)


def test_portrait_support_excludes_distant_salient_object() -> None:
    skin = np.zeros((200, 300), np.uint8)
    skin[45:90, 190:220] = 255

    support = _portrait_support(skin)

    assert support is not None
    assert support[65, 205] > 0.95
    assert support[65, 20] < 0.01


def test_safe_local_masks_are_complementary_at_uncertain_edge() -> None:
    alpha = np.array([[0.0, 0.20, 0.45, 0.60, 0.80, 1.0]], np.float32)

    subject, background = _safe_local_masks(alpha)

    assert background[0, 0] == 1
    assert subject[0, -1] == 1
    assert np.allclose(subject + background, 1)
    assert 0 < subject[0, 2] < 1
    assert 0 < background[0, 2] < 1


def test_safe_local_masks_have_no_visible_neutral_strip() -> None:
    alpha = np.linspace(0, 1, 101, dtype=np.float32)[None, :]
    subject, background = _safe_local_masks(alpha)
    assert np.allclose(subject + background, 1)
    assert not np.any((subject == 0) & (background == 0))
    transition = (subject > 0) & (subject < 1)
    assert np.mean(transition) < 0.25
