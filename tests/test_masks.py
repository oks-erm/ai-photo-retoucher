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


def test_safe_local_masks_leave_uncertain_edge_ungraded() -> None:
    alpha = np.array([[0.0, 0.20, 0.45, 0.60, 0.80, 1.0]], np.float32)

    subject, background = _safe_local_masks(alpha)

    assert background[0, 0] == 1
    assert subject[0, -1] == 1
    assert subject[0, 2] == 0
    assert background[0, 2] == 0
    assert not np.any((subject > 0) & (background > 0))


def test_safe_local_masks_do_not_create_opposing_grade_at_boundary() -> None:
    alpha = np.linspace(0, 1, 101, dtype=np.float32)[None, :]
    subject, background = _safe_local_masks(alpha)
    neutral_pixels = (alpha >= 0.32) & (alpha <= 0.56)

    # Simulate strong, opposing local exposure changes. The uncertain matte edge
    # must remain neutral rather than become a bright/dark seam.
    adjustment = subject * 0.40 - background * 0.40

    assert np.max(np.abs(adjustment[neutral_pixels])) == 0
