import pytest

from backend.schemas import EditPlan


@pytest.fixture
def plan() -> EditPlan:
    return EditPlan.model_validate(
        {
            "version": "1",
            "scene": {"category": "portrait", "lighting": "Warm side light"},
            "global": {
                "exposure_ev": 0.4,
                "contrast": 0.1,
                "black_depth": 0,
                "saturation": 0,
                "vibrance": 0.1,
            },
            "white_balance": {"temperature_delta_k": 250, "tint_delta": 0.01},
            "highlights": {"recovery": 0.3, "warmth": 0.05, "softness": 0.1},
            "shadows": {"lift": 0.1, "warmth": 0},
            "subject": {"enabled": True, "exposure_ev": 0.2, "contrast": 0, "saturation": 0},
            "background": {
                "enabled": True,
                "exposure_ev": -0.2,
                "contrast": 0.1,
                "saturation": -0.1,
            },
            "foliage": {"green_lightness": -0.1, "green_chroma": -0.15, "yellow_chroma": -0.05},
            "skin": {
                "protect": True,
                "exposure": 0.05,
                "warmth": 0,
                "chroma": 0,
                "texture_softening": 0.05,
            },
            "sharpening": {"amount": 0.12},
            "denoise": {"strength": 0.2},
            "protections": {
                "preserve_highlights": True,
                "preserve_deep_shadows": True,
                "preserve_identity": True,
                "preserve_geometry": True,
            },
            "summary": "Natural warm portrait correction",
        }
    )
