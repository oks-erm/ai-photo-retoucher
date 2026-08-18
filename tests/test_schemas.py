import pytest
from pydantic import ValidationError

from backend.schemas import DeltaPlan, EditPlan


def test_edit_plan_serializes_global_alias(plan: EditPlan) -> None:
    dumped = plan.model_dump(by_alias=True)
    assert "global" in dumped
    assert "global_" not in dumped


def test_schema_rejects_out_of_range_value(plan: EditPlan) -> None:
    payload = plan.model_dump(by_alias=True)
    payload["global"]["exposure_ev"] = 10
    with pytest.raises(ValidationError):
        EditPlan.model_validate(payload)


def test_protections_cannot_disable_identity(plan: EditPlan) -> None:
    payload = plan.model_dump(by_alias=True)
    payload["protections"]["preserve_identity"] = False
    with pytest.raises(ValidationError):
        EditPlan.model_validate(payload)


def test_delta_is_bounded() -> None:
    with pytest.raises(ValidationError):
        DeltaPlan.model_validate(
            {"accepted": False, "global_exposure_delta": 0.31, "summary": "Too dark"}
        )
