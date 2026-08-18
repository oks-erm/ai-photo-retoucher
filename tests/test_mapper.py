from backend.mapper import map_edit_plan
from backend.schemas import EditPlan, Mode


def test_local_edits_are_skipped_without_confident_mask(plan: EditPlan) -> None:
    result = map_edit_plan(plan, mask_confidence=0.5, mask_threshold=0.72)
    assert not any(operation.mask for operation in result.operations)
    assert len(result.skipped) >= 2


def test_local_edits_use_subject_and_inverse_masks(plan: EditPlan) -> None:
    result = map_edit_plan(plan, mask_confidence=0.9)
    masks = {operation.mask for operation in result.operations if operation.mask}
    assert masks == {"subject", "background:inverse-subject"}


def test_strength_scales_numeric_adjustments(plan: EditPlan) -> None:
    result = map_edit_plan(plan, strength=0.5)
    exposure = next(
        operation for operation in result.operations if operation.instance == "AI global"
    )
    assert exposure.parameters["exposure"] == 0.2


def test_softening_is_never_executed_without_skin_mask(plan: EditPlan) -> None:
    result = map_edit_plan(plan, mode=Mode.PORTRAIT)
    assert any("post-MVP" in reason for reason in result.skipped)
