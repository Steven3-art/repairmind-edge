"""Tests for deterministic RepairMind Edge inspection reporting."""

from repairmind_edge.agent import (
    InspectionAction,
    InspectionActionType,
    InspectionDecision,
)
from repairmind_edge.models import (
    BoardLocalization,
    ElectricalMeasurement,
    EvidenceType,
    ImageQuality,
    InspectionResult,
    VisualEvidence,
)
from repairmind_edge.reporting import (
    InspectionReportStatus,
    build_inspection_report,
)


def _build_inspection_result() -> InspectionResult:
    """Build deterministic visual evidence for reporting tests."""

    evidence = (
        VisualEvidence(
            evidence_id="VE-0001",
            evidence_type=EvidenceType.IMAGE_QUALITY,
            confidence=1.0,
            description="Input image quality measured.",
        ),
        VisualEvidence(
            evidence_id="VE-0002",
            evidence_type=EvidenceType.BOARD_LOCALIZED,
            confidence=0.863,
            description="PCB boundary localized.",
        ),
        VisualEvidence(
            evidence_id="evidence_roi_001",
            evidence_type=EvidenceType.VISUAL_REGION,
            confidence=1.0,
            description="Visual inspection region proposed.",
            source_roi_id="roi_001",
        ),
        VisualEvidence(
            evidence_id="evidence_roi_002",
            evidence_type=EvidenceType.VISUAL_REGION,
            confidence=1.0,
            description="Visual inspection region proposed.",
            source_roi_id="roi_002",
        ),
    )

    return InspectionResult(
        quality=ImageQuality(
            width=800,
            height=600,
            channels=3,
            brightness=128.0,
            sharpness=100.0,
        ),
        localization=BoardLocalization(
            detected=True,
            confidence=0.863,
            area_ratio=0.747,
        ),
        evidence=evidence,
        normalized_width=631,
        normalized_height=778,
    )


def _build_iteration_one() -> InspectionDecision:
    """Build the first deterministic agent decision."""

    return InspectionDecision(
        iteration=1,
        action=InspectionAction(
            action_type=InspectionActionType.REQUEST_MEASUREMENT,
            reason=("Visual evidence alone cannot establish electrical behavior."),
            instruction="Measure the PCB supply voltage.",
            target_roi_id="roi_002",
        ),
        evidence_ids=(
            "evidence_roi_001",
            "evidence_roi_002",
        ),
    )


def _build_iteration_two() -> InspectionDecision:
    """Build the second deterministic agent decision."""

    return InspectionDecision(
        iteration=2,
        action=InspectionAction(
            action_type=InspectionActionType.REQUEST_OBSERVATION,
            reason=("No expected reference value or tolerance is available."),
            instruction=("Provide the expected supply voltage or reference specification."),
        ),
        evidence_ids=("measurement_001",),
    )


def test_report_summarizes_visual_inspection() -> None:
    """Report should preserve objective perception information."""

    report = build_inspection_report(
        inspection_result=_build_inspection_result(),
        iteration_1=_build_iteration_one(),
    )

    assert report.report_version == "1.0"
    assert report.inspection.board_detected is True
    assert report.inspection.localization_confidence == 0.863
    assert report.inspection.visual_evidence_count == 4
    assert report.inspection.visual_region_count == 2
    assert report.status == InspectionReportStatus.MEASUREMENT_REQUIRED


def test_report_preserves_measurement_without_interpretation() -> None:
    """Report should preserve the observed value without judging it."""

    measurement = ElectricalMeasurement(
        quantity="voltage",
        value=4.927,
        unit="V",
        measurement_point="board_supply",
    )

    report = build_inspection_report(
        inspection_result=_build_inspection_result(),
        iteration_1=_build_iteration_one(),
        measurement=measurement,
        iteration_2=_build_iteration_two(),
    )

    assert report.measurement is not None
    assert report.measurement.quantity == "voltage"
    assert report.measurement.value == 4.927
    assert report.measurement.unit == "V"
    assert report.measurement.measurement_point == "board_supply"

    assert report.iteration_2 is not None
    assert report.iteration_2.action == InspectionActionType.REQUEST_OBSERVATION
    assert report.iteration_2.evidence_ids == ("measurement_001",)
    assert report.status == InspectionReportStatus.ADDITIONAL_EVIDENCE_REQUIRED


def test_report_serializes_deterministically() -> None:
    """Equivalent inputs should produce identical report documents."""

    measurement = ElectricalMeasurement(
        quantity="voltage",
        value=4.927,
        unit="V",
        measurement_point="board_supply",
    )

    first = build_inspection_report(
        inspection_result=_build_inspection_result(),
        iteration_1=_build_iteration_one(),
        measurement=measurement,
        iteration_2=_build_iteration_two(),
    )

    second = build_inspection_report(
        inspection_result=_build_inspection_result(),
        iteration_1=_build_iteration_one(),
        measurement=measurement,
        iteration_2=_build_iteration_two(),
    )

    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_report_rejects_incomplete_second_iteration() -> None:
    """Measurement and its subsequent decision must stay paired."""

    measurement = ElectricalMeasurement(
        quantity="voltage",
        value=4.927,
        unit="V",
        measurement_point="board_supply",
    )

    try:
        build_inspection_report(
            inspection_result=_build_inspection_result(),
            iteration_1=_build_iteration_one(),
            measurement=measurement,
        )
    except ValueError as exc:
        assert (
            str(exc) == "measurement and iteration_2 must either both be "
            "provided or both be omitted"
        )
    else:
        raise AssertionError("Expected incomplete report inputs to be rejected.")
