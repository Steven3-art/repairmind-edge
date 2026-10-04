"""Deterministic inspection reporting for RepairMind Edge."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from repairmind_edge.agent import (
    InspectionActionType,
    InspectionDecision,
)
from repairmind_edge.models import (
    ElectricalMeasurement,
    EvidenceType,
    InspectionResult,
)


class InspectionReportStatus(StrEnum):
    """High-level status derived from the latest agent decision."""

    MEASUREMENT_REQUIRED = "MEASUREMENT_REQUIRED"
    ADDITIONAL_EVIDENCE_REQUIRED = "ADDITIONAL_EVIDENCE_REQUIRED"
    FINDING_AVAILABLE = "FINDING_AVAILABLE"


class InspectionSummary(BaseModel):
    """Summary of the visual inspection stage."""

    model_config = ConfigDict(frozen=True)

    board_detected: bool
    localization_confidence: float = Field(ge=0.0, le=1.0)
    visual_evidence_count: int = Field(ge=0)
    visual_region_count: int = Field(ge=0)


class DecisionSummary(BaseModel):
    """Report-safe projection of an agent decision."""

    model_config = ConfigDict(frozen=True)

    iteration: int = Field(ge=0)
    action: InspectionActionType
    reason: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    target_roi_id: str | None = None
    evidence_ids: tuple[str, ...]


class MeasurementSummary(BaseModel):
    """Electrical measurement preserved without interpretation."""

    model_config = ConfigDict(frozen=True)

    quantity: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)
    measurement_point: str = Field(min_length=1)


class InspectionReport(BaseModel):
    """Consolidated, auditable inspection report."""

    model_config = ConfigDict(frozen=True)

    report_version: str = "1.0"
    inspection: InspectionSummary
    iteration_1: DecisionSummary
    measurement: MeasurementSummary | None = None
    iteration_2: DecisionSummary | None = None
    status: InspectionReportStatus


def _summarize_decision(
    decision: InspectionDecision,
) -> DecisionSummary:
    """Project an agent decision into its immutable report representation."""

    return DecisionSummary(
        iteration=decision.iteration,
        action=decision.action.action_type,
        reason=decision.action.reason,
        instruction=decision.action.instruction,
        target_roi_id=decision.action.target_roi_id,
        evidence_ids=decision.evidence_ids,
    )


def _derive_status(
    latest_decision: InspectionDecision,
) -> InspectionReportStatus:
    """Derive report status without introducing new technical conclusions."""

    if latest_decision.action.action_type == InspectionActionType.REQUEST_MEASUREMENT:
        return InspectionReportStatus.MEASUREMENT_REQUIRED

    if latest_decision.action.action_type == InspectionActionType.REPORT_FINDING:
        return InspectionReportStatus.FINDING_AVAILABLE

    return InspectionReportStatus.ADDITIONAL_EVIDENCE_REQUIRED


def build_inspection_report(
    *,
    inspection_result: InspectionResult,
    iteration_1: InspectionDecision,
    measurement: ElectricalMeasurement | None = None,
    iteration_2: InspectionDecision | None = None,
) -> InspectionReport:
    """Build a deterministic report from validated inspection artifacts."""

    if (measurement is None) != (iteration_2 is None):
        raise ValueError(
            "measurement and iteration_2 must either both be provided or both be omitted"
        )

    visual_region_count = sum(
        item.evidence_type == EvidenceType.VISUAL_REGION for item in inspection_result.evidence
    )

    latest_decision = iteration_2 if iteration_2 is not None else iteration_1

    measurement_summary = (
        MeasurementSummary(
            quantity=measurement.quantity,
            value=measurement.value,
            unit=measurement.unit,
            measurement_point=measurement.measurement_point,
        )
        if measurement is not None
        else None
    )

    return InspectionReport(
        inspection=InspectionSummary(
            board_detected=inspection_result.localization.detected,
            localization_confidence=(inspection_result.localization.confidence),
            visual_evidence_count=len(inspection_result.evidence),
            visual_region_count=visual_region_count,
        ),
        iteration_1=_summarize_decision(iteration_1),
        measurement=measurement_summary,
        iteration_2=(_summarize_decision(iteration_2) if iteration_2 is not None else None),
        status=_derive_status(latest_decision),
    )
