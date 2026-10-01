"""Agentic inspection contracts for RepairMind Edge."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from repairmind_edge.models import (
    ElectricalMeasurement,
    EvidenceType,
    VisualEvidence,
)


class InspectionActionType(StrEnum):
    """Actions that the inspection agent may request."""

    REQUEST_OBSERVATION = "REQUEST_OBSERVATION"
    REQUEST_MEASUREMENT = "REQUEST_MEASUREMENT"
    REPORT_FINDING = "REPORT_FINDING"


class InspectionAction(BaseModel):
    """Next action selected by the inspection agent."""

    model_config = ConfigDict(frozen=True)

    action_type: InspectionActionType
    reason: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    target_roi_id: str | None = None


class InspectionState(BaseModel):
    """Immutable evidence state evaluated during an inspection iteration."""

    model_config = ConfigDict(frozen=True)

    evidence: tuple[VisualEvidence, ...]
    iteration: int = Field(ge=0)


class InspectionDecision(BaseModel):
    """Auditable decision produced from an inspection state."""

    model_config = ConfigDict(frozen=True)

    iteration: int = Field(ge=0)
    action: InspectionAction
    evidence_ids: tuple[str, ...]


def build_electrical_measurement_evidence(
    measurement: ElectricalMeasurement,
    *,
    evidence_id: str,
) -> VisualEvidence:
    """Convert an observed electrical measurement into structured evidence."""

    return VisualEvidence(
        evidence_id=evidence_id,
        evidence_type=EvidenceType.ELECTRICAL_MEASUREMENT,
        confidence=1.0,
        description=("Electrical measurement collected during the inspection."),
        measurements=measurement.model_dump(mode="json"),
    )


class InspectionAgent:
    """Deterministic decision engine for evidence-driven PCB inspection."""

    def decide(self, state: InspectionState) -> InspectionDecision:
        """Select the next inspection action from available evidence."""

        uncertain_localization = tuple(
            item
            for item in state.evidence
            if item.evidence_type == EvidenceType.BOARD_LOCALIZATION_UNCERTAIN
        )

        if uncertain_localization:
            return InspectionDecision(
                iteration=state.iteration,
                action=InspectionAction(
                    action_type=InspectionActionType.REQUEST_OBSERVATION,
                    reason=(
                        "A reliable PCB boundary has not been established "
                        "from the available visual evidence."
                    ),
                    instruction=(
                        "Capture a sharp image showing the complete PCB "
                        "with all board edges visible."
                    ),
                ),
                evidence_ids=tuple(item.evidence_id for item in uncertain_localization),
            )

        localized_evidence = tuple(
            item for item in state.evidence if item.evidence_type == EvidenceType.BOARD_LOCALIZED
        )

        region_evidence = tuple(
            item for item in state.evidence if item.evidence_type == EvidenceType.VISUAL_REGION
        )

        if localized_evidence and not region_evidence:
            return InspectionDecision(
                iteration=state.iteration,
                action=InspectionAction(
                    action_type=InspectionActionType.REQUEST_OBSERVATION,
                    reason=("The PCB is localized, but no visual inspection region is available."),
                    instruction=(
                        "Capture a closer, well-focused image of the PCB "
                        "to expose component-level visual structure."
                    ),
                ),
                evidence_ids=tuple(item.evidence_id for item in localized_evidence),
            )

        electrical_evidence = tuple(
            item
            for item in state.evidence
            if item.evidence_type == EvidenceType.ELECTRICAL_MEASUREMENT
        )

        if region_evidence and not electrical_evidence:
            target_region = max(
                region_evidence,
                key=lambda item: (
                    item.confidence,
                    item.source_roi_id or "",
                    item.evidence_id,
                ),
            )

            return InspectionDecision(
                iteration=state.iteration,
                action=InspectionAction(
                    action_type=InspectionActionType.REQUEST_MEASUREMENT,
                    reason=(
                        "Visual inspection regions are available, but "
                        "visual evidence alone cannot establish "
                        "electrical behavior."
                    ),
                    instruction=(
                        "Measure the PCB supply voltage and provide the measured value in volts."
                    ),
                    target_roi_id=target_region.source_roi_id,
                ),
                evidence_ids=tuple(item.evidence_id for item in region_evidence),
            )

        if electrical_evidence:
            return InspectionDecision(
                iteration=state.iteration,
                action=InspectionAction(
                    action_type=InspectionActionType.REQUEST_OBSERVATION,
                    reason=(
                        "An electrical measurement is available, but no "
                        "expected reference value or tolerance is available "
                        "to interpret it safely."
                    ),
                    instruction=(
                        "Provide the expected supply voltage or reference "
                        "specification for the measured point."
                    ),
                ),
                evidence_ids=tuple(item.evidence_id for item in electrical_evidence),
            )

        return InspectionDecision(
            iteration=state.iteration,
            action=InspectionAction(
                action_type=InspectionActionType.REQUEST_OBSERVATION,
                reason=("The available evidence does not yet support a deterministic finding."),
                instruction=("Provide an additional inspection observation or measurement."),
            ),
            evidence_ids=tuple(item.evidence_id for item in state.evidence),
        )
