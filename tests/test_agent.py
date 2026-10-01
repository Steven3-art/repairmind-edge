"""Tests for RepairMind Edge agentic inspection contracts."""

import pytest
from pydantic import ValidationError

from repairmind_edge.agent import (
    InspectionAction,
    InspectionActionType,
    InspectionAgent,
    InspectionDecision,
    InspectionState,
)
from repairmind_edge.models import EvidenceType, VisualEvidence


def _visual_region_evidence() -> VisualEvidence:
    """Create deterministic visual evidence for agent contract tests."""

    return VisualEvidence(
        evidence_id="evidence_roi_001",
        evidence_type=EvidenceType.VISUAL_REGION,
        confidence=0.82,
        description="Visually active inspection region detected on the normalized PCB.",
        source_roi_id="roi_001",
        measurements={
            "x": 10,
            "y": 20,
            "width": 40,
            "height": 30,
            "edge_density": 0.21,
            "local_contrast": 48.0,
            "proposal_score": 0.82,
        },
    )


def test_inspection_state_preserves_evidence_and_iteration() -> None:
    """Inspection state should preserve its evidence provenance."""

    evidence = _visual_region_evidence()

    state = InspectionState(
        evidence=(evidence,),
        iteration=0,
    )

    assert state.iteration == 0
    assert state.evidence == (evidence,)
    assert state.evidence[0].source_roi_id == "roi_001"


def test_inspection_action_can_target_visual_region() -> None:
    """Inspection actions may explicitly target a proposed ROI."""

    action = InspectionAction(
        action_type=InspectionActionType.REQUEST_OBSERVATION,
        reason="Additional visual detail is required.",
        instruction="Capture a closer image of the selected PCB region.",
        target_roi_id="roi_001",
    )

    assert action.action_type == InspectionActionType.REQUEST_OBSERVATION
    assert action.target_roi_id == "roi_001"


def test_inspection_decision_records_supporting_evidence() -> None:
    """A decision should identify the evidence used to produce it."""

    evidence = _visual_region_evidence()

    action = InspectionAction(
        action_type=InspectionActionType.REQUEST_MEASUREMENT,
        reason="Visual evidence alone cannot establish electrical behavior.",
        instruction="Measure the board supply voltage.",
        target_roi_id="roi_001",
    )

    decision = InspectionDecision(
        iteration=1,
        action=action,
        evidence_ids=(evidence.evidence_id,),
    )

    assert decision.iteration == 1
    assert decision.action == action
    assert decision.evidence_ids == ("evidence_roi_001",)


def test_inspection_contracts_reject_invalid_state() -> None:
    """Inspection contracts should reject structurally invalid values."""

    with pytest.raises(ValidationError):
        InspectionState(
            evidence=(),
            iteration=-1,
        )

    with pytest.raises(ValidationError):
        InspectionAction(
            action_type=InspectionActionType.REPORT_FINDING,
            reason="",
            instruction="Report the available evidence.",
        )


def _evidence(
    evidence_id: str,
    evidence_type: EvidenceType,
    *,
    confidence: float = 1.0,
    source_roi_id: str | None = None,
) -> VisualEvidence:
    """Create minimal deterministic evidence for agent decision tests."""

    return VisualEvidence(
        evidence_id=evidence_id,
        evidence_type=evidence_type,
        confidence=confidence,
        description="Synthetic evidence for deterministic agent testing.",
        source_roi_id=source_roi_id,
    )


def test_agent_requests_new_observation_when_localization_is_uncertain() -> None:
    """Uncertain localization should request a better board observation."""

    evidence = _evidence(
        "VE-0002",
        EvidenceType.BOARD_LOCALIZATION_UNCERTAIN,
        confidence=0.3,
    )

    decision = InspectionAgent().decide(
        InspectionState(
            evidence=(evidence,),
            iteration=1,
        )
    )

    assert decision.iteration == 1
    assert decision.action.action_type == InspectionActionType.REQUEST_OBSERVATION
    assert decision.action.target_roi_id is None
    assert decision.evidence_ids == ("VE-0002",)


def test_agent_requests_closer_view_when_board_has_no_regions() -> None:
    """Localized boards without ROI evidence should request more visual detail."""

    localized = _evidence(
        "VE-0002",
        EvidenceType.BOARD_LOCALIZED,
        confidence=0.86,
    )

    decision = InspectionAgent().decide(
        InspectionState(
            evidence=(localized,),
            iteration=2,
        )
    )

    assert decision.action.action_type == InspectionActionType.REQUEST_OBSERVATION
    assert decision.evidence_ids == ("VE-0002",)


def test_agent_requests_measurement_when_visual_regions_exist() -> None:
    """Visual regions without electrical evidence should trigger measurement."""

    localized = _evidence(
        "VE-0002",
        EvidenceType.BOARD_LOCALIZED,
        confidence=0.86,
    )
    region = _evidence(
        "evidence_roi_001",
        EvidenceType.VISUAL_REGION,
        confidence=0.91,
        source_roi_id="roi_001",
    )

    decision = InspectionAgent().decide(
        InspectionState(
            evidence=(localized, region),
            iteration=3,
        )
    )

    assert decision.action.action_type == InspectionActionType.REQUEST_MEASUREMENT
    assert decision.action.target_roi_id == "roi_001"
    assert decision.evidence_ids == ("evidence_roi_001",)


def test_agent_selects_region_deterministically() -> None:
    """Measurement targeting should be deterministic across visual regions."""

    region_1 = _evidence(
        "evidence_roi_001",
        EvidenceType.VISUAL_REGION,
        confidence=0.75,
        source_roi_id="roi_001",
    )
    region_2 = _evidence(
        "evidence_roi_002",
        EvidenceType.VISUAL_REGION,
        confidence=0.92,
        source_roi_id="roi_002",
    )

    state = InspectionState(
        evidence=(region_1, region_2),
        iteration=4,
    )

    agent = InspectionAgent()

    first = agent.decide(state)
    second = agent.decide(state)

    assert first == second
    assert first.action.action_type == InspectionActionType.REQUEST_MEASUREMENT
    assert first.action.target_roi_id == "roi_002"
    assert first.evidence_ids == (
        "evidence_roi_001",
        "evidence_roi_002",
    )


def test_agent_does_not_invent_finding_from_uninterpreted_measurement() -> None:
    """Electrical evidence alone should not be promoted to a finding."""

    measurement = _evidence(
        "measurement_supply_001",
        EvidenceType.ELECTRICAL_MEASUREMENT,
    )

    decision = InspectionAgent().decide(
        InspectionState(
            evidence=(measurement,),
            iteration=5,
        )
    )

    assert decision.action.action_type == InspectionActionType.REQUEST_OBSERVATION
    assert decision.action.action_type != InspectionActionType.REPORT_FINDING
    assert decision.evidence_ids == ("measurement_supply_001",)
