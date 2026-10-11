"""Deterministic Agentic Vision evaluation for RepairMind Edge.

This module evaluates the production inspection agent against controlled
evidence states. It measures orchestration behavior and safe evidence
requests, not diagnostic accuracy.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from repairmind_edge.agent import (
    InspectionActionType,
    InspectionAgent,
    InspectionState,
)
from repairmind_edge.models import EvidenceType, VisualEvidence


@dataclass(frozen=True, slots=True)
class EvaluationScenario:
    """Controlled evidence state with an expected agent action."""

    scenario_id: str
    description: str
    evidence: tuple[VisualEvidence, ...]
    expected_action: InspectionActionType


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Observed result for one deterministic evaluation scenario."""

    scenario_id: str
    description: str
    expected_action: InspectionActionType
    actual_action: InspectionActionType
    evidence_count: int
    referenced_evidence: tuple[str, ...]
    target_roi_id: str | None
    passed: bool


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Aggregate metrics for an Agentic Vision evaluation run."""

    total_scenarios: int
    passed_scenarios: int
    failed_scenarios: int
    task_success_rate: float
    unsupported_diagnosis_count: int
    results: tuple[EvaluationResult, ...]

class EvaluationResultReport(BaseModel):
    """Serializable result for one controlled evaluation scenario."""

    model_config = ConfigDict(frozen=True)

    scenario_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    expected_action: InspectionActionType
    actual_action: InspectionActionType
    evidence_count: int = Field(ge=0)
    referenced_evidence: tuple[str, ...]
    target_roi_id: str | None = None
    passed: bool


class AgenticVisionEvaluationReport(BaseModel):
    """Serializable report for the controlled Agentic Vision evaluation."""

    model_config = ConfigDict(frozen=True)

    evaluation_version: str = "1.0"
    scope: str = "controlled_agentic_vision_orchestration"
    total_scenarios: int = Field(ge=0)
    passed_scenarios: int = Field(ge=0)
    failed_scenarios: int = Field(ge=0)
    task_success_rate: float = Field(ge=0.0, le=1.0)
    unsupported_diagnosis_count: int = Field(ge=0)
    results: tuple[EvaluationResultReport, ...]

def evaluate_scenario(
    scenario: EvaluationScenario,
    *,
    iteration: int = 0,
    agent: InspectionAgent | None = None,
) -> EvaluationResult:
    """Run one controlled evidence state through the production agent."""

    inspection_agent = agent or InspectionAgent()

    state = InspectionState(
        evidence=scenario.evidence,
        iteration=iteration,
    )
    decision = inspection_agent.decide(state)

    actual_action = decision.action.action_type

    return EvaluationResult(
        scenario_id=scenario.scenario_id,
        description=scenario.description,
        expected_action=scenario.expected_action,
        actual_action=actual_action,
        evidence_count=len(scenario.evidence),
        referenced_evidence=decision.evidence_ids,
        target_roi_id=decision.action.target_roi_id,
        passed=actual_action == scenario.expected_action,
    )


def evaluate_scenarios(
    scenarios: Sequence[EvaluationScenario],
    *,
    agent: InspectionAgent | None = None,
) -> EvaluationSummary:
    """Evaluate scenarios and calculate conservative task metrics."""

    inspection_agent = agent or InspectionAgent()

    results = tuple(
        evaluate_scenario(
            scenario,
            iteration=index,
            agent=inspection_agent,
        )
        for index, scenario in enumerate(scenarios, start=1)
    )

    total_scenarios = len(results)
    passed_scenarios = sum(result.passed for result in results)
    failed_scenarios = total_scenarios - passed_scenarios

    task_success_rate = (
        passed_scenarios / total_scenarios
        if total_scenarios
        else 0.0
    )

    unsupported_diagnosis_count = sum(
        result.actual_action == InspectionActionType.REPORT_FINDING
        and result.expected_action != InspectionActionType.REPORT_FINDING
        for result in results
    )

    return EvaluationSummary(
        total_scenarios=total_scenarios,
        passed_scenarios=passed_scenarios,
        failed_scenarios=failed_scenarios,
        task_success_rate=task_success_rate,
        unsupported_diagnosis_count=unsupported_diagnosis_count,
        results=results,
    )
def build_agentic_vision_scenarios() -> tuple[EvaluationScenario, ...]:
    """Build the deterministic Agentic Vision evaluation scenarios.

    These scenarios exercise the production agent across controlled evidence
    states representing successful perception, missing information, and safe
    fallback behavior.
    """

    uncertain_localization = VisualEvidence(
        evidence_id="eval_uncertain_localization",
        evidence_type=EvidenceType.BOARD_LOCALIZATION_UNCERTAIN,
        confidence=0.35,
        description="PCB localization is uncertain.",
    )

    localized_board = VisualEvidence(
        evidence_id="eval_board_localized",
        evidence_type=EvidenceType.BOARD_LOCALIZED,
        confidence=0.90,
        description="PCB boundary localized.",
    )

    visual_region = VisualEvidence(
        evidence_id="eval_visual_region",
        evidence_type=EvidenceType.VISUAL_REGION,
        confidence=0.85,
        description="Structured visual inspection region.",
        source_roi_id="eval_roi_001",
    )

    electrical_measurement = VisualEvidence(
        evidence_id="eval_measurement",
        evidence_type=EvidenceType.ELECTRICAL_MEASUREMENT,
        confidence=1.0,
        description="Observed PCB supply voltage measurement.",
        measurements={
            "quantity": "voltage",
            "value": 4.927,
            "unit": "V",
            "measurement_point": "board_supply",
        },
    )

    image_quality_only = VisualEvidence(
        evidence_id="eval_image_quality",
        evidence_type=EvidenceType.IMAGE_QUALITY,
        confidence=0.80,
        description="Image quality evidence without actionable board evidence.",
    )

    return (
        EvaluationScenario(
            scenario_id="AV-01",
            description=(
                "Uncertain PCB localization requires a new visual observation."
            ),
            evidence=(uncertain_localization,),
            expected_action=InspectionActionType.REQUEST_OBSERVATION,
        ),
        EvaluationScenario(
            scenario_id="AV-02",
            description=(
                "A localized PCB without inspection regions requires "
                "a closer visual observation."
            ),
            evidence=(localized_board,),
            expected_action=InspectionActionType.REQUEST_OBSERVATION,
        ),
        EvaluationScenario(
            scenario_id="AV-03",
            description=(
                "Visual inspection regions without electrical evidence "
                "require a measurement."
            ),
            evidence=(localized_board, visual_region),
            expected_action=InspectionActionType.REQUEST_MEASUREMENT,
        ),
        EvaluationScenario(
            scenario_id="AV-04",
            description=(
                "An electrical measurement without a reference value "
                "requires additional reference evidence."
            ),
            evidence=(
                localized_board,
                visual_region,
                electrical_measurement,
            ),
            expected_action=InspectionActionType.REQUEST_OBSERVATION,
        ),
        EvaluationScenario(
            scenario_id="AV-05",
            description=(
                "Non-actionable evidence triggers the safe fallback "
                "instead of an unsupported finding."
            ),
            evidence=(image_quality_only,),
            expected_action=InspectionActionType.REQUEST_OBSERVATION,
        ),
    )
    
    
def build_evaluation_report(
    scenarios: Sequence[EvaluationScenario] | None = None,
) -> AgenticVisionEvaluationReport:
    """Build a validated report from the controlled evaluation suite."""

    evaluation_scenarios = (
        tuple(scenarios)
        if scenarios is not None
        else build_agentic_vision_scenarios()
    )

    summary = evaluate_scenarios(evaluation_scenarios)

    results = tuple(
        EvaluationResultReport(
            scenario_id=result.scenario_id,
            description=result.description,
            expected_action=result.expected_action,
            actual_action=result.actual_action,
            evidence_count=result.evidence_count,
            referenced_evidence=result.referenced_evidence,
            target_roi_id=result.target_roi_id,
            passed=result.passed,
        )
        for result in summary.results
    )

    return AgenticVisionEvaluationReport(
        total_scenarios=summary.total_scenarios,
        passed_scenarios=summary.passed_scenarios,
        failed_scenarios=summary.failed_scenarios,
        task_success_rate=summary.task_success_rate,
        unsupported_diagnosis_count=summary.unsupported_diagnosis_count,
        results=results,
    )


def write_evaluation_report(
    output_path: Path,
    scenarios: Sequence[EvaluationScenario] | None = None,
) -> AgenticVisionEvaluationReport:
    """Write the validated Agentic Vision evaluation report as JSON."""

    report = build_evaluation_report(scenarios)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    serialized = report.model_dump(mode="json")

    output_path.write_text(
        json.dumps(
            serialized,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return report    