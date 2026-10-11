"""Tests for the RepairMind Edge Agentic Vision evaluation suite."""

import pytest

from repairmind_edge.agent import InspectionActionType
from repairmind_edge.evaluation import (
    EvaluationScenario,
    build_agentic_vision_scenarios,
    evaluate_scenario,
    evaluate_scenarios,
)


@pytest.fixture
def scenarios() -> tuple[EvaluationScenario, ...]:
    """Return the controlled Agentic Vision evaluation scenarios."""

    return build_agentic_vision_scenarios()


def test_evaluation_contains_five_controlled_scenarios(
    scenarios: tuple[EvaluationScenario, ...],
) -> None:
    """The published evaluation must contain the five defined scenarios."""

    assert len(scenarios) == 5
    assert tuple(scenario.scenario_id for scenario in scenarios) == (
        "AV-01",
        "AV-02",
        "AV-03",
        "AV-04",
        "AV-05",
    )


def test_all_controlled_scenarios_produce_expected_actions(
    scenarios: tuple[EvaluationScenario, ...],
) -> None:
    """Every controlled scenario must produce its expected safe action."""

    summary = evaluate_scenarios(scenarios)

    assert summary.total_scenarios == 5
    assert summary.passed_scenarios == 5
    assert summary.failed_scenarios == 0
    assert summary.task_success_rate == 1.0


def test_evaluation_produces_no_unsupported_diagnosis(
    scenarios: tuple[EvaluationScenario, ...],
) -> None:
    """Controlled incomplete-evidence states must not report a finding."""

    summary = evaluate_scenarios(scenarios)

    assert summary.unsupported_diagnosis_count == 0

    assert all(
        result.actual_action != InspectionActionType.REPORT_FINDING
        for result in summary.results
    )


def test_visual_evidence_changes_next_action_to_measurement(
    scenarios: tuple[EvaluationScenario, ...],
) -> None:
    """AV-03 demonstrates the core perception-to-action transition."""

    scenario = next(
        item for item in scenarios if item.scenario_id == "AV-03"
    )

    result = evaluate_scenario(scenario, iteration=1)

    assert result.actual_action == InspectionActionType.REQUEST_MEASUREMENT
    assert result.target_roi_id == "eval_roi_001"
    assert "eval_visual_region" in result.referenced_evidence


def test_measurement_changes_next_action_to_reference_request(
    scenarios: tuple[EvaluationScenario, ...],
) -> None:
    """AV-04 demonstrates evidence-driven multi-step orchestration."""

    scenario = next(
        item for item in scenarios if item.scenario_id == "AV-04"
    )

    result = evaluate_scenario(scenario, iteration=2)

    assert result.actual_action == InspectionActionType.REQUEST_OBSERVATION
    assert result.target_roi_id is None
    assert result.referenced_evidence == ("eval_measurement",)


def test_uncertain_localization_is_traceable(
    scenarios: tuple[EvaluationScenario, ...],
) -> None:
    """Localization failure must produce a traceable observation request."""

    scenario = next(
        item for item in scenarios if item.scenario_id == "AV-01"
    )

    result = evaluate_scenario(scenario)

    assert result.actual_action == InspectionActionType.REQUEST_OBSERVATION
    assert result.referenced_evidence == (
        "eval_uncertain_localization",
    )


def test_non_actionable_evidence_uses_safe_fallback(
    scenarios: tuple[EvaluationScenario, ...],
) -> None:
    """Non-actionable evidence must not produce an unsupported finding."""

    scenario = next(
        item for item in scenarios if item.scenario_id == "AV-05"
    )

    result = evaluate_scenario(scenario)

    assert result.actual_action == InspectionActionType.REQUEST_OBSERVATION
    assert result.referenced_evidence == ("eval_image_quality",)
    


def test_evaluation_is_deterministic(
    scenarios: tuple[EvaluationScenario, ...],
) -> None:
    """Repeated evaluation of identical evidence must produce identical results."""

    first = evaluate_scenarios(scenarios)
    second = evaluate_scenarios(scenarios)

    assert first == second