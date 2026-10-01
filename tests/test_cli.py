"""Tests for the RepairMind Edge command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

from _pytest.capture import CaptureFixture

from repairmind_edge.cli import main
from repairmind_edge.models import EvidenceType, VisualEvidence


def _write_inspection_result(path: Path) -> None:
    """Write a minimal valid inspection result for CLI reasoning tests."""

    evidence = (
        VisualEvidence(
            evidence_id="VE-0002",
            evidence_type=EvidenceType.BOARD_LOCALIZED,
            confidence=0.86,
            description="PCB boundary localized.",
        ),
        VisualEvidence(
            evidence_id="evidence_roi_001",
            evidence_type=EvidenceType.VISUAL_REGION,
            confidence=0.91,
            description="Visually active inspection region detected.",
            source_roi_id="roi_001",
            measurements={
                "proposal_score": 0.91,
            },
        ),
    )

    document = {"evidence": [item.model_dump(mode="json") for item in evidence]}

    path.write_text(
        json.dumps(document, indent=2) + "\n",
        encoding="utf-8",
    )


def test_reason_command_writes_deterministic_decision(
    tmp_path: Path,
) -> None:
    """Reason command should persist the next evidence-driven action."""

    result_path = tmp_path / "result.json"
    decision_path = tmp_path / "decision.json"

    _write_inspection_result(result_path)

    exit_code = main(
        [
            "reason",
            str(result_path),
            "--iteration",
            "3",
        ]
    )

    assert exit_code == 0
    assert decision_path.is_file()

    document = json.loads(decision_path.read_text(encoding="utf-8"))

    assert document["iteration"] == 3
    assert document["action"]["action_type"] == "REQUEST_MEASUREMENT"
    assert document["action"]["target_roi_id"] == "roi_001"
    assert document["evidence_ids"] == ["evidence_roi_001"]


def test_reason_command_rejects_invalid_json(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    """Reason command should reject malformed inspection documents."""

    result_path = tmp_path / "result.json"
    result_path.write_text(
        "{not-valid-json",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "reason",
            str(result_path),
        ]
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]

    assert exit_code == 3
    assert "could not load inspection result" in captured.err
    assert not (tmp_path / "decision.json").exists()


def test_reason_command_rejects_negative_iteration(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    """Reason command should reject negative inspection iterations."""

    result_path = tmp_path / "result.json"
    _write_inspection_result(result_path)

    exit_code = main(
        [
            "reason",
            str(result_path),
            "--iteration",
            "-1",
        ]
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]

    assert exit_code == 2
    assert "iteration must be greater than or equal to zero" in captured.err
    assert not (tmp_path / "decision.json").exists()


def test_measure_command_persists_measurement_and_second_decision(
    tmp_path: Path,
) -> None:
    """Measure command should close the second inspection iteration."""

    result_path = tmp_path / "result.json"
    _write_inspection_result(result_path)

    exit_code = main(
        [
            "measure",
            str(result_path),
            "--quantity",
            "voltage",
            "--value",
            "4.927",
            "--unit",
            "V",
            "--point",
            "board_supply",
        ]
    )

    assert exit_code == 0

    measurement_path = tmp_path / "measurement.json"
    decision_path = tmp_path / "decision_iteration_2.json"

    assert measurement_path.is_file()
    assert decision_path.is_file()

    measurement_document = json.loads(measurement_path.read_text(encoding="utf-8"))

    assert measurement_document["evidence_type"] == "ELECTRICAL_MEASUREMENT"
    assert measurement_document["evidence_id"] == "measurement_001"
    assert measurement_document["measurements"] == {
        "quantity": "voltage",
        "value": 4.927,
        "unit": "V",
        "measurement_point": "board_supply",
    }

    decision_document = json.loads(decision_path.read_text(encoding="utf-8"))

    assert decision_document["iteration"] == 2
    assert decision_document["action"]["action_type"] == "REQUEST_OBSERVATION"
    assert decision_document["evidence_ids"] == ["measurement_001"]


def test_measure_command_rejects_negative_iteration(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    """Measure command should reject invalid iteration numbers."""

    result_path = tmp_path / "result.json"
    _write_inspection_result(result_path)

    exit_code = main(
        [
            "measure",
            str(result_path),
            "--quantity",
            "voltage",
            "--value",
            "4.927",
            "--unit",
            "V",
            "--point",
            "board_supply",
            "--iteration",
            "-1",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "iteration must be greater than or equal to zero" in captured.err
    assert not (tmp_path / "measurement.json").exists()
