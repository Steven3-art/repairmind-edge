"""Tests for the RepairMind Edge command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from _pytest.capture import CaptureFixture
from pytest import MonkeyPatch

from repairmind_edge.cli import main
from repairmind_edge.models import EvidenceType, VisualEvidence


def _write_inspection_result(path: Path) -> None:
    """Write a valid deterministic inspection result for CLI tests."""

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

    document = {
        "quality": {
            "width": 800,
            "height": 600,
            "channels": 3,
            "brightness": 128.0,
            "sharpness": 100.0,
        },
        "localization": {
            "detected": True,
            "confidence": 0.86,
            "area_ratio": 0.747,
        },
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "normalized_width": 631,
        "normalized_height": 778,
    }

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


def test_demo_command_runs_complete_inspection_loop(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """Demo should orchestrate both agent iterations deterministically."""

    image_path = tmp_path / "board.jpg"
    output_directory = tmp_path / "demo"

    image_path.write_bytes(b"synthetic-test-input")

    def fake_inspect(
        image_path: Path,
        output_directory: Path,
    ) -> int:
        """Persist deterministic perception evidence for CLI orchestration."""

        assert image_path.is_file()

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        _write_inspection_result(output_directory / "result.json")

        return 0

    monkeypatch.setattr(
        "repairmind_edge.cli._inspect",
        fake_inspect,
    )

    exit_code = main(
        [
            "demo",
            str(image_path),
            "--output",
            str(output_directory),
            "--measurement-value",
            "4.927",
            "--measurement-unit",
            "V",
            "--measurement-point",
            "board_supply",
        ]
    )

    assert exit_code == 0

    result_path = output_directory / "result.json"
    first_decision_path = output_directory / "decision_iteration_1.json"
    measurement_path = output_directory / "measurement.json"
    second_decision_path = output_directory / "decision_iteration_2.json"

    assert result_path.is_file()
    assert first_decision_path.is_file()
    assert measurement_path.is_file()
    assert second_decision_path.is_file()

    report_path = output_directory / "inspection_report.json"

    assert report_path.is_file()

    report_document = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_document["report_version"] == "1.0"
    assert report_document["inspection"]["board_detected"] is True
    assert report_document["inspection"]["visual_region_count"] == 1

    assert report_document["iteration_1"]["action"] == "REQUEST_MEASUREMENT"

    assert report_document["measurement"] == {
        "quantity": "voltage",
        "value": 4.927,
        "unit": "V",
        "measurement_point": "board_supply",
    }

    assert report_document["iteration_2"]["action"] == "REQUEST_OBSERVATION"

    assert report_document["status"] == "ADDITIONAL_EVIDENCE_REQUIRED"

    first_decision = json.loads(first_decision_path.read_text(encoding="utf-8"))

    assert first_decision["iteration"] == 1
    assert first_decision["action"]["action_type"] == "REQUEST_MEASUREMENT"
    assert first_decision["action"]["target_roi_id"] == "roi_001"
    assert first_decision["evidence_ids"] == ["evidence_roi_001"]

    measurement_document = json.loads(measurement_path.read_text(encoding="utf-8"))

    assert measurement_document["evidence_type"] == "ELECTRICAL_MEASUREMENT"
    assert measurement_document["evidence_id"] == "measurement_001"
    assert measurement_document["measurements"] == {
        "quantity": "voltage",
        "value": 4.927,
        "unit": "V",
        "measurement_point": "board_supply",
    }

    second_decision = json.loads(second_decision_path.read_text(encoding="utf-8"))

    assert second_decision["iteration"] == 2
    assert second_decision["action"]["action_type"] == "REQUEST_OBSERVATION"
    assert second_decision["evidence_ids"] == ["measurement_001"]


def test_demo_command_can_stop_after_first_agent_iteration(
    tmp_path: Path,
) -> None:
    """Demo should stop safely when no measurement is supplied."""

    image_path = tmp_path / "board.jpg"
    output_directory = tmp_path / "demo"

    image = np.full(
        (600, 800, 3),
        128,
        dtype=np.uint8,
    )

    assert cv2.imwrite(str(image_path), image)

    exit_code = main(
        [
            "demo",
            str(image_path),
            "--output",
            str(output_directory),
        ]
    )

    assert exit_code == 0

    assert (output_directory / "result.json").is_file()
    assert (output_directory / "decision_iteration_1.json").is_file()

    assert not (output_directory / "measurement.json").exists()
    assert not (output_directory / "decision_iteration_2.json").exists()

    report_path = output_directory / "inspection_report.json"

    assert report_path.is_file()

    report_document = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_document["measurement"] is None
    assert report_document["iteration_2"] is None
    assert report_document["status"] == "ADDITIONAL_EVIDENCE_REQUIRED"

def test_cloud_publish_uses_memory_provider(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    artifact_directory = tmp_path / "demo"
    artifact_directory.mkdir()

    report = {
        "report_version": "1.0",
        "inspection": {
            "board_detected": True,
            "localization_confidence": 0.863,
            "visual_evidence_count": 15,
            "visual_region_count": 12,
        },
        "iteration_1": {
            "iteration": 1,
            "action": "REQUEST_MEASUREMENT",
            "reason": "Visual evidence requires an electrical measurement.",
            "instruction": "Measure the PCB supply voltage.",
            "target_roi_id": "roi_012",
            "evidence_ids": ["evidence_roi_012"],
        },
        "measurement": {
            "quantity": "voltage",
            "value": 4.927,
            "unit": "V",
            "measurement_point": "board_supply",
        },
        "iteration_2": {
            "iteration": 2,
            "action": "REQUEST_OBSERVATION",
            "reason": "Expected reference voltage is unavailable.",
            "instruction": "Provide the expected supply voltage.",
            "target_roi_id": None,
            "evidence_ids": ["measurement_001"],
        },
        "status": "ADDITIONAL_EVIDENCE_REQUIRED",
    }

    (artifact_directory / "inspection_report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    (artifact_directory / "annotated.jpg").write_bytes(b"annotated")
    (artifact_directory / "debug_contours.jpg").write_bytes(b"debug")

    exit_code = main(
        [
            "cloud",
            "publish",
            str(artifact_directory),
            "--provider",
            "memory",
            "--session-id",
            "inspection-cli-test",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "RepairMind Edge cloud publication complete" in captured.out
    assert "Session:   inspection-cli-test" in captured.out
    assert "Provider:  memory" in captured.out
    assert "Artifacts: 2" in captured.out
    assert "inspection_report.json" in captured.out
    assert "annotated.jpg" in captured.out
    assert "debug_contours.jpg" not in captured.out


def test_cloud_publish_rejects_missing_report(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    artifact_directory = tmp_path / "demo"
    artifact_directory.mkdir()

    exit_code = main(
        [
            "cloud",
            "publish",
            str(artifact_directory),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 3
    assert "could not build cloud session" in captured.err
    assert "Inspection report not found" in captured.err


def test_cloud_publish_rejects_missing_directory(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "cloud",
            "publish",
            str(tmp_path / "missing"),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 3
    assert "could not build cloud session" in captured.err
    assert "Artifact directory not found" in captured.err
