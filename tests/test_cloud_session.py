"""Tests for RepairMind Edge cloud session construction."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from repairmind_edge.cloud_models import CloudArtifactType
from repairmind_edge.cloud_session import (
    build_cloud_session,
    is_generated_session_id,
)


def _write_report(
    directory: Path,
    *,
    status: str = "ADDITIONAL_EVIDENCE_REQUIRED",
) -> None:
    """Write a valid inspection report fixture."""

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
            "reason": (
                "Visual evidence alone cannot establish electrical behavior."
            ),
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
        "status": status,
    }

    (directory / "inspection_report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )


def test_build_cloud_session_selects_only_allowlisted_artifacts(
    tmp_path: Path,
) -> None:
    """Only explicitly approved artifacts should enter a cloud session."""

    _write_report(tmp_path)

    annotated = b"annotated-image"
    normalized = b"normalized-image"
    overlay = b"roi-overlay"

    (tmp_path / "annotated.jpg").write_bytes(annotated)
    (tmp_path / "normalized.jpg").write_bytes(normalized)
    (tmp_path / "roi_overlay.jpg").write_bytes(overlay)

    # Diagnostic output must never be published automatically.
    (tmp_path / "debug_contours.jpg").write_bytes(b"debug")

    # Arbitrary files must also remain outside the publication boundary.
    (tmp_path / "secret.txt").write_text(
        "must-not-be-published",
        encoding="utf-8",
    )

    created_at = datetime(
        2026,
        10,
        5,
        8,
        0,
        tzinfo=UTC,
    )

    manifest, payloads = build_cloud_session(
        tmp_path,
        session_id="inspection-test-001",
        created_at=created_at,
    )

    assert manifest.session_id == "inspection-test-001"
    assert manifest.created_at == created_at
    assert manifest.report_version == "1.0"
    assert (
        manifest.inspection_status
        == "ADDITIONAL_EVIDENCE_REQUIRED"
    )

    assert [
        artifact.filename
        for artifact in manifest.artifacts
    ] == [
        "inspection_report.json",
        "annotated.jpg",
        "normalized.jpg",
        "roi_overlay.jpg",
    ]

    assert set(payloads) == {
        "inspection_report.json",
        "annotated.jpg",
        "normalized.jpg",
        "roi_overlay.jpg",
    }

    assert "debug_contours.jpg" not in payloads
    assert "secret.txt" not in payloads

    annotated_metadata = next(
        artifact
        for artifact in manifest.artifacts
        if artifact.artifact_type
        == CloudArtifactType.ANNOTATED_IMAGE
    )

    assert annotated_metadata.size_bytes == len(annotated)
    assert annotated_metadata.sha256 == hashlib.sha256(
        annotated
    ).hexdigest()


def test_build_cloud_session_requires_report(
    tmp_path: Path,
) -> None:
    """A cloud session cannot exist without its inspection report."""

    with pytest.raises(
        FileNotFoundError,
        match="Inspection report not found",
    ):
        build_cloud_session(tmp_path)


def test_build_cloud_session_rejects_invalid_report(
    tmp_path: Path,
) -> None:
    """Malformed or schema-invalid reports must not be published."""

    (tmp_path / "inspection_report.json").write_text(
        '{"invalid": true}',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid inspection report",
    ):
        build_cloud_session(tmp_path)


def test_build_cloud_session_accepts_missing_optional_images(
    tmp_path: Path,
) -> None:
    """The validated report is mandatory while visual artifacts are optional."""

    _write_report(tmp_path)

    manifest, payloads = build_cloud_session(
        tmp_path,
        session_id="inspection-report-only",
        created_at=datetime(
            2026,
            10,
            5,
            tzinfo=UTC,
        ),
    )

    assert len(manifest.artifacts) == 1
    assert (
        manifest.artifacts[0].artifact_type
        == CloudArtifactType.INSPECTION_REPORT
    )
    assert set(payloads) == {"inspection_report.json"}


def test_build_cloud_session_generates_uuid_session_id(
    tmp_path: Path,
) -> None:
    """Generated session identifiers should use the RepairMind UUID format."""

    _write_report(tmp_path)

    manifest, _ = build_cloud_session(tmp_path)

    assert is_generated_session_id(manifest.session_id)


def test_build_cloud_session_rejects_empty_session_id(
    tmp_path: Path,
) -> None:
    """Explicit session identifiers must contain meaningful content."""

    _write_report(tmp_path)

    with pytest.raises(
        ValueError,
        match="session_id must not be empty",
    ):
        build_cloud_session(
            tmp_path,
            session_id="   ",
        )


def test_build_cloud_session_rejects_naive_timestamp(
    tmp_path: Path,
) -> None:
    """Cloud session timestamps must include timezone information."""

    _write_report(tmp_path)

    with pytest.raises(
        ValueError,
        match="created_at must be timezone-aware",
    ):
        build_cloud_session(
            tmp_path,
            created_at=datetime(
                2026,
                10,
                5,
            ),
        )