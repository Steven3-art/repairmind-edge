"""Tests for RepairMind Edge cloud models."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from repairmind_edge.cloud_models import (
    CloudArtifact,
    CloudArtifactType,
    InspectionSessionManifest,
)


def test_cloud_artifact_accepts_valid_sha256() -> None:
    artifact = CloudArtifact(
        artifact_type=CloudArtifactType.INSPECTION_REPORT,
        filename="inspection_report.json",
        content_type="application/json",
        sha256="a" * 64,
        size_bytes=128,
    )

    assert artifact.filename == "inspection_report.json"
    assert artifact.sha256 == "a" * 64


def test_cloud_artifact_rejects_invalid_sha256() -> None:
    with pytest.raises(ValidationError):
        CloudArtifact(
            artifact_type=CloudArtifactType.INSPECTION_REPORT,
            filename="inspection_report.json",
            content_type="application/json",
            sha256="invalid",
            size_bytes=128,
        )


def test_inspection_session_manifest_is_immutable() -> None:
    manifest = InspectionSessionManifest(
        session_id="session-001",
        created_at=datetime(2026, 10, 5, tzinfo=UTC),
        report_version="1.0",
        inspection_status="ADDITIONAL_EVIDENCE_REQUIRED",
        artifacts=(),
    )

    with pytest.raises(ValidationError):
        manifest.session_id = "changed"  # type: ignore[misc]